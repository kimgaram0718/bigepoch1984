# predict_info/management/commands/generate_daily_predictions.py
import time
import traceback
from datetime import datetime, timedelta, date as date_type

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import tensorflow as tf
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from pandas.tseries.offsets import BDay # Business Day
import holidays # For calculating trading days

# 앱 내부 모듈 임포트
from predict_info.models import PredictedStockPrice, StockPrice
from predict_info.utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from predict_info.views import ( # views.py의 헬퍼 함수들을 최대한 활용
    ML_MODELS_DIR, TIME_STEPS, FUTURE_TARGET_DAYS, MIN_DATA_DAYS_FOR_PREDICT,
    get_feature_columns_for_market, load_model_and_scalers,
    get_krx_stock_list_predict_cached, get_future_trading_dates_list
)

# 이 커맨드에서 사용할 상수
DAYS_TO_KEEP_PREDICTIONS = 14 # 최근 14일치 예측만 보관

class Command(BaseCommand):
    help = 'Generates daily stock price predictions for all listed KOSPI and KOSDAQ stocks and saves them to the database.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--markets',
            type=str,
            default='KOSPI,KOSDAQ',
            help='Comma-separated list of markets to process (e.g., KOSPI,KOSDAQ). Default is KOSPI,KOSDAQ.'
        )
        parser.add_argument(
            '--analysis_type',
            type=str,
            default='technical',
            help='The type of analysis model to use for prediction (e.g., technical). Default is technical.'
        )
        parser.add_argument(
            '--delete_old_after_days',
            type=int,
            default=DAYS_TO_KEEP_PREDICTIONS,
            help=f'Number of days to keep old predictions. Default is {DAYS_TO_KEEP_PREDICTIONS}.'
        )

    def _get_stock_data_for_prediction(self, stock_code, stock_name, market_name_upper, feature_names_for_model_input):
        """
        views.py의 get_latest_stock_data_with_features와 유사하지만,
        이 커맨드 실행 시점(보통 장 마감 후)의 데이터를 기준으로 함.
        이 함수는 StockPrice DB에서 데이터를 가져오도록 수정될 수 있으나,
        초기 버전에서는 FDR을 통해 최신 데이터를 가져오는 views.py의 로직을 일부 차용합니다.
        실제 운영 시에는 update_daily_data 커맨드가 먼저 실행되어 StockPrice DB가 최신 상태임을 가정합니다.
        """
        if not PANDAS_TA_AVAILABLE:
            self.stderr.write(f"[CRITICAL] pandas_ta library is not available. Cannot generate features for {stock_name}.")
            return None, None

        try:
            self.stdout.write(f"  Fetching and preparing data for {stock_name}({stock_code}) [{market_name_upper}]...")
            
            # 예측 기준일은 "오늘" (스크립트 실행일)
            # 이 날짜까지의 데이터로 예측을 수행. 실제 데이터는 전 거래일까지 있을 것.
            prediction_execution_date = timezone.now().date()

            # StockPrice 모델에서 데이터 조회
            # 최소 MIN_DATA_DAYS_FOR_PREDICT + 여유분(TIME_STEPS + 버퍼) 만큼의 데이터 필요
            required_data_start_date = prediction_execution_date - timedelta(days=MIN_DATA_DAYS_FOR_PREDICT + TIME_STEPS + 60) # 넉넉하게 60일 추가 버퍼
            
            stock_prices_qs = StockPrice.objects.filter(
                stock_code=stock_code,
                date__gte=required_data_start_date,
                date__lte=prediction_execution_date # 기준일 당일까지의 데이터 포함 (실제로는 전 거래일까지 있을 것)
            ).order_by('date')

            if not stock_prices_qs.exists() or stock_prices_qs.count() < (TIME_STEPS + 30): # 최소 데이터 길이 체크
                self.stderr.write(f"  [WARNING] Insufficient historical data in DB for {stock_name}({stock_code}). Found {stock_prices_qs.count()} records. Skipping.")
                return None, None

            df_ohlcv_raw = pd.DataFrame.from_records(stock_prices_qs.values(
                'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 
                'Indi', 'Foreign', 'Organ' # 투자자 정보도 StockPrice에 있다면 가져옴
            ))
            df_ohlcv_raw.rename(columns={'date': 'Date'}, inplace=True)
            df_ohlcv_raw['Date'] = pd.to_datetime(df_ohlcv_raw['Date'])
            df_ohlcv_raw.set_index('Date', inplace=True)

            # 마지막 데이터 날짜 (이 날짜를 기준으로 미래 예측)
            last_data_date_in_db = df_ohlcv_raw.index.max().date()
            
            # 시장 지수 및 환율 데이터 (DB 데이터 기간에 맞춰서)
            fetch_start_date_for_others = df_ohlcv_raw.index.min().date()
            fetch_end_date_for_others = last_data_date_in_db

            market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
            df_market_index, df_macro_fx = get_market_macro_data(
                fetch_start_date_for_others, 
                fetch_end_date_for_others, 
                market_fdr_code=market_fdr_code_param
            )

            df_merged = df_ohlcv_raw.copy()
            if not df_market_index.empty: df_merged = df_merged.join(df_market_index, how='left')
            else:
                market_cols_to_add = [col for col in feature_names_for_model_input if market_name_upper in col and ("_Close" in col or "_Change" in col)]
                for col in market_cols_to_add: df_merged[col] = np.nan
            
            if not df_macro_fx.empty: df_merged = df_merged.join(df_macro_fx, how='left')
            else:
                for col in ['USD_KRW_Close', 'USD_KRW_Change']: df_merged[col] = np.nan # views.py의 MACRO_DATA_COLS 참고
            
            # 투자자 정보는 df_ohlcv_raw에서 이미 가져왔으므로 별도 병합 불필요 (StockPrice에 Indi, Foreign, Organ 컬럼이 있다고 가정)
            # 만약 StockPrice에 없다면 여기서 0으로 채우거나 다른 소스에서 가져와야 함.
            # 현재 StockPrice 모델에는 해당 컬럼이 없으므로, 임시로 0으로 채움 (views.py와 동일하게)
            for col in ['Indi', 'Foreign', 'Organ']: # views.py의 INVESTOR_COLS_STANDARD 참고
                if col not in df_merged.columns:
                    df_merged[col] = 0.0


            df_merged.ffill(inplace=True)
            
            df_with_all_ta = calculate_all_features(df_merged.copy(), market_name_upper=market_name_upper)
            
            missing_in_df_after_ta = []
            for col in feature_names_for_model_input:
                if col not in df_with_all_ta.columns:
                    self.stderr.write(f"    [ERROR] Feature '{col}' missing after TA calculation for {stock_name}. Adding as NaN.")
                    df_with_all_ta[col] = np.nan
                    missing_in_df_after_ta.append(col)
            
            if missing_in_df_after_ta:
                 self.stderr.write(f"    [ERROR] Missing features forced to NaN for {stock_name}: {missing_in_df_after_ta}. Prediction accuracy may be affected.")

            df_selected_features = df_with_all_ta[feature_names_for_model_input].copy()
            df_selected_features.ffill(inplace=True)
            df_selected_features.bfill(inplace=True)
            df_selected_features.fillna(0.0, inplace=True)

            if len(df_selected_features) < TIME_STEPS:
                self.stderr.write(f"  [ERROR] Insufficient final feature data length for {stock_name}: {len(df_selected_features)} (need {TIME_STEPS}).")
                return None, None

            recent_features_df = df_selected_features.tail(TIME_STEPS)
            
            # 이 예측의 기준이 되는 날짜는 DB 상의 마지막 데이터 날짜
            return recent_features_df, last_data_date_in_db

        except Exception as e:
            self.stderr.write(f"  [CRITICAL] Error preparing data for {stock_name}({stock_code}): {e}")
            self.stderr.write(traceback.format_exc())
            return None, None

    def handle(self, *args, **options):
        start_time_total = time.time()
        self.stdout.write(self.style.SUCCESS(f"Starting daily prediction generation at {timezone.now()}..."))

        markets_to_run = [m.strip().upper() for m in options['markets'].split(',') if m.strip()]
        analysis_type_to_run = options['analysis_type'].lower()
        delete_older_than_days = options['delete_old_after_days']

        if not PANDAS_TA_AVAILABLE:
            self.stderr.write(self.style.ERROR("pandas_ta library is not available. Predictions cannot be generated."))
            return

        all_krx_stocks = get_krx_stock_list_predict_cached()
        if not all_krx_stocks:
            self.stderr.write(self.style.ERROR("Could not fetch KRX stock list. Aborting."))
            return

        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # 예측 기준일은 이 커맨드가 실행되는 "오늘"
        # 실제 데이터는 어제까지의 데이터로 예측하고, 예측일은 내일부터 시작.
        # DB에 저장될 prediction_base_date는 이 예측에 사용된 데이터의 마지막 날짜가 되어야 함.
        # _get_stock_data_for_prediction 에서 반환하는 last_data_date_in_db 를 사용.

        for market_name_filter in markets_to_run:
            if market_name_filter not in ["KOSPI", "KOSDAQ"]:
                self.stderr.write(self.style.WARNING(f"Unsupported market '{market_name_filter}' found in --markets option. Skipping."))
                continue

            self.stdout.write(self.style.SUCCESS(f"\nProcessing {market_name_filter} market..."))
            
            model_key = f"{market_name_filter.lower()}_{analysis_type_to_run}"
            selected_model, selected_scaler_X, selected_scaler_y = load_model_and_scalers(model_key)

            if not all([selected_model, selected_scaler_X, selected_scaler_y]):
                self.stderr.write(self.style.ERROR(f"Failed to load model/scalers for {model_key}. Skipping {market_name_filter} market."))
                continue

            current_market_feature_columns = get_feature_columns_for_market(market_name_filter)
            
            stocks_in_market = [s for s in all_krx_stocks if s.get('market', '').upper().startswith(market_name_filter)]
            
            if not stocks_in_market:
                self.stdout.write(f"No stocks found for market {market_name_filter} in the cached list.")
                continue

            market_stock_count = len(stocks_in_market)
            self.stdout.write(f"Found {market_stock_count} stocks in {market_name_filter} market.")

            for i, stock_info in enumerate(stocks_in_market):
                stock_code = stock_info.get('code')
                stock_name = stock_info.get('name')
                
                self.stdout.write(f"  ({i+1}/{market_stock_count}) Predicting for {stock_name} ({stock_code})...")

                # 1. 데이터 준비
                recent_features_df, actual_prediction_base_date = self._get_stock_data_for_prediction(
                    stock_code, stock_name, market_name_filter, current_market_feature_columns
                )

                if recent_features_df is None or actual_prediction_base_date is None:
                    self.stderr.write(f"    Could not prepare data for {stock_name}. Skipping.")
                    skipped_count += 1
                    continue
                
                if len(recent_features_df.columns) != len(current_market_feature_columns) or recent_features_df.isnull().values.any():
                    self.stderr.write(f"    Feature mismatch or NaN found in final data for {stock_name}. Skipping. Check logs.")
                    error_count +=1
                    continue

                # 2. 예측 수행
                try:
                    input_data_for_scaling = recent_features_df[current_market_feature_columns].values
                    input_data_scaled = selected_scaler_X.transform(input_data_for_scaling)
                    input_data_reshaped = input_data_scaled.reshape(1, TIME_STEPS, len(current_market_feature_columns))
                    
                    prediction_scaled = selected_model.predict(input_data_reshaped, verbose=0)
                    prediction_actual_prices_raw = selected_scaler_y.inverse_transform(prediction_scaled)[0]
                    
                    future_dates_dt = get_future_trading_dates_list(actual_prediction_base_date, FUTURE_TARGET_DAYS)

                    if len(future_dates_dt) != FUTURE_TARGET_DAYS:
                        self.stderr.write(f"    Error calculating future trading dates for {stock_name} based on {actual_prediction_base_date}. Skipping.")
                        error_count += 1
                        continue
                    
                    model_name_or_path = getattr(selected_model, 'name', "") or getattr(selected_model, 'filepath', "")
                    model_was_log_trained = "_log_" in model_name_or_path.lower()

                    # 3. 예측 결과 DB 저장
                    predictions_to_save = []
                    current_base_price_for_clipping = recent_features_df['Close'].iloc[-1]

                    for day_idx in range(FUTURE_TARGET_DAYS):
                        pred_price_day_i_scaled = prediction_actual_prices_raw[day_idx]
                        final_pred_price_day_i = np.expm1(pred_price_day_i_scaled) if model_was_log_trained else pred_price_day_i_scaled
                        
                        upper_limit = current_base_price_for_clipping * 1.30 # 상한가 30%
                        lower_limit = current_base_price_for_clipping * 0.70 # 하한가 30%
                        clipped_price = np.clip(final_pred_price_day_i, lower_limit, upper_limit)
                        
                        predictions_to_save.append({
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'market_name': market_name_filter,
                            'prediction_base_date': actual_prediction_base_date,
                            'predicted_date': future_dates_dt[day_idx],
                            'predicted_price': round(float(clipped_price)),
                            'analysis_type': analysis_type_to_run,
                        })
                        current_base_price_for_clipping = clipped_price # 다음날 클리핑 기준은 전날 예측가

                    # 기존 예측 삭제 후 새로 저장 (또는 update_or_create 사용)
                    # 여기서는 해당 prediction_base_date에 대한 예측을 한번에 삭제하고 새로 insert
                    PredictedStockPrice.objects.filter(
                        stock_code=stock_code,
                        prediction_base_date=actual_prediction_base_date,
                        analysis_type=analysis_type_to_run
                    ).delete()
                    
                    PredictedStockPrice.objects.bulk_create([
                        PredictedStockPrice(**data) for data in predictions_to_save
                    ])
                    self.stdout.write(self.style.SUCCESS(f"    Successfully predicted and saved for {stock_name} ({stock_code}) based on {actual_prediction_base_date}."))
                    processed_count += 1

                except Exception as e_pred:
                    self.stderr.write(self.style.ERROR(f"    Error during prediction or saving for {stock_name} ({stock_code}): {e_pred}"))
                    self.stderr.write(traceback.format_exc())
                    error_count += 1
                
                time.sleep(0.1) # 각 종목 처리 후 약간의 딜레이 (DB 부하 감소)

        self.stdout.write(self.style.SUCCESS(f"\n--- Prediction Generation Summary ---"))
        self.stdout.write(f"Total stocks processed (attempted): {processed_count + skipped_count + error_count}")
        self.stdout.write(f"Successfully predicted and saved: {processed_count}")
        self.stdout.write(f"Skipped (e.g., insufficient data): {skipped_count}")
        self.stdout.write(f"Errors during processing: {error_count}")

        # 4. 오래된 예측 데이터 삭제
        if delete_older_than_days > 0:
            cutoff_date = timezone.now().date() - timedelta(days=delete_older_than_days)
            self.stdout.write(f"\nDeleting predictions with prediction_base_date older than {cutoff_date} (keeping last {delete_older_than_days} days)...")
            deleted_info = PredictedStockPrice.objects.filter(prediction_base_date__lt=cutoff_date).delete()
            deleted_count = deleted_info[0] if isinstance(deleted_info, tuple) else 0
            self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} old prediction records."))
        else:
            self.stdout.write(f"Old prediction deletion skipped as --delete_old_after_days is not positive ({delete_older_than_days}).")

        end_time_total = time.time()
        self.stdout.write(self.style.SUCCESS(f"Daily prediction generation finished in {end_time_total - start_time_total:.2f} seconds."))

