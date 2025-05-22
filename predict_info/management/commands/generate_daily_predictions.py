# predict_info/management/commands/generate_daily_predictions.py
import time
import traceback
from datetime import datetime, timedelta, date as date_type

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
# import tensorflow as tf # load_model_and_scalers 함수에서 임포트하므로 여기서는 제거 가능
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
# from pandas.tseries.offsets import BDay # 현재 직접 사용되지 않음
# import holidays # 현재 직접 사용되지 않음

from predict_info.models import PredictedStockPrice, StockPrice
from predict_info.utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from predict_info.views import ( 
    ML_MODELS_DIR, TIME_STEPS, FUTURE_TARGET_DAYS, MIN_DATA_DAYS_FOR_PREDICT,
    get_feature_columns_for_market, load_model_and_scalers,
    get_krx_stock_list_predict_cached, get_future_trading_dates_list
)

DAYS_TO_KEEP_PREDICTIONS = 14 
# DB 필드명 (StockPrice 모델 기준)
BASE_OHLCV_COLS_DB = ['open_price', 'high_price', 'low_price', 'close_price', 'volume'] # 소문자로 변경
INVESTOR_COLS_DB = ['indi_volume', 'foreign_volume', 'organ_volume'] 
FUNDAMENTAL_COLS_DB = ['market_cap', 'per', 'pbr'] 

# Pandas DataFrame 및 모델 입력 시 사용할 표준 컬럼명 (views.py의 get_feature_columns_for_market 와 일관성 유지)
BASE_OHLCV_COLS_DF = ['Open', 'High', 'Low', 'Close', 'Volume'] # 첫 글자 대문자
INVESTOR_COLS_MODEL_INPUT = ['Indi', 'Foreign', 'Organ']
FUNDAMENTAL_COLS_MODEL_INPUT = ['Marcap', 'PER', 'PBR']


class Command(BaseCommand):
    help = 'Generates daily stock price predictions for all listed KOSPI and KOSDAQ stocks and saves them to the database.'

    def add_arguments(self, parser):
        parser.add_argument('--markets', type=str, default='KOSPI,KOSDAQ', help='Comma-separated list of markets (KOSPI,KOSDAQ).')
        parser.add_argument('--analysis_type', type=str, default='technical', help='Analysis model type (e.g., technical).')
        parser.add_argument('--delete_old_after_days', type=int, default=DAYS_TO_KEEP_PREDICTIONS, help=f'Days to keep old predictions. Default: {DAYS_TO_KEEP_PREDICTIONS}.')

    def _get_stock_data_for_prediction(self, stock_code, stock_name, market_name_upper, feature_names_for_model_input):
        self.stdout.write(f"  [DATA_PREP] Preparing data for: {stock_name}({stock_code}) - Market: {market_name_upper}")
        if not PANDAS_TA_AVAILABLE:
            self.stderr.write(f"    [CRITICAL_ERROR] pandas_ta library unavailable. Cannot generate features for {stock_name}.")
            return None, None
        try:
            prediction_execution_date = timezone.now().date()
            required_data_start_date = prediction_execution_date - timedelta(days=MIN_DATA_DAYS_FOR_PREDICT + TIME_STEPS + 90) 
            
            self.stdout.write(f"    [DATA_PREP] Fetching data from DB for {stock_code} between {required_data_start_date} and {prediction_execution_date}")

            # DB에서 가져올 컬럼 목록 (StockPrice 모델 필드명과 일치)
            db_columns_to_fetch = ['date'] + BASE_OHLCV_COLS_DB + \
                                  INVESTOR_COLS_DB + FUNDAMENTAL_COLS_DB
            
            stock_prices_qs = StockPrice.objects.filter(
                stock_code=stock_code,
                date__gte=required_data_start_date,
                date__lte=prediction_execution_date 
            ).order_by('date').values(*db_columns_to_fetch)

            if not stock_prices_qs.exists():
                self.stderr.write(f"    [WARNING] No historical data in DB for {stock_name}({stock_code}) in range. Skipping.")
                return None, None
            
            df_from_db = pd.DataFrame.from_records(stock_prices_qs)
            self.stdout.write(f"    [DATA_PREP] Fetched {len(df_from_db)} records from DB for {stock_code}.")

            if len(df_from_db) < (TIME_STEPS + 30): 
                self.stderr.write(f"    [WARNING] Insufficient DB data for {stock_name}({stock_code}). Found {len(df_from_db)}, need {TIME_STEPS + 30}. Skipping.")
                return None, None

            # Date 컬럼명 변경 및 타입 변환
            df_from_db.rename(columns={'date': 'Date'}, inplace=True)
            df_from_db['Date'] = pd.to_datetime(df_from_db['Date'])
            df_from_db.set_index('Date', inplace=True)

            # DB 컬럼명(소문자) -> Pandas DataFrame 및 모델 입력용 표준 컬럼명(대문자 시작)으로 변경
            db_to_df_col_map = {}
            for db_col, df_col in zip(BASE_OHLCV_COLS_DB, BASE_OHLCV_COLS_DF):
                db_to_df_col_map[db_col] = df_col
            for db_col, model_col in zip(INVESTOR_COLS_DB, INVESTOR_COLS_MODEL_INPUT):
                db_to_df_col_map[db_col] = model_col
            for db_col, model_col in zip(FUNDAMENTAL_COLS_DB, FUNDAMENTAL_COLS_MODEL_INPUT):
                db_to_df_col_map[db_col] = model_col
            
            df_from_db.rename(columns=db_to_df_col_map, inplace=True)

            # 이제 DataFrame의 컬럼명은 'Open', 'High', 'Low', 'Close', 'Volume', 'Indi', 'Marcap' 등 표준화된 이름임
            for col in BASE_OHLCV_COLS_DF: # 표준화된 OHLCV 컬럼명으로 타입 확인
                if col not in df_from_db.columns:
                    self.stderr.write(f"    [CRITICAL_ERROR] Column '{col}' missing after renaming from DB data for {stock_name}. Skipping.")
                    return None, None
                df_from_db[col] = pd.to_numeric(df_from_db[col], errors='coerce')
            
            df_from_db['Change'] = df_from_db['Close'].pct_change() # 'Close' (대문자) 사용

            for col_list in [INVESTOR_COLS_MODEL_INPUT, FUNDAMENTAL_COLS_MODEL_INPUT]:
                for col_name in col_list:
                    if col_name in df_from_db.columns:
                        df_from_db[col_name] = pd.to_numeric(df_from_db[col_name], errors='coerce').fillna(0.0)
                    else: 
                        self.stdout.write(f"    [DATA_PREP] Column '{col_name}' not found after renaming, initializing to 0.0 for {stock_code}.")
                        df_from_db[col_name] = 0.0
            
            last_data_date_in_df = df_from_db.index.max().date()
            self.stdout.write(f"    [DATA_PREP] Last data date for {stock_code} is {last_data_date_in_df}.")
            
            fetch_start_date_for_others = df_from_db.index.min().date()
            fetch_end_date_for_others = last_data_date_in_df

            market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
            df_market_index, df_macro_fx = get_market_macro_data(fetch_start_date_for_others, fetch_end_date_for_others, market_fdr_code=market_fdr_code_param)
            
            df_merged = df_from_db.copy() 
            
            market_specific_cols_expected = [col for col in feature_names_for_model_input if market_name_upper in col and ("_Close" in col or "_Change" in col)]
            if not df_market_index.empty: df_merged = df_merged.join(df_market_index, how='left')
            else: 
                for col in market_specific_cols_expected: df_merged[col] = np.nan
            
            macro_cols_expected = ['USD_KRW_Close', 'USD_KRW_Change'] # views.py의 MACRO_DATA_COLS와 일치
            if not df_macro_fx.empty: df_merged = df_merged.join(df_macro_fx, how='left')
            else: 
                for col in macro_cols_expected: df_merged[col] = np.nan
            
            df_merged.ffill(inplace=True)

            df_with_all_ta = calculate_all_features(df_merged.copy(), market_name_upper=market_name_upper)
            self.stdout.write(f"    [DATA_PREP] Calculated TAs for {stock_code}. Shape: {df_with_all_ta.shape}")

            missing_features_after_ta = []
            for col in feature_names_for_model_input: 
                if col not in df_with_all_ta.columns:
                    self.stderr.write(f"      [CRITICAL_ERROR] Feature '{col}' MISSING for {stock_name} after TA. Adding as NaN.")
                    df_with_all_ta[col] = np.nan 
                    missing_features_after_ta.append(col)
            
            if missing_features_after_ta:
                 self.stderr.write(f"    [CRITICAL_ERROR] Critical features missing for {stock_name}: {missing_features_after_ta}.")

            df_selected_features = df_with_all_ta[feature_names_for_model_input].copy()
            df_selected_features.ffill(inplace=True).bfill(inplace=True) 
            
            nan_cols_final_check = df_selected_features.columns[df_selected_features.isnull().any()].tolist()
            if nan_cols_final_check:
                self.stderr.write(f"    [WARNING] NaN values in final features for {stock_name} after ffill/bfill: {nan_cols_final_check}. Filling with 0.0.")
                df_selected_features.fillna(0.0, inplace=True)
            
            if df_selected_features.isnull().values.any():
                self.stderr.write(f"    [CRITICAL_ERROR] Unrecoverable NaN in final features for {stock_name}. Skipping.")
                return None, None

            if len(df_selected_features) < TIME_STEPS:
                self.stderr.write(f"    [ERROR] Insufficient final feature data for {stock_name}: {len(df_selected_features)} (need {TIME_STEPS}). Skipping.")
                return None, None

            recent_features_df = df_selected_features.tail(TIME_STEPS)
            self.stdout.write(f"    [DATA_PREP] Successfully prepared final features for {stock_name}. Shape: {recent_features_df.shape}. Last data date: {last_data_date_in_df}")
            return recent_features_df, last_data_date_in_df
        except Exception as e:
            self.stderr.write(f"    [CRITICAL_ERROR] Exception during data prep for {stock_name}({stock_code}): {e}")
            self.stderr.write(traceback.format_exc())
            return None, None

    def handle(self, *args, **options):
        start_time_total = time.time()
        self.stdout.write(self.style.SUCCESS(f"Starting daily prediction generation at {timezone.now()}..."))

        markets_to_run = [m.strip().upper() for m in options['markets'].split(',') if m.strip()]
        analysis_type_to_run = options['analysis_type'].lower()
        delete_older_than_days = options['delete_old_after_days']

        if not PANDAS_TA_AVAILABLE:
            self.stderr.write(self.style.ERROR("pandas_ta library unavailable. Predictions cannot be generated."))
            return

        all_krx_stocks = get_krx_stock_list_predict_cached()
        if not all_krx_stocks:
            self.stderr.write(self.style.ERROR("Could not fetch KRX stock list. Aborting."))
            return

        processed_count, skipped_count, error_count = 0, 0, 0
        
        for market_name_filter in markets_to_run:
            if market_name_filter not in ["KOSPI", "KOSDAQ"]:
                self.stderr.write(self.style.WARNING(f"Unsupported market '{market_name_filter}'. Skipping."))
                continue

            self.stdout.write(self.style.SUCCESS(f"\nProcessing {market_name_filter} market..."))
            model_key = f"{market_name_filter.lower()}_{analysis_type_to_run}"
            selected_model, selected_scaler_X, selected_scaler_y = load_model_and_scalers(model_key)

            if not all([selected_model, selected_scaler_X, selected_scaler_y]):
                self.stderr.write(self.style.ERROR(f"Failed to load model/scalers for {model_key}. Skipping {market_name_filter}."))
                error_count +=1
                continue

            try:
                current_market_feature_columns = get_feature_columns_for_market(market_name_filter)
            except ValueError as e_feat:
                self.stderr.write(self.style.ERROR(f"Error getting feature columns for {market_name_filter}: {e_feat}. Skipping market."))
                error_count +=1
                continue
            
            stocks_in_market = [s for s in all_krx_stocks if s.get('market', '').upper().startswith(market_name_filter)]
            if not stocks_in_market:
                self.stdout.write(f"No stocks found for market {market_name_filter}.")
                continue

            market_stock_count = len(stocks_in_market)
            self.stdout.write(f"Found {market_stock_count} stocks in {market_name_filter}.")

            for i, stock_info in enumerate(stocks_in_market):
                stock_code, stock_name = stock_info.get('code'), stock_info.get('name')
                if not stock_code or not stock_name:
                    self.stderr.write(f"  Skipping stock with missing info: {stock_info}"); skipped_count +=1; continue
                
                self.stdout.write(f"  ({i+1}/{market_stock_count}) Processing: {stock_name} ({stock_code})")
                recent_features_df, actual_prediction_base_date = self._get_stock_data_for_prediction(
                    stock_code, stock_name, market_name_filter, current_market_feature_columns
                )

                if recent_features_df is None or actual_prediction_base_date is None:
                    self.stderr.write(f"    Data preparation failed for {stock_name}. Skipping."); skipped_count += 1; continue
                
                if len(recent_features_df.columns) != len(current_market_feature_columns):
                    self.stderr.write(f"    [CRITICAL_ERROR] Feature column count mismatch for {stock_name}. Expected {len(current_market_feature_columns)}, got {len(recent_features_df.columns)}. Skipping.")
                    error_count +=1; continue
                if recent_features_df.isnull().values.any():
                    nan_cols_in_final_df = recent_features_df.columns[recent_features_df.isnull().any()].tolist()
                    self.stderr.write(f"    [CRITICAL_ERROR] NaN values in final features for {stock_name} before scaling: {nan_cols_in_final_df}. Skipping.")
                    error_count +=1; continue

                try:
                    input_data_for_scaling = recent_features_df[current_market_feature_columns].values 
                    input_data_scaled = selected_scaler_X.transform(input_data_for_scaling)
                    input_data_reshaped = input_data_scaled.reshape(1, TIME_STEPS, len(current_market_feature_columns))
                    prediction_scaled = selected_model.predict(input_data_reshaped, verbose=0)
                    
                    if prediction_scaled.shape != (1, FUTURE_TARGET_DAYS):
                         self.stderr.write(f"    [ERROR] Unexpected prediction_scaled shape for {stock_name}: {prediction_scaled.shape}. Expected (1, {FUTURE_TARGET_DAYS}). Skipping.")
                         error_count += 1; continue
                    
                    prediction_reshaped_for_scaler = prediction_scaled.reshape(-1, 1)
                    prediction_actual_prices_raw = selected_scaler_y.inverse_transform(prediction_reshaped_for_scaler).flatten()
                    future_dates_dt = get_future_trading_dates_list(actual_prediction_base_date, FUTURE_TARGET_DAYS)

                    if len(future_dates_dt) != FUTURE_TARGET_DAYS:
                        self.stderr.write(f"    Error calculating future trading dates for {stock_name}. Expected {FUTURE_TARGET_DAYS}, got {len(future_dates_dt)}. Skipping.")
                        error_count += 1; continue
                    
                    model_name_or_path = getattr(selected_model, 'name', "") or getattr(selected_model, 'filepath', "") # Keras 3 호환성
                    model_was_log_trained = "_log_" in model_name_or_path.lower()

                    predictions_to_save = []
                    current_base_price_for_clipping = recent_features_df['Close'].iloc[-1] 

                    for day_idx in range(FUTURE_TARGET_DAYS):
                        pred_price_day_i_raw = prediction_actual_prices_raw[day_idx]
                        final_pred_price_day_i = np.expm1(pred_price_day_i_raw) if model_was_log_trained else pred_price_day_i_raw
                        upper_limit = current_base_price_for_clipping * 1.30; lower_limit = current_base_price_for_clipping * 0.70 
                        clipped_price = np.clip(final_pred_price_day_i, lower_limit, upper_limit)
                        
                        predictions_to_save.append({
                            'stock_code': stock_code, 'stock_name': stock_name, 'market_name': market_name_filter,
                            'prediction_base_date': actual_prediction_base_date, 'predicted_date': future_dates_dt[day_idx],
                            'predicted_price': round(float(clipped_price)), 'analysis_type': analysis_type_to_run,
                        })
                        current_base_price_for_clipping = clipped_price 
                    
                    delete_count, _ = PredictedStockPrice.objects.filter(
                        stock_code=stock_code, prediction_base_date=actual_prediction_base_date, analysis_type=analysis_type_to_run
                    ).delete()
                    if delete_count > 0: self.stdout.write(f"    Deleted {delete_count} existing predictions for {stock_name} on {actual_prediction_base_date} for {analysis_type_to_run}.")
                    
                    PredictedStockPrice.objects.bulk_create([PredictedStockPrice(**data) for data in predictions_to_save])
                    self.stdout.write(self.style.SUCCESS(f"    Successfully saved {len(predictions_to_save)} days for {stock_name} ({stock_code}) based on data up to {actual_prediction_base_date}."))
                    processed_count += 1
                except Exception as e_pred:
                    self.stderr.write(self.style.ERROR(f"    Error during prediction/saving for {stock_name} ({stock_code}): {e_pred}"))
                    self.stderr.write(traceback.format_exc()); error_count += 1
                
                if (i+1) % 50 == 0: self.stdout.write(f"  ... processed {i+1}/{market_stock_count} stocks in {market_name_filter} ...")
                time.sleep(0.05) 

        self.stdout.write(self.style.SUCCESS(f"\n--- Prediction Generation Summary ---"))
        self.stdout.write(f"Total stocks considered: {len(all_krx_stocks)}")
        self.stdout.write(f"Successfully predicted/saved: {processed_count}")
        self.stdout.write(f"Skipped (e.g., insufficient data): {skipped_count}") # 이 값이 크게 나온다면 데이터 준비 단계 문제
        self.stdout.write(f"Errors (data prep, model load, prediction): {error_count}")

        if delete_older_than_days > 0:
            cutoff_date = timezone.now().date() - timedelta(days=delete_older_than_days)
            self.stdout.write(f"\nDeleting predictions with prediction_base_date older than {cutoff_date}...")
            try:
                deleted_info = PredictedStockPrice.objects.filter(prediction_base_date__lt=cutoff_date).delete()
                self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_info[0]} old prediction records."))
            except Exception as e_delete: self.stderr.write(self.style.ERROR(f"Error deleting old predictions: {e_delete}"))
        else: self.stdout.write(f"Old prediction deletion skipped (--delete_old_after_days={delete_older_than_days}).")

        end_time_total = time.time()
        self.stdout.write(self.style.SUCCESS(f"Daily prediction generation finished in {end_time_total - start_time_total:.2f} seconds."))

