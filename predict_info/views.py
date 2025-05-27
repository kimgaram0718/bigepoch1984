from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
import os
import traceback
import holidays
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
import json
import joblib # 스케일러 로드/저장
import tensorflow as tf # 모델 로드
from django.utils import timezone

from .utils import (
    get_market_macro_data, PANDAS_TA_AVAILABLE,
    add_fundamental_indicator_features,
    get_krx_stock_list, get_future_trading_dates_list, calculate_all_features,
    get_kr_holidays, 
    get_past_trading_dates_list # utils.py에 이 함수가 정의되어 있어야 합니다.
)
from .models import PredictedStockPrice, FavoriteStock, StockPrice

ML_MODELS_DIR = settings.ML_MODELS_DIR

TIME_STEPS = 10 
FUTURE_TARGET_DAYS = 5 

models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}
model_info_cache = {}

KOSPI_TECH_LSTM_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'ATR_14', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    'MACDs_12_26_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14',
    'DMP_14', 'DMN_14', 'KOSPI_Close', 'KOSPI_Change', 'USD_KRW_Close',
    'USD_KRW_Change', 'Indi', 'Foreign', 'Organ', 'MarketCap', 'PBR', 'PER',
    'MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero'
]
KOSPI_LOG_TRANSFORMED_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'KOSPI_Close', 'USD_KRW_Close',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'MarketCap', 'PBR', 'PER'
]

KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER = KOSPI_TECH_LSTM_FEATURES.copy() 
KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER = KOSPI_LOG_TRANSFORMED_FEATURES.copy()
if 'KOSPI_Close' in KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER:
    idx = KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER.index('KOSPI_Close')
    KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER[idx] = 'KOSDAQ_Close'
if 'KOSPI_Change' in KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER:
    idx = KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER.index('KOSPI_Change')
    KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER[idx] = 'KOSDAQ_Change'
if 'KOSPI_Close' in KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER: 
    idx_log = KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER.index('KOSPI_Close')
    KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER[idx_log] = 'KOSDAQ_Close'

KOSPI_GENERAL_LSTM_FEATURES_PLACEHOLDER = KOSPI_TECH_LSTM_FEATURES.copy() 
KOSPI_GENERAL_LOG_FEATURES_PLACEHOLDER = KOSPI_LOG_TRANSFORMED_FEATURES.copy()

KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER = KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER.copy() 
KOSDAQ_GENERAL_LOG_FEATURES_PLACEHOLDER = KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER.copy()
if 'KOSPI_Close' in KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER:
    idx = KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER.index('KOSPI_Close')
    KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER[idx] = 'KOSDAQ_Close'
if 'KOSPI_Change' in KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER:
    idx = KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER.index('KOSPI_Change')
    KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER[idx] = 'KOSDAQ_Change'
if 'KOSPI_Close' in KOSDAQ_GENERAL_LOG_FEATURES_PLACEHOLDER: 
    idx_log = KOSDAQ_GENERAL_LOG_FEATURES_PLACEHOLDER.index('KOSPI_Close')
    KOSDAQ_GENERAL_LOG_FEATURES_PLACEHOLDER[idx_log] = 'KOSDAQ_Close'

DEFAULT_MODEL_PARAMS = {
    'kospi_technical_lstm': {
        'model_filename': 'kospi_technical_model.keras',
        'scaler_X_filename': 'kospi_technical_scaler_X.joblib',
        'scaler_y_filename': 'kospi_technical_scaler_y.joblib',
        'time_steps': TIME_STEPS, 'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI', 
        'trained_feature_list': KOSPI_TECH_LSTM_FEATURES,
        'log_transformed_input_features': KOSPI_LOG_TRANSFORMED_FEATURES,
    },
    'kosdaq_technical_lstm': {
        'model_filename': 'kosdaq_technical_model.keras', 
        'scaler_X_filename': 'kosdaq_technical_scaler_X.joblib',
        'scaler_y_filename': 'kosdaq_technical_scaler_y.joblib',
        'time_steps': TIME_STEPS, 'model_was_log_trained': True, 
        'market_name_for_features': 'KOSDAQ',
        'trained_feature_list': KOSDAQ_TECH_LSTM_FEATURES_PLACEHOLDER, 
        'log_transformed_input_features': KOSDAQ_LOG_TRANSFORMED_FEATURES_PLACEHOLDER, 
    },
    'kospi_lstm': { 
        'model_filename': 'kospi_lstm_model.keras', 
        'scaler_X_filename': 'kospi_lstm_scaler_X.joblib',
        'scaler_y_filename': 'kospi_lstm_scaler_y.joblib',
        'time_steps': TIME_STEPS, 'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI',
        'trained_feature_list': KOSPI_GENERAL_LSTM_FEATURES_PLACEHOLDER, 
        'log_transformed_input_features': KOSPI_GENERAL_LOG_FEATURES_PLACEHOLDER, 
    },
    'kosdaq_lstm': { 
        'model_filename': 'kosdaq_lstm_model.keras', 
        'scaler_X_filename': 'kosdaq_lstm_scaler_X.joblib',
        'scaler_y_filename': 'kosdaq_lstm_scaler_y.joblib',
        'time_steps': TIME_STEPS, 'model_was_log_trained': True, 
        'market_name_for_features': 'KOSDAQ',
        'trained_feature_list': KOSDAQ_GENERAL_LSTM_FEATURES_PLACEHOLDER, 
        'log_transformed_input_features': KOSDAQ_GENERAL_LOG_FEATURES_PLACEHOLDER, 
    }
}

def load_model_and_scalers(market_name, model_type_key):
    full_model_key = f"{market_name.lower()}_{model_type_key}"
    if full_model_key in models_cache:
        return (models_cache.get(full_model_key), scalers_X_cache.get(full_model_key),
                scalers_y_cache.get(full_model_key), model_info_cache.get(full_model_key))
    params = DEFAULT_MODEL_PARAMS.get(full_model_key)
    if not params:
        print(f"[ERROR][load_model_and_scalers] No model parameters for key: {full_model_key}")
        return None, None, None, None
    if not params.get('trained_feature_list'):
         print(f"[ERROR][load_model_and_scalers] 'trained_feature_list' missing for model: {full_model_key}")
         return None, None, None, None
    model_path = os.path.join(ML_MODELS_DIR, params['model_filename'])
    scaler_X_path = os.path.join(ML_MODELS_DIR, params['scaler_X_filename'])
    scaler_y_path = os.path.join(ML_MODELS_DIR, params['scaler_y_filename'])
    try:
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_X_path): raise FileNotFoundError(f"Scaler X not found: {scaler_X_path}")
        if not os.path.exists(scaler_y_path): raise FileNotFoundError(f"Scaler Y not found: {scaler_y_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        current_model_info = {
            'time_steps': params['time_steps'],
            'model_was_log_trained': params['model_was_log_trained'], 
            'market_name_for_features': params['market_name_for_features'], 
            'trained_feature_list': params.get('trained_feature_list'),
            'log_transformed_input_features': params.get('log_transformed_input_features', []) 
        }
        models_cache[full_model_key] = model
        scalers_X_cache[full_model_key] = scaler_X
        scalers_y_cache[full_model_key] = scaler_y
        model_info_cache[full_model_key] = current_model_info
        print(f"[INFO][load_model_and_scalers] Loaded model/scalers for {full_model_key}")
        return model, scaler_X, scaler_y, current_model_info
    except FileNotFoundError as fnf_err:
        print(f"[ERROR][load_model_and_scalers] File not found for {full_model_key}: {fnf_err}")
    except Exception as e:
        print(f"[ERROR][load_model_and_scalers] Error loading for {full_model_key}: {e}\n{traceback.format_exc()}")
    return None, None, None, None

def predict_info_view(request):
    user = request.user
    initial_stock_code, initial_stock_name, initial_market_name = None, None, None
    is_favorite, stock_name_for_display = False, ""
    if user.is_authenticated:
        favorite = FavoriteStock.objects.filter(user=user).order_by('added_at').first()
        if favorite:
            initial_stock_code, initial_stock_name, initial_market_name = favorite.stock_code, favorite.stock_name, favorite.market_name
            is_favorite, stock_name_for_display = True, initial_stock_name
    stock_query_from_url = request.GET.get('stock_query')
    if stock_query_from_url:
        temp_stock_info = None
        krx_list_all_markets = get_krx_stock_list(market='KOSPI,KOSDAQ')
        if stock_query_from_url.isdigit() and len(stock_query_from_url) == 6:
            found_stock = next((s for s in krx_list_all_markets if s['Code'] == stock_query_from_url), None)
            if found_stock: temp_stock_info = found_stock
        else:
            found_stock = next((s for s in krx_list_all_markets if stock_query_from_url.lower() in s['Name'].lower()), None)
            if found_stock: temp_stock_info = found_stock
        if temp_stock_info:
            initial_stock_code, initial_stock_name, initial_market_name = temp_stock_info['Code'], temp_stock_info['Name'], temp_stock_info['Market']
            stock_name_for_display = initial_stock_name
            if user.is_authenticated: is_favorite = FavoriteStock.objects.filter(user=user, stock_code=initial_stock_code).exists()
            else: is_favorite = False
        else: stock_name_for_display = stock_query_from_url
    context = {'ticker': initial_stock_code, 'stock_name_for_display': stock_name_for_display,
               'market_name': initial_market_name, 'is_favorite': is_favorite,
               'max_favorites': settings.MAX_FAVORITES_PER_USER if hasattr(settings, 'MAX_FAVORITES_PER_USER') else 5}
    return render(request, 'predict_info/predict_info.html', context)

def predict_stock_price_ajax(request):
    if request.method == 'POST':
        stock_input_from_frontend = request.POST.get('stock_input')
        analysis_type_from_frontend = request.POST.get('analysis_type', 'technical') 
        if not stock_input_from_frontend:
            return JsonResponse({'error': '종목명 또는 코드를 입력해주세요.'}, status=400)
        resolved_stock_code, resolved_stock_name, resolved_market_name = None, None, None
        krx_list = get_krx_stock_list(market='KOSPI,KOSDAQ')
        if stock_input_from_frontend.isdigit() and len(stock_input_from_frontend) == 6:
            found_stock = next((s for s in krx_list if s['Code'] == stock_input_from_frontend), None)
            if found_stock:
                resolved_stock_code, resolved_stock_name, resolved_market_name = found_stock['Code'], found_stock['Name'], found_stock['Market'].upper()
        else:
            found_stock = next((s for s in krx_list if stock_input_from_frontend.lower() in s['Name'].lower()), None)
            if found_stock:
                resolved_stock_code, resolved_stock_name, resolved_market_name = found_stock['Code'], found_stock['Name'], found_stock['Market'].upper()
        if not resolved_stock_code or not resolved_market_name:
            return JsonResponse({'error': f"'{stock_input_from_frontend}'에 해당하는 종목을 찾을 수 없습니다."}, status=400)

        model_type_key_for_load = 'technical_lstm' if analysis_type_from_frontend == 'technical' else ('lstm' if analysis_type_from_frontend == 'lstm' else 'technical_lstm')
        
        try:
            model, scaler_X, scaler_y, model_info = load_model_and_scalers(resolved_market_name, model_type_key=model_type_key_for_load)
            if not model or not scaler_X or not scaler_y or not model_info:
                return JsonResponse({'error': f'{resolved_market_name} 시장의 {model_type_key_for_load} 모델/스케일러를 로드할 수 없습니다.'}, status=500)
            
            trained_feature_list = model_info.get('trained_feature_list')
            if not trained_feature_list: 
                return JsonResponse({'error': f"모델 '{resolved_market_name}_{model_type_key_for_load}'에 대한 학습된 피처 목록이 정의되지 않았습니다."}, status=500)

            log_transformed_input_features = model_info.get('log_transformed_input_features', [])
            time_steps = model_info['time_steps']
            model_was_log_trained_target = model_info['model_was_log_trained']
            market_name_for_feature_calc = model_info.get('market_name_for_features', resolved_market_name)

            latest_data_in_db = StockPrice.objects.filter(stock_code=resolved_stock_code).order_by('-date').first()
            if not latest_data_in_db:
                return JsonResponse({'error': f'{resolved_stock_name}({resolved_stock_code}) DB에 데이터가 없습니다.'}, status=400)
            
            prediction_base_date_for_model_input = latest_data_in_db.date 

            min_ta_window = 120
            min_records_needed_for_sequence_and_ta = min_ta_window + time_steps
            calendar_day_fetch_multiplier = 1.7 
            fetch_buffer_days = 45 
            required_history_calendar_days = int(min_records_needed_for_sequence_and_ta * calendar_day_fetch_multiplier) + fetch_buffer_days
            db_start_date = prediction_base_date_for_model_input - timedelta(days=required_history_calendar_days)
            
            stock_price_qs = StockPrice.objects.filter(
                stock_code=resolved_stock_code, 
                date__gte=db_start_date, 
                date__lte=prediction_base_date_for_model_input
            ).order_by('date')

            if stock_price_qs.count() < min_records_needed_for_sequence_and_ta:
                 return JsonResponse({'error': f'{resolved_stock_name}({resolved_stock_code}) 예측을 위한 DB 데이터 부족 (필요: {min_records_needed_for_sequence_and_ta}일, 현재: {stock_price_qs.count()}일). 기준일: {prediction_base_date_for_model_input}'}, status=400)

            raw_df_from_db = pd.DataFrame(list(stock_price_qs.values()))
            raw_df_from_db['date'] = pd.to_datetime(raw_df_from_db['date']) # 'date' 컬럼을 datetime으로 변환
            raw_df_from_db.set_index('date', inplace=True) # DatetimeIndex로 설정
            
            past_data_for_graph = []
            if len(raw_df_from_db) >= 5:
                past_df_slice = raw_df_from_db.iloc[-5:] 
                for p_date_idx, p_row in past_df_slice.iterrows():
                    past_data_for_graph.append({
                        'date': p_date_idx.strftime('%Y-%m-%d'), 
                        'price': round(float(p_row['close_price']), 2) if pd.notna(p_row['close_price']) else None
                    })
            
            base_cols_map_for_calc = {
                'Open': 'open_price', 'High': 'high_price', 'Low': 'low_price', 'Close': 'close_price', 'Volume': 'volume',
                'MarketCap': 'market_cap', 'PBR': 'pbr', 'PER': 'per',
                'Indi': 'indi_volume', 'Foreign': 'foreign_volume', 'Organ': 'organ_volume'
            }
            # df_for_feature_calc는 raw_df_from_db (DatetimeIndex를 가짐)를 기반으로 생성
            df_for_feature_calc = pd.DataFrame(index=raw_df_from_db.index) 
            for calc_col, db_field_name in base_cols_map_for_calc.items():
                if db_field_name in raw_df_from_db:
                    df_for_feature_calc[calc_col] = raw_df_from_db[db_field_name]
                else:
                    if calc_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                         return JsonResponse({'error': f"DB에 필수 컬럼 '{db_field_name}'({calc_col}) 없음"}, status=500)
                    df_for_feature_calc[calc_col] = np.nan

            if 'Close' in df_for_feature_calc.columns:
                df_for_feature_calc['Change'] = df_for_feature_calc['Close'].pct_change()
            else: df_for_feature_calc['Change'] = np.nan

            calc_start_date_str = df_for_feature_calc.index.min().strftime('%Y-%m-%d')
            calc_end_date_str = df_for_feature_calc.index.max().strftime('%Y-%m-%d')
            
            other_market_for_mm_calc = 'KOSDAQ' if market_name_for_feature_calc.upper() == 'KOSPI' else 'KOSPI'
            market_macro_df_calc = get_market_macro_data(
                calc_start_date_str, calc_end_date_str, 
                market_name_for_feature_calc.upper(),
                other_market_name_for_index=other_market_for_mm_calc
            )

            processed_df_superset = calculate_all_features(
                stock_df_ohlcv=df_for_feature_calc[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']],
                market_macro_data_df=market_macro_df_calc,
                investor_df=df_for_feature_calc[['Indi', 'Foreign', 'Organ']],
                fundamental_df=df_for_feature_calc[['MarketCap', 'PBR', 'PER']],
                pandas_ta_available=PANDAS_TA_AVAILABLE
            )
            processed_df_superset = add_fundamental_indicator_features(processed_df_superset) 
            
            final_features_df = pd.DataFrame(index=processed_df_superset.index)
            missing_features_in_superset = []
            for feature_name in trained_feature_list:
                if feature_name in processed_df_superset:
                    final_features_df[feature_name] = processed_df_superset[feature_name]
                else:
                    missing_features_in_superset.append(feature_name)
            
            if missing_features_in_superset:
                return JsonResponse({'error': f"필수 피처 누락: {missing_features_in_superset}"}, status=500)
            
            for col in final_features_df.columns: 
                if final_features_df[col].dtype == 'object':
                    try: final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce')
                    except Exception as e_conv:
                        print(f"Warning: Col {col} to numeric failed for {resolved_stock_code}. Err: {e_conv}")
                        final_features_df[col] = np.nan 
                if not pd.api.types.is_float_dtype(final_features_df[col]) and pd.api.types.is_numeric_dtype(final_features_df[col]):
                     if col not in ['MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero']:
                        try: final_features_df[col] = final_features_df[col].astype(float)
                        except ValueError: final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce')

            if 'Change' in final_features_df.columns:
                final_features_df['Change'] = pd.to_numeric(final_features_df['Change'], errors='coerce').fillna(0)
            final_features_df = final_features_df.ffill().bfill()

            if log_transformed_input_features:
                for col_to_log in log_transformed_input_features:
                    if col_to_log in final_features_df.columns:
                        numeric_col = pd.to_numeric(final_features_df[col_to_log], errors='coerce')
                        final_features_df[col_to_log] = np.log1p(numeric_col.clip(lower=0)) if not numeric_col.isnull().all() else np.nan
                    else: print(f"[WARN] Col '{col_to_log}' for log transform not found for {resolved_stock_code}.")
                final_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                final_features_df = final_features_df.ffill().bfill()

            if final_features_df.isnull().values.any():
                nan_counts = final_features_df.isnull().sum()
                nan_cols_with_counts = nan_counts[nan_counts > 0]
                cols_to_check_for_nan_warn = [c for c in nan_cols_with_counts.index if not c.endswith('_is_nan') and not c.endswith('_is_zero')]
                if cols_to_check_for_nan_warn:
                    print(f"[WARN] NaNs before scaling for {resolved_stock_code} in non-indicator: {nan_cols_with_counts[cols_to_check_for_nan_warn]}")

            if final_features_df.shape[0] < time_steps:
                 return JsonResponse({'error': f'최종 데이터 부족 (필요: {time_steps}일, 현재: {final_features_df.shape[0]}일).'}, status=400)

            if hasattr(scaler_X, 'n_features_in_') and scaler_X.n_features_in_ != final_features_df.shape[1]:
                return JsonResponse({'error': f"스케일러 X 피처 불일치 ({scaler_X.n_features_in_} vs {final_features_df.shape[1]})"}, status=500)

            try:
                scaled_features = scaler_X.transform(final_features_df.astype(float).values)
            except ValueError as e_scale:
                if 'Input contains NaN' in str(e_scale):
                    print(f"[WARN] Scaler X NaNs for {resolved_stock_code}. Filling with 0.")
                    scaled_features = scaler_X.transform(final_features_df.fillna(0).astype(float).values)
                else: raise e_scale

            last_sequence = scaled_features[-time_steps:]
            last_sequence_reshaped = np.reshape(last_sequence, (1, time_steps, last_sequence.shape[1]))
            predicted_scaled_values = model.predict(last_sequence_reshaped, verbose=0)

            try:
                predicted_actual_values = scaler_y.inverse_transform(predicted_scaled_values)
            except ValueError as ve_scaler_y:
                if predicted_scaled_values.shape[1] == FUTURE_TARGET_DAYS and hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ == 1:
                    single_day_pred_scaled = predicted_scaled_values[:, 0].reshape(-1,1)
                    predicted_actual_values = np.full((1, FUTURE_TARGET_DAYS), scaler_y.inverse_transform(single_day_pred_scaled)[0,0])
                else: return JsonResponse({'error': f"Scaler Y 오류: {ve_scaler_y}"}, status=500)

            if model_was_log_trained_target:
                predicted_actual_values = np.expm1(predicted_actual_values)
            
            # IndexError 해결: df_for_feature_calc의 마지막 행의 'Close' 값을 사용
            if not df_for_feature_calc.empty:
                last_actual_close_for_clipping = df_for_feature_calc['Close'].iloc[-1]
                if pd.isna(last_actual_close_for_clipping): # 마지막 값이 NaN이면 그 이전 유효한 값으로
                    last_actual_close_for_clipping = df_for_feature_calc['Close'].ffill().iloc[-1]
            else:
                last_actual_close_for_clipping = 0 
                print(f"[WARN] df_for_feature_calc is empty for {resolved_stock_code}, cannot get last_actual_close_for_clipping.")


            kr_holidays_for_future = get_kr_holidays([prediction_base_date_for_model_input.year, prediction_base_date_for_model_input.year + 1])
            future_dates = get_future_trading_dates_list(prediction_base_date_for_model_input, FUTURE_TARGET_DAYS, kr_holidays_for_future)

            predictions_list = []
            current_reference_price_for_clipping = last_actual_close_for_clipping if pd.notna(last_actual_close_for_clipping) else 0

            for i in range(FUTURE_TARGET_DAYS):
                predicted_price = predicted_actual_values[0, i]
                # 예측값이 NaN이거나 유효하지 않으면 클리핑하지 않고 None으로 처리할 수 있음
                if pd.isna(predicted_price):
                    clipped_price = np.nan # 또는 None
                else:
                    price_change_limit_factor = 0.30 
                    upper_limit = current_reference_price_for_clipping * (1 + price_change_limit_factor)
                    lower_limit = current_reference_price_for_clipping * (1 - price_change_limit_factor)
                    clipped_price = np.clip(predicted_price, lower_limit, upper_limit)
                
                predictions_list.append({
                    'date': future_dates[i].strftime('%Y-%m-%d'),
                    'price': round(float(clipped_price), 2) if pd.notna(clipped_price) else None
                })
                current_reference_price_for_clipping = clipped_price if pd.notna(clipped_price) else current_reference_price_for_clipping
            
            is_favorite_for_user = False
            if request.user.is_authenticated:
                is_favorite_for_user = FavoriteStock.objects.filter(user=request.user, stock_code=resolved_stock_code).exists()

            return JsonResponse({
                'stock_code': resolved_stock_code,
                'stock_name': resolved_stock_name,
                'market_name': resolved_market_name,
                'prediction_base_date': prediction_base_date_for_model_input.strftime('%Y-%m-%d'),
                'last_actual_close': round(float(last_actual_close_for_clipping), 2) if pd.notna(last_actual_close_for_clipping) else None,
                'past_data': past_data_for_graph, 
                'predictions': predictions_list,
                'is_favorite': is_favorite_for_user,
                'is_authenticated': request.user.is_authenticated
            })

        except FileNotFoundError as e_fnf:
            return JsonResponse({'error': f'모델 관련 파일을 찾을 수 없습니다: {e_fnf}'}, status=500)
        except Exception as e:
            print(f"[ERROR][predict_stock_price_ajax] Prediction error for {resolved_stock_code} ({resolved_market_name}): {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'예측 중 오류 발생: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': '잘못된 요청입니다 (POST 방식 필요).'}, status=400)

def search_stocks_ajax(request):
    query = request.GET.get('term', '').strip()
    if len(query) < 1 and '*' not in query : return JsonResponse([], safe=False)
    cache_key = f"stock_search_all_{query}"
    cached_results = cache.get(cache_key)
    if cached_results: return JsonResponse(cached_results, safe=False)
    results = []
    try:
        all_krx_stocks = get_krx_stock_list(market='KOSPI,KOSDAQ')
        if query == '*':
            results = [{"label": f"{s['Name']} ({s['Code']}, {s['Market']})", "value": s['Name'], "code": s['Code'], "market": s['Market']} for s in all_krx_stocks[:200]]
        else:
            for stock in all_krx_stocks:
                if query.lower() in stock['Name'].lower() or query in stock['Code']:
                    results.append({"label": f"{stock['Name']} ({stock['Code']}, {stock['Market']})", "value": stock['Name'], "code": stock['Code'], "market": stock['Market']})
                if len(results) >= 20: break
        cache.set(cache_key, results, timeout=60*15)
    except Exception as e:
        print(f"[ERROR][search_stocks_ajax] Error searching stocks for query '{query}': {e}")
        return JsonResponse({'error': '검색 중 오류가 발생했습니다.'}, safe=False)
    return JsonResponse(results, safe=False)

MAX_FAVORITES = settings.MAX_FAVORITES_PER_USER if hasattr(settings, 'MAX_FAVORITES_PER_USER') else 5
@login_required
def toggle_favorite_stock_ajax(request):
    if request.method == 'POST':
        try: data = json.loads(request.body)
        except json.JSONDecodeError: return JsonResponse({'status': 'error', 'message': '잘못된 JSON 형식입니다.'}, status=400)
        stock_code = data.get('stock_code')
        stock_name = data.get('stock_name')
        market_name = data.get('market_name')
        if not stock_code or not stock_name or not market_name:
            return JsonResponse({'status': 'error', 'message': '필수 정보(종목코드, 종목명, 시장)가 누락되었습니다.'}, status=400)
        user = request.user
        try:
            favorite_obj, created = FavoriteStock.objects.get_or_create(user=user, stock_code=stock_code, defaults={'stock_name': stock_name, 'market_name': market_name})
            if created:
                if FavoriteStock.objects.filter(user=user).count() > MAX_FAVORITES:
                    favorite_obj.delete()
                    return JsonResponse({'status': 'error', 'message': f'관심 종목은 최대 {MAX_FAVORITES}개까지 추가할 수 있습니다.', 'is_favorite': False})
                return JsonResponse({'status': 'success', 'message': f"'{stock_name}'을(를) 관심 종목에 추가했습니다.", 'is_favorite': True})
            else:
                if favorite_obj.stock_name != stock_name or favorite_obj.market_name != market_name:
                    favorite_obj.stock_name = stock_name
                    favorite_obj.market_name = market_name
                    favorite_obj.save()
                favorite_obj.delete()
                return JsonResponse({'status': 'success', 'message': f"'{stock_name}'을(를) 관심 종목에서 삭제했습니다.", 'is_favorite': False})
        except Exception as e:
            print(f"[ERROR][toggle_favorite_stock_ajax] 관심종목 처리 중 오류 ({user.username}, {stock_code}): {e}\n{traceback.format_exc()}")
            return JsonResponse({'status': 'error', 'message': '관심 종목 처리 중 오류가 발생했습니다.'}, status=500)
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다 (POST 방식, JSON 필요).'}, status=400)
