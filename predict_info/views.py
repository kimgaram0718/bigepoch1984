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
import unicodedata # 문자열 정규화
import re # 정규 표현식 추가

from .utils import (
    get_market_macro_data, PANDAS_TA_AVAILABLE,
    add_fundamental_indicator_features,
    get_krx_stock_list, get_future_trading_dates_list, calculate_all_features,
    get_kr_holidays, 
    get_past_trading_dates_list 
)
from .models import PredictedStockPrice, FavoriteStock, StockPrice

ML_MODELS_DIR = settings.ML_MODELS_DIR

TIME_STEPS = 10 
FUTURE_TARGET_DAYS = 5 

# --- 모델별 캐시 ---
models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}
model_info_cache = {}

# --- KOSPI 모델 피처 정의 ---
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
KOSPI_COMPREHENSIVE_LSTM_FEATURES = KOSPI_TECH_LSTM_FEATURES.copy() 
KOSPI_COMPREHENSIVE_LOG_FEATURES = KOSPI_LOG_TRANSFORMED_FEATURES.copy()


# --- KOSDAQ 모델 피처 정의 ---
# 사용자님이 제공해주신 코스닥 모델 학습 피처를 그대로 사용합니다.
# 이 리스트에는 'KOSPI_Close'와 'KOSPI_Change'가 포함되어 있습니다.
# print("[DEBUG] Defining KOSDAQ_TECH_LSTM_FEATURES directly from user provided list.")
KOSDAQ_TECH_LSTM_FEATURES = [ # 사용자가 제공한 34개 피처 리스트
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'ATR_14', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    'MACDs_12_26_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14',
    'DMP_14', 'DMN_14', 
    'KOSPI_Close', # 사용자 제공: 코스닥 모델 학습 시 KOSPI 지수 사용
    'KOSPI_Change',# 사용자 제공: 코스닥 모델 학습 시 KOSPI 지수 사용
    'USD_KRW_Close', 'USD_KRW_Change', 'Indi', 'Foreign', 'Organ', 'MarketCap', 'PBR', 'PER',
    'MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero'
]
# print(f"[DEBUG] KOSDAQ_TECH_LSTM_FEATURES set to (count: {len(KOSDAQ_TECH_LSTM_FEATURES)}): {KOSDAQ_TECH_LSTM_FEATURES}")

# 사용자님이 제공해주신 코스닥 모델 로그 변환 대상 컬럼을 그대로 사용합니다.
# print("[DEBUG] Defining KOSDAQ_LOG_TRANSFORMED_FEATURES directly from user provided list.")
KOSDAQ_LOG_TRANSFORMED_FEATURES = [ # 사용자가 제공한 12개 로그 변환 대상 컬럼
    'Open', 'High', 'Low', 'Close', 
    'KOSPI_Close', # 사용자 제공: 코스닥 모델 학습 시 KOSPI 지수 사용
    'USD_KRW_Close', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'MarketCap', 'PBR', 'PER'
]
# print(f"[DEBUG] KOSDAQ_LOG_TRANSFORMED_FEATURES set to (count: {len(KOSDAQ_LOG_TRANSFORMED_FEATURES)}): {KOSDAQ_LOG_TRANSFORMED_FEATURES}")


# KOSDAQ 종합 모델용 피처 (실제 학습된 피처로 수정 필요)
# 현재는 기술적 분석 피처(사용자 제공 KOSPI 지수 포함 버전)를 기본으로 가정
KOSDAQ_COMPREHENSIVE_LSTM_FEATURES = KOSDAQ_TECH_LSTM_FEATURES.copy() 
KOSDAQ_COMPREHENSIVE_LOG_FEATURES = KOSDAQ_LOG_TRANSFORMED_FEATURES.copy()


DEFAULT_MODEL_PARAMS = {
    'kospi_technical_lstm': {
        'model_filename': 'kospi_technical_model.keras',
        'scaler_X_filename': 'kospi_technical_scaler_X.joblib',
        'scaler_y_filename': 'kospi_technical_scaler_y.joblib',
        'time_steps': TIME_STEPS, 
        'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI', 
        'trained_feature_list': KOSPI_TECH_LSTM_FEATURES,
        'log_transformed_input_features': KOSPI_LOG_TRANSFORMED_FEATURES,
    },
    'kosdaq_technical_lstm': {
        'model_filename': 'kosdaq_technical_model.keras', 
        'scaler_X_filename': 'kosdaq_technical_scaler_X.joblib',
        'scaler_y_filename': 'kosdaq_technical_scaler_y.joblib',
        'time_steps': TIME_STEPS, 
        'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI', # 코스닥 모델이 KOSPI 지수를 피처로 사용하므로 'KOSPI'
        'trained_feature_list': KOSDAQ_TECH_LSTM_FEATURES, 
        'log_transformed_input_features': KOSDAQ_LOG_TRANSFORMED_FEATURES, 
    },
    'kospi_lstm': { 
        'model_filename': 'kospi_lstm_model.keras', 
        'scaler_X_filename': 'kospi_lstm_scaler_X.joblib',
        'scaler_y_filename': 'kospi_lstm_scaler_y.joblib',
        'time_steps': TIME_STEPS, 
        'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI',
        'trained_feature_list': KOSPI_COMPREHENSIVE_LSTM_FEATURES, 
        'log_transformed_input_features': KOSPI_COMPREHENSIVE_LOG_FEATURES, 
    },
    'kosdaq_lstm': { 
        'model_filename': 'kosdaq_lstm_model.keras', 
        'scaler_X_filename': 'kosdaq_lstm_scaler_X.joblib',
        'scaler_y_filename': 'kosdaq_lstm_scaler_y.joblib',
        'time_steps': TIME_STEPS, 
        'model_was_log_trained': True, 
        'market_name_for_features': 'KOSPI', # 코스닥 종합 모델도 KOSPI 지수 사용 가정
        'trained_feature_list': KOSDAQ_COMPREHENSIVE_LSTM_FEATURES, 
        'log_transformed_input_features': KOSDAQ_COMPREHENSIVE_LOG_FEATURES, 
    }
}

def load_model_and_scalers(market_name, model_type_key):
    full_model_key = f"{market_name.lower()}_{model_type_key}"
    # print(f"[DEBUG][load_model_and_scalers] Attempting to load model for key: {full_model_key}")
    
    if full_model_key in models_cache:
        # print(f"[DEBUG][load_model_and_scalers] Model for {full_model_key} found in cache.")
        return (models_cache.get(full_model_key), scalers_X_cache.get(full_model_key),
                scalers_y_cache.get(full_model_key), model_info_cache.get(full_model_key))

    params = DEFAULT_MODEL_PARAMS.get(full_model_key)
    if not params:
        print(f"[ERROR][load_model_and_scalers] No model parameters found for key: {full_model_key}")
        return None, None, None, None
    
    if not params.get('trained_feature_list'):
         print(f"[ERROR][load_model_and_scalers] 'trained_feature_list' is missing for model: {full_model_key}")
         return None, None, None, None

    model_path = os.path.join(ML_MODELS_DIR, params['model_filename'])
    scaler_X_path = os.path.join(ML_MODELS_DIR, params['scaler_X_filename'])
    scaler_y_path = os.path.join(ML_MODELS_DIR, params['scaler_y_filename'])
    
    # print(f"[DEBUG][load_model_and_scalers] For key '{full_model_key}':")
    # print(f"    Model path: {model_path}")
    # print(f"    Scaler X path: {scaler_X_path}")
    # print(f"    Scaler Y path: {scaler_y_path}")
    # print(f"    market_name_for_features: {params.get('market_name_for_features')}")

    try:
        if not os.path.exists(model_path): 
            raise FileNotFoundError(f"Model file not found: {model_path} for key {full_model_key}")
        if not os.path.exists(scaler_X_path): 
            raise FileNotFoundError(f"Scaler X file not found: {scaler_X_path} for key {full_model_key}")
        if not os.path.exists(scaler_y_path): 
            raise FileNotFoundError(f"Scaler Y file not found: {scaler_y_path} for key {full_model_key}")

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
        
        print(f"[INFO][load_model_and_scalers] Successfully loaded model and scalers for {full_model_key}")
        return model, scaler_X, scaler_y, current_model_info

    except FileNotFoundError as fnf_err:
        print(f"[ERROR][load_model_and_scalers] File not found for {full_model_key}: {fnf_err}")
    except Exception as e:
        print(f"[ERROR][load_model_and_scalers] Error loading model/scalers for {full_model_key}: {e}\n{traceback.format_exc()}")
    
    return None, None, None, None

def normalize_stock_name(name):
    """문자열 정규화 (NFC), 소문자 변환, 양쪽 공백 제거, 괄호 및 내용 제거, 특수문자 일부 제거"""
    if not isinstance(name, str):
        return ""
    name_normalized = unicodedata.normalize('NFC', name)
    name_lower = name_normalized.lower()
    # 괄호와 그 안의 내용 제거 (예: (주), (Reg.S), (舊 ...))
    name_no_parentheses = re.sub(r'\([^)]*\)', '', name_lower)
    # 일부 특수 문자(알파벳, 숫자, 한글, 공백 제외) 제거. 필요에 따라 조정.
    # 너무 많은 것을 제거하면 오탐색 가능성. 우선은 최소한으로.
    # name_alphanumeric_korean = re.sub(r'[^\w\s가-힣]', '', name_no_parentheses, flags=re.UNICODE)
    # 위 방식은 밑줄(_)도 남기므로, 좀 더 명시적으로 허용 문자 지정
    name_cleaned = re.sub(r'[^a-z0-9가-힣\s]', '', name_no_parentheses) # 영어소문자, 숫자, 한글, 공백만 허용
    name_stripped = name_cleaned.strip()
    # 연속된 공백을 하나의 공백으로 (선택적, 현재는 포함하지 않음)
    # name_single_space = " ".join(name_stripped.split())
    # print(f"    [normalize_stock_name] Original: '{name}' -> Normalized: '{name_stripped}'")
    return name_stripped

def predict_info_view(request):
    user = request.user
    initial_stock_code, initial_stock_name, initial_market_name = None, None, None
    is_favorite, stock_name_for_display = False, ""

    stock_query_from_url = request.GET.get('stock_query')
    if stock_query_from_url:
        temp_stock_info = None
        # print(f"[DEBUG][predict_info_view] Received stock_query from URL: {stock_query_from_url}")
        krx_list_all_markets = get_krx_stock_list(market='KOSPI,KOSDAQ') 
        if not krx_list_all_markets:
            print("[DEBUG][predict_info_view] KRX stock list is empty or could not be fetched by get_krx_stock_list!")
        # else:
            # print(f"[DEBUG][predict_info_view] KRX stock list fetched, count: {len(krx_list_all_markets)}")
        
        if stock_query_from_url.isdigit() and len(stock_query_from_url) == 6: 
            # print(f"[DEBUG][predict_info_view] Searching by code: {stock_query_from_url}")
            found_stock = next((s for s in krx_list_all_markets if s.get('Code') == stock_query_from_url), None)
            if found_stock: 
                # print(f"[DEBUG][predict_info_view] Found by code: {found_stock}")
                temp_stock_info = found_stock
        else: 
            # print(f"[DEBUG][predict_info_view] Searching by name: {stock_query_from_url}")
            normalized_query = normalize_stock_name(stock_query_from_url)
            found_stock = next((s for s in krx_list_all_markets if normalized_query in normalize_stock_name(s.get('Name',''))), None)
            if found_stock: 
                # print(f"[DEBUG][predict_info_view] Found by name (normalized): {found_stock}")
                temp_stock_info = found_stock
        
        if temp_stock_info:
            initial_stock_code = temp_stock_info.get('Code')
            initial_stock_name = temp_stock_info.get('Name')
            initial_market_name = temp_stock_info.get('Market','').upper() 
            stock_name_for_display = initial_stock_name
            if user.is_authenticated:
                is_favorite = FavoriteStock.objects.filter(user=user, stock_code=initial_stock_code).exists()
        else: 
            # print(f"[DEBUG][predict_info_view] Stock not found for query: {stock_query_from_url}")
            stock_name_for_display = stock_query_from_url 
    elif user.is_authenticated:
        favorite = FavoriteStock.objects.filter(user=user).order_by('added_at').first()
        if favorite:
            initial_stock_code = favorite.stock_code
            initial_stock_name = favorite.stock_name
            initial_market_name = favorite.market_name.upper() if favorite.market_name else None 
            is_favorite = True
            stock_name_for_display = initial_stock_name
            # print(f"[DEBUG][predict_info_view] Loaded favorite stock: {initial_stock_name}")
            
    context = {
        'ticker': initial_stock_code, 
        'stock_name_for_display': stock_name_for_display,
        'market_name': initial_market_name, 
        'is_favorite': is_favorite,
        'max_favorites': settings.MAX_FAVORITES_PER_USER if hasattr(settings, 'MAX_FAVORITES_PER_USER') else 5
    }
    return render(request, 'predict_info/predict_info.html', context)

def predict_stock_price_ajax(request):
    if request.method == 'POST':
        stock_input_from_frontend = request.POST.get('stock_input')
        analysis_type_from_frontend = request.POST.get('analysis_type', 'technical') 
        
        # print(f"[DEBUG][predict_stock_price_ajax] Received stock_input: {stock_input_from_frontend}, analysis_type: {analysis_type_from_frontend}")

        if not stock_input_from_frontend:
            return JsonResponse({'error': '종목명 또는 코드를 입력해주세요.'}, status=400)

        resolved_stock_code, resolved_stock_name, resolved_market_name = None, None, None
        krx_list = get_krx_stock_list(market='KOSPI,KOSDAQ') 
        if not krx_list:
             print("[ERROR][predict_stock_price_ajax] Failed to get KRX stock list for resolving stock input.")
             return JsonResponse({'error': 'KRX 종목 목록을 가져오는데 실패했습니다. 잠시 후 다시 시도해주세요.'}, status=500)

        if stock_input_from_frontend.isdigit() and len(stock_input_from_frontend) == 6: 
            found_stock = next((s for s in krx_list if s.get('Code') == stock_input_from_frontend), None)
            if found_stock:
                resolved_stock_code = found_stock.get('Code')
                resolved_stock_name = found_stock.get('Name')
                resolved_market_name = found_stock.get('Market','').upper() 
        else: 
            normalized_query = normalize_stock_name(stock_input_from_frontend)
            found_stock = next((s for s in krx_list if normalized_query in normalize_stock_name(s.get('Name',''))), None)
            if found_stock:
                resolved_stock_code = found_stock.get('Code')
                resolved_stock_name = found_stock.get('Name')
                resolved_market_name = found_stock.get('Market','').upper()

        if not resolved_stock_code or not resolved_market_name:
            print(f"[DEBUG][predict_stock_price_ajax] Stock not resolved for input: {stock_input_from_frontend}. This is after checking KRX list.")
            return JsonResponse({'error': f"'{stock_input_from_frontend}'에 해당하는 종목을 찾을 수 없습니다. KRX 목록에 없거나 검색 로직에 문제가 있을 수 있습니다."}, status=400)
        
        # print(f"[DEBUG][predict_stock_price_ajax] Resolved stock: {resolved_stock_name} ({resolved_stock_code}), Market: {resolved_market_name}")

        if analysis_type_from_frontend == 'technical':
            model_type_key_for_load = 'technical_lstm'
        elif analysis_type_from_frontend == 'comprehensive': 
            model_type_key_for_load = 'lstm' 
        else: 
            model_type_key_for_load = 'technical_lstm'
        
        # print(f"[DEBUG][predict_stock_price_ajax] Model type key for load: {model_type_key_for_load}")
        
        try:
            model, scaler_X, scaler_y, model_info = load_model_and_scalers(resolved_market_name, model_type_key=model_type_key_for_load)
            
            if not model or not scaler_X or not scaler_y or not model_info:
                error_msg = f'{resolved_market_name} 시장의 {model_type_key_for_load} 모델/스케일러를 로드할 수 없습니다. 관리자에게 문의하세요. (모델 파일 또는 설정 확인 필요)'
                print(f"[CRITICAL_ERROR_PREDICTION] {error_msg} for stock {resolved_stock_code}")
                return JsonResponse({'error': error_msg}, status=500)
            
            trained_feature_list = model_info.get('trained_feature_list')
            if not trained_feature_list: 
                error_msg = f"모델 '{resolved_market_name}_{model_type_key_for_load}'에 대한 학습된 피처 목록이 정의되지 않았습니다. (model_info 누락)"
                print(f"[CRITICAL_ERROR_PREDICTION] {error_msg} for stock {resolved_stock_code}")
                return JsonResponse({'error': error_msg}, status=500)
            
            log_transformed_input_features = model_info.get('log_transformed_input_features', [])
            time_steps = model_info['time_steps']
            model_was_log_trained_target = model_info['model_was_log_trained']
            market_name_for_feature_calc = model_info.get('market_name_for_features', resolved_market_name) 
            # print(f"[DEBUG][predict_stock_price_ajax] market_name_for_features (from model_info for feature calculation): {market_name_for_feature_calc}")

            latest_data_in_db = StockPrice.objects.filter(stock_code=resolved_stock_code).order_by('-date').first()
            if not latest_data_in_db:
                return JsonResponse({'error': f'{resolved_stock_name}({resolved_stock_code}) DB에 데이터가 없습니다. 데이터 업데이트 후 시도해주세요.'}, status=400)
            
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
                 return JsonResponse({'error': f'{resolved_stock_name}({resolved_stock_code}) 예측을 위한 DB 데이터가 부족합니다 (필요 거래일: {min_records_needed_for_sequence_and_ta}일, 현재 DB 기록: {stock_price_qs.count()}일). 기준일: {prediction_base_date_for_model_input}'}, status=400)

            raw_df_from_db = pd.DataFrame(list(stock_price_qs.values()))
            raw_df_from_db['date'] = pd.to_datetime(raw_df_from_db['date']) 
            raw_df_from_db.set_index('date', inplace=True) 
            
            past_data_for_graph = []
            num_past_days_for_graph = 5 
            if len(raw_df_from_db) >= num_past_days_for_graph:
                past_df_slice = raw_df_from_db.iloc[-(num_past_days_for_graph):] 
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
            df_for_feature_calc = pd.DataFrame(index=raw_df_from_db.index) 
            for calc_col, db_field_name in base_cols_map_for_calc.items():
                if db_field_name in raw_df_from_db:
                    df_for_feature_calc[calc_col] = raw_df_from_db[db_field_name]
                else: 
                    if calc_col in ['Open', 'High', 'Low', 'Close', 'Volume']: 
                         return JsonResponse({'error': f"DB에 필수 컬럼 '{db_field_name}'({calc_col})이(가) 없습니다. 데이터 확인이 필요합니다."}, status=500)
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
                    # print(f"[WARN_PREDICTION] Feature '{feature_name}' is in trained_feature_list but NOT FOUND in processed_df_superset for {resolved_stock_code}.")
                    missing_features_in_superset.append(feature_name)
            
            if missing_features_in_superset:
                error_msg = f"모델 입력에 필요한 피처 중 일부가 생성되지 않았습니다: {missing_features_in_superset}. 피처 생성 로직 또는 trained_feature_list 정의를 확인하세요."
                print(f"[ERROR_PREDICTION] {error_msg} for stock {resolved_stock_code}. Available in superset: {processed_df_superset.columns.tolist()}")
                return JsonResponse({'error': error_msg}, status=500)
            
            for col in final_features_df.columns: 
                if final_features_df[col].dtype == 'object': 
                    try: final_features_df[col] = pd.to_numeric(final_features_df[col], errors='coerce')
                    except Exception as e_conv:
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
                    # else: 
                        # print(f"[WARN_PREDICTION] Column '{col_to_log}' for log transformation not found in final_features_df for {resolved_stock_code}.")
                final_features_df.replace([np.inf, -np.inf], np.nan, inplace=True) 
                final_features_df = final_features_df.ffill().bfill() 

            if final_features_df.isnull().values.any():
                nan_counts = final_features_df.isnull().sum()
                nan_cols_with_counts = nan_counts[nan_counts > 0]
                cols_to_check_for_nan_warn = [c for c in nan_cols_with_counts.index if not c.endswith('_is_nan') and not c.endswith('_is_zero')]
                if cols_to_check_for_nan_warn:
                    print(f"[WARN_PREDICTION] NaNs present before scaling for {resolved_stock_code} in non-indicator columns: {nan_cols_with_counts[cols_to_check_for_nan_warn]}")

            if final_features_df.shape[0] < time_steps: 
                 return JsonResponse({'error': f'최종 피처 데이터가 LSTM 시퀀스 길이({time_steps}일)보다 짧습니다 ({final_features_df.shape[0]}일).'}, status=400)

            if hasattr(scaler_X, 'n_features_in_') and scaler_X.n_features_in_ != final_features_df.shape[1]:
                error_msg = f"스케일러 X와 입력 데이터의 피처 수가 일치하지 않습니다. (스케일러 기대 피처 수: {scaler_X.n_features_in_}개, 현재 데이터 피처 수: {final_features_df.shape[1]}개). 모델 재학습 또는 피처 목록 확인이 필요합니다."
                print(f"[ERROR_PREDICTION] {error_msg} for stock {resolved_stock_code}.")
                if hasattr(scaler_X, 'feature_names_in_'):
                     print(f"    Scaler_X expected features ({len(scaler_X.feature_names_in_)}): {scaler_X.feature_names_in_}")
                print(f"    Actual data columns for scaling ({len(final_features_df.columns.tolist())}): {final_features_df.columns.tolist()}")
                return JsonResponse({'error': error_msg}, status=500)

            try:
                scaled_features = scaler_X.transform(final_features_df.astype(float).values)
            except ValueError as e_scale:
                if 'Input contains NaN' in str(e_scale): 
                    print(f"[WARN_PREDICTION] Scaler X encountered NaNs for {resolved_stock_code}. Attempting fillna(0) as fallback.")
                    scaled_features = scaler_X.transform(final_features_df.fillna(0).astype(float).values)
                else: raise e_scale 

            last_sequence = scaled_features[-time_steps:]
            last_sequence_reshaped = np.reshape(last_sequence, (1, time_steps, last_sequence.shape[1]))
            
            predicted_scaled_values = model.predict(last_sequence_reshaped, verbose=0) 

            try:
                predicted_actual_values = scaler_y.inverse_transform(predicted_scaled_values)
            except ValueError as ve_scaler_y: 
                if predicted_scaled_values.shape[1] == FUTURE_TARGET_DAYS and hasattr(scaler_y, 'n_features_in_') and scaler_y.n_features_in_ == 1:
                    temp_inverted_predictions = []
                    for day_idx in range(predicted_scaled_values.shape[1]):
                        single_day_pred_scaled = predicted_scaled_values[:, day_idx].reshape(-1,1)
                        single_day_pred_actual = scaler_y.inverse_transform(single_day_pred_scaled)
                        temp_inverted_predictions.append(single_day_pred_actual[0,0])
                    predicted_actual_values = np.array([temp_inverted_predictions])
                else: 
                    return JsonResponse({'error': f"Scaler Y 역변환 오류: {ve_scaler_y}. 모델 출력 형태와 스케일러 확인 필요."}, status=500)

            if model_was_log_trained_target:
                predicted_actual_values = np.expm1(predicted_actual_values)
            
            last_actual_close_for_clipping = None
            if not df_for_feature_calc.empty and 'Close' in df_for_feature_calc.columns:
                last_close_val = df_for_feature_calc['Close'].iloc[-1]
                if pd.notna(last_close_val):
                    last_actual_close_for_clipping = float(last_close_val)
                else: 
                    filled_closes = df_for_feature_calc['Close'].ffill()
                    if not filled_closes.empty and pd.notna(filled_closes.iloc[-1]):
                        last_actual_close_for_clipping = float(filled_closes.iloc[-1])
            
            if last_actual_close_for_clipping is None: 
                last_actual_close_for_clipping = 0 

            current_year = prediction_base_date_for_model_input.year
            kr_holidays_for_future = get_kr_holidays([current_year, current_year + 1]) 
            future_dates = get_future_trading_dates_list(prediction_base_date_for_model_input, FUTURE_TARGET_DAYS, kr_holidays_for_future)

            predictions_list = []
            current_reference_price_for_clipping = last_actual_close_for_clipping

            for i in range(FUTURE_TARGET_DAYS):
                predicted_price_raw = predicted_actual_values[0, i]
                clipped_price_final = None
                if pd.notna(predicted_price_raw):
                    price_change_limit_factor = 0.30 
                    upper_limit = current_reference_price_for_clipping * (1 + price_change_limit_factor)
                    lower_limit = current_reference_price_for_clipping * (1 - price_change_limit_factor)
                    clipped_price_val = np.clip(predicted_price_raw, lower_limit, upper_limit)
                    clipped_price_final = round(float(clipped_price_val), 2) 
                else: 
                    clipped_price_final = None
                
                predictions_list.append({
                    'date': future_dates[i].strftime('%Y-%m-%d'),
                    'price': clipped_price_final 
                })
                
                if clipped_price_final is not None:
                    current_reference_price_for_clipping = clipped_price_final
            
            is_favorite_for_user = False
            if request.user.is_authenticated:
                is_favorite_for_user = FavoriteStock.objects.filter(user=request.user, stock_code=resolved_stock_code).exists()

            return JsonResponse({
                'stock_code': resolved_stock_code,
                'stock_name': resolved_stock_name,
                'market_name': resolved_market_name, 
                'prediction_base_date': prediction_base_date_for_model_input.strftime('%Y-%m-%d'),
                'last_actual_close': last_actual_close_for_clipping, 
                'past_data': past_data_for_graph, 
                'predictions': predictions_list,
                'is_favorite': is_favorite_for_user,
                'is_authenticated': request.user.is_authenticated
            })

        except FileNotFoundError as e_fnf:
            print(f"[ERROR_PREDICTION] Model/Scaler file not found for {resolved_stock_code} ({resolved_market_name}, type: {model_type_key_for_load}): {e_fnf}")
            return JsonResponse({'error': f'모델 관련 파일을 찾을 수 없습니다: {e_fnf}. 관리자에게 문의하세요.'}, status=500)
        except Exception as e:
            print(f"[ERROR_PREDICTION] General prediction error for {resolved_stock_code} ({resolved_market_name}, type: {model_type_key_for_load}): {e}\n{traceback.format_exc()}")
            return JsonResponse({'error': f'예측 중 오류가 발생했습니다: {str(e)}'}, status=500)
    else: 
        return JsonResponse({'error': '잘못된 요청입니다 (POST 방식 필요).'}, status=400)

def search_stocks_ajax(request):
    query = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 10)) 
    print(f"[DEBUG][search_stocks_ajax] Received query: '{query}', limit: {limit}")

    if len(query) < 1 and '*' not in query : 
        return JsonResponse([], safe=False)

    results = []
    try:
        all_krx_stocks = get_krx_stock_list(market='KOSPI,KOSDAQ') 
        if not all_krx_stocks:
            print("[ERROR][search_stocks_ajax] KRX stock list is empty or could not be fetched by get_krx_stock_list.")
            return JsonResponse({'error': '전체 종목 목록을 가져오는데 실패했습니다.'}, status=500, safe=False)
        
        print(f"[DEBUG][search_stocks_ajax] Fetched {len(all_krx_stocks)} stocks from KRX list.")
        
        # 특정 코스닥 종목이 KRX 리스트에 어떻게 포함되어 있는지 확인 (예: "알테오젠", "휴젤")
        target_debug_names = ["알테오젠", "휴젤"]
        for stock_item_debug in all_krx_stocks:
            original_name_debug = stock_item_debug.get('Name','')
            normalized_name_debug = normalize_stock_name(original_name_debug)
            if any(debug_name in original_name_debug for debug_name in target_debug_names):
                 print(f"    [INSPECT_KRX_FOR_TARGETS] Original: '{original_name_debug}', Normalized: '{normalized_name_debug}', Code: {stock_item_debug.get('Code')}, Market: {stock_item_debug.get('Market')}")


        if query == '*': 
            results = [
                {"label": f"{s.get('Name','')} ({s.get('Code','')}, {s.get('Market','')})", 
                 "value": s.get('Name',''), 
                 "code": s.get('Code',''), 
                 "market": s.get('Market','').upper()} 
                for s in all_krx_stocks[:limit]
            ]
        else:
            normalized_query = normalize_stock_name(query) 
            print(f"    [DEBUG_SEARCH] Normalized query for search: '{normalized_query}'")
            
            for stock_idx, stock in enumerate(all_krx_stocks):
                stock_name_original = stock.get('Name', '')
                stock_code_original = stock.get('Code', '')
                stock_market_original = stock.get('Market', '')

                normalized_stock_name = normalize_stock_name(stock_name_original) 
                
                # 검색 조건 로깅 (필요시 특정 종목에 대해 더 자세히)
                # if stock_code_original == "196170": # 알테오젠 코드
                #     print(f"        [DEBUG_SEARCH_ALTEOGEN] Query:'{normalized_query}' vs Stock:'{normalized_stock_name}' (Orig:'{stock_name_original}')")

                name_match = False
                if normalized_query and normalized_stock_name: # 둘 다 비어있지 않을 때만 비교
                    name_match = normalized_query in normalized_stock_name
                
                code_match = (query.isdigit() and len(query) == 6 and query == stock_code_original)
                # 정확한 이름 일치도 정규화된 이름으로 비교
                exact_name_match = (not query.isdigit() and normalized_query and normalized_stock_name and normalized_query == normalized_stock_name)


                if name_match or code_match or exact_name_match:
                    results.append({
                        "label": f"{stock_name_original} ({stock_code_original}, {stock_market_original})", 
                        "value": stock_name_original, 
                        "code": stock_code_original, 
                        "market": stock_market_original.upper() 
                    })
                if len(results) >= limit: 
                    break
        
        print(f"[DEBUG][search_stocks_ajax] Found {len(results)} results for query: '{query}'")
        if not results and query != '*': 
             print(f"    No results for '{query}'. Normalized query was '{normalized_query}'. Check KRX list content and normalization logic if stock should exist.")
        
        # cache.set(cache_key, results, timeout=60*5) # 디버깅 중에는 캐시 주석 처리 또는 짧게
    except Exception as e:
        print(f"[ERROR][search_stocks_ajax] Error searching stocks for query '{query}': {e}\n{traceback.format_exc()}")
        return JsonResponse({'error': '종목 검색 중 오류가 발생했습니다.'}, status=500, safe=False)
    
    return JsonResponse(results, safe=False)


MAX_FAVORITES = settings.MAX_FAVORITES_PER_USER if hasattr(settings, 'MAX_FAVORITES_PER_USER') else 5
@login_required
def toggle_favorite_stock_ajax(request):
    if request.method == 'POST':
        try: 
            data = json.loads(request.body)
        except json.JSONDecodeError: 
            return JsonResponse({'status': 'error', 'message': '잘못된 JSON 형식입니다.'}, status=400)
        
        stock_code = data.get('stock_code')
        stock_name = data.get('stock_name')
        market_name = data.get('market_name') 

        if not stock_code or not stock_name or not market_name:
            return JsonResponse({'status': 'error', 'message': '필수 정보(종목코드, 종목명, 시장)가 누락되었습니다.'}, status=400)
        
        user = request.user
        try:
            favorite_obj, created = FavoriteStock.objects.get_or_create(
                user=user, 
                stock_code=stock_code,
                defaults={'stock_name': stock_name, 'market_name': market_name.upper()} 
            )
            if created: 
                if FavoriteStock.objects.filter(user=user).count() > MAX_FAVORITES:
                    favorite_obj.delete() 
                    return JsonResponse({'status': 'error', 'message': f'관심 종목은 최대 {MAX_FAVORITES}개까지 추가할 수 있습니다.', 'is_favorite': False})
                return JsonResponse({'status': 'success', 'message': f"'{stock_name}'을(를) 관심 종목에 추가했습니다.", 'is_favorite': True})
            else: 
                if favorite_obj.stock_name != stock_name or favorite_obj.market_name != market_name.upper():
                    favorite_obj.stock_name = stock_name
                    favorite_obj.market_name = market_name.upper()
                    favorite_obj.save()
                
                favorite_obj.delete()
                return JsonResponse({'status': 'success', 'message': f"'{stock_name}'을(를) 관심 종목에서 삭제했습니다.", 'is_favorite': False})
        except Exception as e:
            print(f"[ERROR][toggle_favorite_stock_ajax] 관심종목 처리 중 오류 ({user.username}, {stock_code}): {e}\n{traceback.format_exc()}")
            return JsonResponse({'status': 'error', 'message': '관심 종목 처리 중 오류가 발생했습니다.'}, status=500)
    
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다 (POST 방식, JSON 필요).'}, status=400)

