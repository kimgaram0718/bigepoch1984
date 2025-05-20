# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
from pandas.tseries.offsets import BDay
import os
import traceback
import holidays
from django.core.cache import cache
from django.contrib.auth.decorators import login_required # 즐겨찾기 기능에 필요할 수 있음

# utils.py의 함수 및 PANDAS_TA_AVAILABLE 플래그 임포트
from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
# models.py에서 필요한 모델 임포트 수정
from .models import PredictedStockPrice, StockPrice 
# FavoriteStock 모델이 있다면 여기에 추가 (예: from .models import FavoriteStock)

# --- 상수 정의 ---
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models')

BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
EXISTING_TA_COLS = [
    'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' 
]
NEW_TA_COLS = ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14']
MACRO_DATA_COLS = ['USD_KRW_Close', 'USD_KRW_Change']
INVESTOR_COLS_STANDARD = ['Indi', 'Foreign', 'Organ']

def get_feature_columns_for_market(market_name_upper):
    market_specific_cols = []
    if market_name_upper == "KOSPI":
        market_specific_cols = ['KOSPI_Close', 'KOSPI_Change']
    elif market_name_upper == "KOSDAQ":
        market_specific_cols = ['KOSDAQ_Close', 'KOSDAQ_Change']
    else:
        error_message = f"지원되지 않는 시장 '{market_name_upper}'에 대한 피처 컬럼을 정의할 수 없습니다."
        print(f"[CRITICAL ERROR][get_feature_columns_for_market] {error_message}")
        raise ValueError(error_message) 

    final_columns = (BASE_OHLCV_COLS +
                     EXISTING_TA_COLS +
                     NEW_TA_COLS +
                     market_specific_cols + 
                     MACRO_DATA_COLS +
                     INVESTOR_COLS_STANDARD)
    
    if len(final_columns) != 27:
        error_message = f"피처 컬럼 정의 오류: {market_name_upper} 시장에 대해 {len(final_columns)}개의 피처가 정의됨 (27개 필요). 코드의 피처 리스트 정의를 확인하세요."
        print(f"[CRITICAL ERROR][get_feature_columns_for_market] {error_message}")
        raise ValueError(error_message) 
    return final_columns

TIME_STEPS = 10 
FUTURE_TARGET_DAYS = 5 
MIN_DATA_DAYS_FOR_PREDICT = 150 # 실시간 예측 시 FDR에서 가져올 최소 기간 (TA 계산용)

models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}

def load_model_and_scalers(model_key_name):
    if model_key_name in models_cache:
        print(f"[INFO][load_model_and_scalers] 캐시에서 {model_key_name} 모델/스케일러 사용.")
        return models_cache[model_key_name], scalers_X_cache[model_key_name], scalers_y_cache[model_key_name]

    market_type = model_key_name.split("_")[0] 
    model_file = f"{market_type}_technical_model.keras" # 분석 유형에 따라 파일명 변경 가능
    scaler_x_file = f"{market_type}_technical_scaler_X.joblib"
    scaler_y_file = f"{market_type}_technical_scaler_y.joblib"

    model_path = os.path.join(ML_MODELS_DIR, model_file)
    scaler_x_path = os.path.join(ML_MODELS_DIR, scaler_x_file)
    scaler_y_path = os.path.join(ML_MODELS_DIR, scaler_y_file)

    loaded_model, loaded_scaler_X, loaded_scaler_y = None, None, None
    
    try:
        if os.path.exists(model_path): loaded_model = tf.keras.models.load_model(model_path)
        else: print(f"[ERROR][load_model_and_scalers] 모델 파일 없음: {model_path}"); return None, None, None
        if os.path.exists(scaler_x_path): loaded_scaler_X = joblib.load(scaler_x_path)
        else: print(f"[ERROR][load_model_and_scalers] X 스케일러 파일 없음: {scaler_x_path}"); return None, None, None
        if os.path.exists(scaler_y_path): loaded_scaler_y = joblib.load(scaler_y_path)
        else: print(f"[ERROR][load_model_and_scalers] Y 스케일러 파일 없음: {scaler_y_path}"); return None, None, None
            
        models_cache[model_key_name] = loaded_model
        scalers_X_cache[model_key_name] = loaded_scaler_X
        scalers_y_cache[model_key_name] = loaded_scaler_y
        print(f"[INFO][load_model_and_scalers] {model_key_name} 모델 및 스케일러 로드/캐싱 완료.")
        return loaded_model, loaded_scaler_X, loaded_scaler_y
        
    except Exception as e:
        print(f"[ERROR][load_model_and_scalers] {model_key_name} 로드 중 심각한 오류: {e}")
        traceback.print_exc()
        return None, None, None

def get_krx_stock_list_predict_cached():
    cache_key = 'krx_stock_list_predict_app_v4' # 캐시 키 버전 관리
    cached_list = cache.get(cache_key)
    if cached_list is not None: return cached_list
    try:
        df_krx_fdr = fdr.StockListing('KRX')
        if df_krx_fdr.empty: cache.set(cache_key, [], timeout=60*10); return [] # 오류 시 짧게 캐시
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'
        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns: 
            cache.set(cache_key, [], timeout=60*10); return []
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col]) # 중복 제거
        stock_list = [{'name': str(r['Name']).strip(), 
                       'code': str(r[code_col]).strip(), 
                       'market': str(r.get('Market', '')).strip().upper()} 
                      for _, r in df_krx_fdr.iterrows()]
        cache.set(cache_key, stock_list, timeout=60*60*24) # 하루 캐시
        return stock_list
    except Exception as e:
        print(f"[ERROR][get_krx_stock_list_predict_cached] KRX 목록 가져오기 오류: {e}")
        cache.set(cache_key, [], timeout=60*10); return []

def get_stock_info_for_predict(stock_input_query):
    list_of_stock_dicts = get_krx_stock_list_predict_cached()
    if not list_of_stock_dicts: return None, None, None
    
    found_stock_dict = None
    processed_input = stock_input_query.strip()
    processed_input_upper = processed_input.upper()

    if processed_input.isdigit() and len(processed_input) == 6: # 종목 코드로 검색
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('code') == processed_input), None)
    else: # 종목명으로 검색
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('name') and s.get('name').strip().upper() == processed_input_upper), None)
    
    if found_stock_dict:
        market_name = found_stock_dict.get('market', '')
        standardized_market = market_name.upper() 
        if "KOSDAQ GLOBAL" in market_name: # KOSDAQ GLOBAL도 KOSDAQ으로 처리
            standardized_market = "KOSDAQ"
        return standardized_market, found_stock_dict.get('code'), found_stock_dict.get('name')
    return None, None, None

def get_latest_stock_data_with_features(stock_code, stock_name, market_name_upper, feature_names_for_model_input):
    """
    실시간 예측이 필요할 경우 (DB에 예측값이 없거나, 다른 용도로) 호출될 수 있는 함수.
    FDR을 통해 최신 데이터를 가져오고 모든 피처를 계산합니다.
    """
    if not PANDAS_TA_AVAILABLE:
        print(f"[CRITICAL ERROR][get_latest_stock_data_with_features] pandas_ta 라이브러리가 없어 피처 생성이 불가능합니다. ({stock_name})")
        return None, None
        
    try:
        print(f"[INFO][get_latest_stock_data_with_features] 실시간 데이터 조회 및 피처 준비: '{stock_name}({stock_code})' (시장: {market_name_upper})...")
        end_date = datetime.now()
        # TA 계산 및 LSTM 입력 시퀀스(TIME_STEPS) 구성을 위해 충분한 과거 데이터 필요
        start_date_fetch = end_date - timedelta(days=MIN_DATA_DAYS_FOR_PREDICT + TIME_STEPS + 60) # 여유분 추가

        df_ohlcv_raw = fdr.DataReader(stock_code, start=start_date_fetch, end=end_date)
        if df_ohlcv_raw.empty or len(df_ohlcv_raw) < (TIME_STEPS + 30): # 최소 데이터 길이 (TA계산+시퀀스)
            print(f"[WARNING][get_latest_stock_data_with_features] OHLCV 데이터 부족 (실시간): {stock_name}({stock_code}), 가져온 데이터 길이: {len(df_ohlcv_raw)}")
            return None, None
        df_ohlcv_raw.index = pd.to_datetime(df_ohlcv_raw.index)
        if 'Change' not in df_ohlcv_raw.columns and 'Close' in df_ohlcv_raw.columns:
            df_ohlcv_raw['Change'] = df_ohlcv_raw['Close'].pct_change()
        elif 'Change' not in df_ohlcv_raw.columns: # 'Close'도 없는 극단적 경우
             df_ohlcv_raw['Change'] = 0.0

        market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
        df_market_index, df_macro_fx = get_market_macro_data(start_date_fetch, end_date, market_fdr_code=market_fdr_code_param)

        df_investor = pd.DataFrame(0.0, index=df_ohlcv_raw.index, columns=INVESTOR_COLS_STANDARD)
        
        df_merged = df_ohlcv_raw.copy()
        if not df_market_index.empty: 
            df_merged = df_merged.join(df_market_index, how='left')
        else: 
            market_cols_to_add = [col for col in feature_names_for_model_input if market_name_upper in col and ("_Close" in col or "_Change" in col)]
            for col in market_cols_to_add: df_merged[col] = np.nan
        
        if not df_macro_fx.empty: 
            df_merged = df_merged.join(df_macro_fx, how='left')
        else: 
            for col in MACRO_DATA_COLS: df_merged[col] = np.nan
        
        if not df_investor.empty: 
            df_merged = df_merged.join(df_investor, how='left')
        else:
            for col in INVESTOR_COLS_STANDARD: 
                df_merged[col] = np.nan
        
        df_merged.ffill(inplace=True)

        df_with_all_ta = calculate_all_features(df_merged.copy(), market_name_upper=market_name_upper)
        
        missing_in_df_after_ta = []
        for col in feature_names_for_model_input:
            if col not in df_with_all_ta.columns:
                df_with_all_ta[col] = np.nan 
                missing_in_df_after_ta.append(col)
        
        if missing_in_df_after_ta:
             print(f"[ERROR][get_latest_stock_data_with_features] 최종 피처셋 구성 중 누락되어 NaN으로 추가된 피처 목록: {missing_in_df_after_ta} ({stock_name}).")

        df_selected_features = df_with_all_ta[feature_names_for_model_input].copy()
        df_selected_features.ffill(inplace=True)
        df_selected_features.bfill(inplace=True) 
        if df_selected_features.isnull().values.any(): 
            nan_cols_final = df_selected_features.columns[df_selected_features.isnull().any()].tolist()
            print(f"[WARNING][get_latest_stock_data_with_features] 최종 선택된 피처에 NaN 존재 (0.0으로 대체): {nan_cols_final} ({stock_name})")
            df_selected_features.fillna(0.0, inplace=True) 

        if len(df_selected_features) < TIME_STEPS: 
            print(f"[ERROR][get_latest_stock_data_with_features] 최종 피처 데이터 길이 부족: {stock_name}, {len(df_selected_features)}일 (필요: {TIME_STEPS}일)")
            return None, None

        recent_features_df = df_selected_features.tail(TIME_STEPS) 
        last_data_date = pd.to_datetime(df_ohlcv_raw.index[-1]).date() 

        print(f"[INFO][get_latest_stock_data_with_features] '{stock_name}({stock_code})' 실시간 피처 준비 완료. 최종 shape: {recent_features_df.shape}. NaN 없음: {not recent_features_df.isnull().values.any()}")
        return recent_features_df, last_data_date

    except Exception as e:
        print(f"[CRITICAL ERROR][get_latest_stock_data_with_features] 실시간 데이터/피처 준비 중 심각한 오류 발생 ({stock_name}): {e}")
        traceback.print_exc()
        return None, None

def get_future_trading_dates_list(start_date_input, num_days):
    if not isinstance(start_date_input, date_type):
        try: start_date_input = pd.to_datetime(start_date_input).date()
        except: start_date_input = datetime.now().date() 
    
    kr_holidays_years = list(set([start_date_input.year, start_date_input.year + 1, datetime.now().year + 2])) 
    kr_holidays = holidays.KR(years=kr_holidays_years)
    
    future_dates = []
    current_date_pd = pd.Timestamp(start_date_input)
    
    current_date_pd += BDay(1) 
    while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays: 
        current_date_pd += BDay(1)
        
    while len(future_dates) < num_days:
        if current_date_pd.weekday() < 5 and current_date_pd.date() not in kr_holidays: 
            future_dates.append(current_date_pd.date())
        current_date_pd += BDay(1)
        while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays:
            current_date_pd += BDay(1)
            
    return future_dates

def predict_info_view(request):
    context = {'stock_name_for_display': '', 'ticker': '', 'error_message': None}
    initial_query = request.GET.get('stock_query', '').strip()
    if initial_query:
        _, code, name = get_stock_info_for_predict(initial_query)
        if code and name: 
            context['stock_name_for_display'] = name
            context['ticker'] = code
        else: 
            context['stock_name_for_display'] = initial_query
            context['error_message'] = f"'{initial_query}'에 대한 종목 정보를 찾을 수 없습니다. 정확한 종목명이나 코드를 입력해주세요."
    return render(request, 'predict_info/predict_info.html', context)

def predict_stock_price_ajax(request):
    if request.method == 'POST':
        print(f"[INFO][predict_stock_price_ajax] POST 요청 수신: {request.POST}")
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type = request.POST.get('analysis_type', 'technical').strip().lower()

        if not stock_input: 
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        market_raw, stock_code, stock_name = get_stock_info_for_predict(stock_input)
        if not market_raw or not stock_code: 
            return JsonResponse({'error': f"'{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요."}, status=400)

        market_name_upper = market_raw.upper(); 
        if "KOSDAQ GLOBAL" in market_name_upper: market_name_upper = "KOSDAQ"
        if market_name_upper not in ["KOSPI", "KOSDAQ"]: 
            return JsonResponse({'error': f"'{market_raw}' 시장은 현재 예측을 지원하지 않습니다. (지원 시장: KOSPI, KOSDAQ)"}, status=400)

        try:
            latest_prediction_entry = PredictedStockPrice.objects.filter(
                stock_code=stock_code,
                analysis_type=analysis_type
            ).order_by('-prediction_base_date').first()

            if not latest_prediction_entry:
                return JsonResponse({'error': f"'{stock_name}'에 대한 저장된 예측 결과를 찾을 수 없습니다. (배치 작업이 아직 실행되지 않았거나, 해당 종목의 예측이 생성되지 않았을 수 있습니다.)"}, status=404)

            prediction_base_date_from_db = latest_prediction_entry.prediction_base_date
            
            predictions_from_db = PredictedStockPrice.objects.filter(
                stock_code=stock_code,
                prediction_base_date=prediction_base_date_from_db,
                analysis_type=analysis_type
            ).order_by('predicted_date')

            if predictions_from_db.exists():
                predictions_output = [{'date': p.predicted_date.strftime('%Y-%m-%d'), 
                                       'price': round(float(p.predicted_price))} 
                                      for p in predictions_from_db]
                
                model_was_log_trained_for_display = getattr(latest_prediction_entry, 'model_log_trained', False) 

                return JsonResponse({
                    'stock_code': stock_code, 
                    'stock_name': stock_name, 
                    'market': market_raw,
                    'analysis_type': analysis_type, 
                    'predictions': predictions_output,
                    'last_data_date': prediction_base_date_from_db.strftime('%Y-%m-%d'), 
                    'model_log_trained': model_was_log_trained_for_display, 
                    'data_source': 'database_prediction' 
                })
            else:
                return JsonResponse({'error': f"'{stock_name}'에 대한 저장된 예측 데이터 구성에 문제가 있습니다. (기준일: {prediction_base_date_from_db})"}, status=404)

        except Exception as e_db_lookup:
            print(f"[ERROR][predict_stock_price_ajax] DB에서 예측 결과 조회 중 오류 ({stock_name}): {e_db_lookup}")
            traceback.print_exc()
            return JsonResponse({'error': f"저장된 예측 결과를 가져오는 중 서버 오류가 발생했습니다."}, status=500)
            
    return JsonResponse({'error': '잘못된 요청입니다 (POST 요청이 필요합니다).'}, status=400)

def search_stocks_ajax(request):
    term = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7)) 
    if not term: return JsonResponse([], safe=False)
    all_stocks_list = get_krx_stock_list_predict_cached()
    if not all_stocks_list: 
        return JsonResponse({'error': '종목 목록을 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.'}, status=500)
    results = []
    term_upper = term.upper() 
    for item in all_stocks_list:
        stock_name_val = item.get('name','')
        stock_code_val = item.get('code','')
        market_val = item.get('market','')
        if term_upper in stock_name_val.upper() or term_upper in stock_code_val:
            results.append({'label': f"{stock_name_val} ({stock_code_val}) - {market_val}", 
                            'value': stock_name_val, 
                            'code': stock_code_val,  
                            'market': market_val    
                           })
        if len(results) >= limit: 
            break
    return JsonResponse(results, safe=False)

# @login_required # 필요시 주석 해제
def toggle_favorite_stock_ajax(request):
    if request.method == 'POST':
        # 이 함수는 현재 실제 즐겨찾기 로직을 구현하지 않습니다.
        # FavoriteStock 모델 및 관련 로직이 필요합니다.
        # 지금은 오류를 방지하기 위한 기본 응답만 반환합니다.
        stock_code = request.POST.get('stock_code')
        if not stock_code:
            return JsonResponse({'status': 'error', 'message': '종목 코드가 필요합니다.'}, status=400)
        
        # 예시: 실제로는 여기서 FavoriteStock 모델을 사용하여 DB 작업을 수행
        # user = request.user
        # is_favorite = # DB에서 확인하는 로직
        # if is_favorite:
        #     # 즐겨찾기 해제
        # else:
        #     # 즐겨찾기 추가
        
        print(f"[INFO][toggle_favorite_stock_ajax] '{stock_code}' 즐겨찾기 토글 요청 (현재 기능 미구현)")
        return JsonResponse({'status': 'success', 'message': f'{stock_code} 즐겨찾기 토글 기능은 아직 구현되지 않았습니다.', 'is_favorite': False}) # 임시로 is_favorite: False 반환
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다.'}, status=400)

