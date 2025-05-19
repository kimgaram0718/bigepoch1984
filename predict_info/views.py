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

# utils.py의 함수 및 PANDAS_TA_AVAILABLE 플래그 임포트
from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from .models import StockPrice # 투자자별 매매동향 데이터용 (실제 연동 필요)

# --- 상수 정의 ---
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models')

# 모델 학습 시 사용된 27개 피처 목록 (순서 중요!)
# pandas-ta 최신 버전(예: 0.3.14b0)은 컬럼명에 길이와 파라미터를 포함 (예: ATR_14, BBL_20_2.0)
# 이 목록은 학습 코드에서 사용한 정확한 컬럼명 및 순서와 100% 일치해야 합니다.
BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
EXISTING_TA_COLS = ['ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
                    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
NEW_TA_COLS = ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14']
MACRO_DATA_COLS = ['USD_KRW_Close', 'USD_KRW_Change']
INVESTOR_COLS_STANDARD = ['Indi', 'Foreign', 'Organ']

def get_feature_columns_for_market(market_name_upper):
    """
    주어진 시장에 맞는 전체 27개 피처 컬럼 리스트를 반환합니다.
    이 함수가 반환하는 컬럼명과 순서는 모델 학습 시 사용된 것과 정확히 일치해야 합니다.
    """
    market_specific_cols = []
    if market_name_upper == "KOSPI":
        market_specific_cols = ['KOSPI_Close', 'KOSPI_Change']
    elif market_name_upper == "KOSDAQ":
        market_specific_cols = ['KOSDAQ_Close', 'KOSDAQ_Change']
    else:
        error_message = f"지원되지 않는 시장 '{market_name_upper}'에 대한 피처 컬럼을 정의할 수 없습니다."
        print(f"[CRITICAL ERROR][get_feature_columns_for_market] {error_message}")
        raise ValueError(error_message)

    # 피처 순서: OHLCV(6) + 기존TA(8) + 신규TA(6) + 시장(2) + 거시(2) + 투자자(3) = 27
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
MIN_DATA_DAYS_FOR_PREDICT = 150 

# --- 모델 및 스케일러 캐싱 ---
models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}

def load_model_and_scalers(model_key_name):
    if model_key_name in models_cache:
        print(f"[INFO][load_model_and_scalers] 캐시에서 {model_key_name} 모델/스케일러 사용.")
        return models_cache[model_key_name], scalers_X_cache[model_key_name], scalers_y_cache[model_key_name]

    market_type = model_key_name.split("_")[0] 
    model_file = f"{market_type}_technical_model.keras" 
    scaler_x_file = f"{market_type}_technical_scaler_X.joblib"
    scaler_y_file = f"{market_type}_technical_scaler_y.joblib"

    model_path = os.path.join(ML_MODELS_DIR, model_file)
    scaler_x_path = os.path.join(ML_MODELS_DIR, scaler_x_file)
    scaler_y_path = os.path.join(ML_MODELS_DIR, scaler_y_file)

    loaded_model, loaded_scaler_X, loaded_scaler_y = None, None, None
    
    try:
        if os.path.exists(model_path):
            loaded_model = tf.keras.models.load_model(model_path)
        else:
            print(f"[ERROR][load_model_and_scalers] 모델 파일 없음: {model_path}")
            return None, None, None # 필수 파일 없으면 실패

        if os.path.exists(scaler_x_path):
            loaded_scaler_X = joblib.load(scaler_x_path)
        else:
            print(f"[ERROR][load_model_and_scalers] X 스케일러 파일 없음: {scaler_x_path}")
            return None, None, None

        if os.path.exists(scaler_y_path):
            loaded_scaler_y = joblib.load(scaler_y_path)
        else:
            print(f"[ERROR][load_model_and_scalers] Y 스케일러 파일 없음: {scaler_y_path}")
            return None, None, None
            
        models_cache[model_key_name] = loaded_model
        scalers_X_cache[model_key_name] = loaded_scaler_X
        scalers_y_cache[model_key_name] = loaded_scaler_y
        print(f"[INFO][load_model_and_scalers] {model_key_name} 모델 및 스케일러 로드/캐싱 완료.")
        return loaded_model, loaded_scaler_X, loaded_scaler_y
        
    except Exception as e:
        print(f"[ERROR][load_model_and_scalers] {model_key_name} 로드 중 심각한 오류: {e}")
        traceback.print_exc()
        return None, None, None

# --- 자동완성 및 종목 정보 조회 (이전과 동일) ---
def get_krx_stock_list_predict_cached():
    cache_key = 'krx_stock_list_predict_app_v3'
    cached_list = cache.get(cache_key)
    if cached_list is not None: return cached_list
    try:
        df_krx_fdr = fdr.StockListing('KRX')
        if df_krx_fdr.empty:
            cache.set(cache_key, [], timeout=60*10); return []
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'
        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns:
            cache.set(cache_key, [], timeout=60*10); return []
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col])
        stock_list = [{'name': str(r['Name']).strip(), 'code': str(r[code_col]).strip(), 'market': str(r.get('Market', '')).strip().upper()} for _, r in df_krx_fdr.iterrows()]
        cache.set(cache_key, stock_list, timeout=60*60*24)
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
    if processed_input.isdigit() and len(processed_input) == 6:
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('code') == processed_input), None)
    else:
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('name') and s.get('name').strip().upper() == processed_input_upper), None)
    if found_stock_dict:
        market_name = found_stock_dict.get('market', '')
        standardized_market = market_name.upper() 
        if "KOSDAQ GLOBAL" in market_name: standardized_market = "KOSDAQ"
        return standardized_market, found_stock_dict.get('code'), found_stock_dict.get('name')
    return None, None, None

# --- 핵심 데이터 준비 함수 ---
def get_latest_stock_data_with_features(stock_code, stock_name, market_name_upper, feature_names_for_model_input):
    """
    최신 주가 데이터를 가져오고 모든 기술적 지표 및 추가 피처를 계산하여 모델 입력용 DataFrame을 반환합니다.
    """
    if not PANDAS_TA_AVAILABLE:
        print(f"[CRITICAL ERROR][get_latest_stock_data_with_features] pandas_ta 라이브러리가 없어 피처 생성이 불가능합니다. ({stock_name})")
        return None, None
        
    try:
        print(f"[INFO][get_latest_stock_data_with_features] '{stock_name}({stock_code})' 데이터 및 피처 준비 시작 (시장: {market_name_upper})...")
        end_date = datetime.now()
        start_date_fetch = end_date - timedelta(days=MIN_DATA_DAYS_FOR_PREDICT + 90) 

        # 1. 개별 종목 OHLCV 데이터 (FDR)
        df_ohlcv_raw = fdr.DataReader(stock_code, start=start_date_fetch, end=end_date)
        if df_ohlcv_raw.empty or len(df_ohlcv_raw) < (TIME_STEPS + 30): 
            print(f"[WARNING][get_latest_stock_data_with_features] OHLCV 데이터 부족: {stock_name}({stock_code}), 가져온 데이터 길이: {len(df_ohlcv_raw)}")
            return None, None
        df_ohlcv_raw.index = pd.to_datetime(df_ohlcv_raw.index)
        if 'Change' not in df_ohlcv_raw.columns and 'Close' in df_ohlcv_raw.columns:
            df_ohlcv_raw['Change'] = df_ohlcv_raw['Close'].pct_change()
        elif 'Change' not in df_ohlcv_raw.columns: 
             df_ohlcv_raw['Change'] = 0.0 # 또는 np.nan 후 ffill/bfill

        # 2. 시장 지수 및 환율 데이터 (utils.get_market_macro_data 사용)
        market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
        df_market_index, df_macro_fx = get_market_macro_data(start_date_fetch, end_date, market_fdr_code=market_fdr_code_param)

        # 3. 투자자별 매매동향 데이터 (Indi, Foreign, Organ) - !!!! 실제 데이터 연동 필수 !!!!
        print(f"[WARNING][get_latest_stock_data_with_features] 투자자별 매매동향(Indi, Foreign, Organ) 데이터가 임시로 0.0으로 채워집니다. ({stock_name}) 실제 데이터 연동이 필요합니다.")
        df_investor = pd.DataFrame(0.0, index=df_ohlcv_raw.index, columns=INVESTOR_COLS_STANDARD)
        # df_investor = df_investor.infer_objects(copy=False) # FutureWarning 방지용 (Pandas 2.x)
                                                          # 또는 fillna(0.0)으로 float 타입 명시


        # 4. 모든 데이터 병합 (OHLCV 데이터프레임 기준, left join)
        df_merged = df_ohlcv_raw.copy()
        # 시장 지수 데이터 병합
        if not df_market_index.empty:
            df_merged = df_merged.join(df_market_index, how='left')
        else: # 시장 데이터가 없는 경우, 해당 컬럼들을 NaN으로 명시적 추가
            market_cols_to_add = [col for col in feature_names_for_model_input if market_name_upper in col and ("_Close" in col or "_Change" in col)]
            for col in market_cols_to_add: df_merged[col] = np.nan
            print(f"[WARNING][get_latest_stock_data_with_features] 시장 지수({market_name_upper}) 데이터가 없어 해당 피처가 NaN으로 시작됩니다. ({stock_name})")

        # 환율 데이터 병합
        if not df_macro_fx.empty:
            df_merged = df_merged.join(df_macro_fx, how='left')
        else: # 거시 데이터가 없는 경우, 해당 컬럼들을 NaN으로 명시적 추가
            for col in MACRO_DATA_COLS: df_merged[col] = np.nan
            print(f"[WARNING][get_latest_stock_data_with_features] 환율 데이터가 없어 해당 피처가 NaN으로 시작됩니다. ({stock_name})")
        
        # 투자자 데이터 병합
        if not df_investor.empty: 
            df_merged = df_merged.join(df_investor, how='left')
        else: # 투자자 데이터가 없는 경우 (실제로는 위에서 0으로 채워짐)
             for col in INVESTOR_COLS_STANDARD: df_merged[col] = np.nan
        
        df_merged.ffill(inplace=True) # 병합 후 NaN은 앞의 값으로 채움
        print(f"[DEBUG][get_latest_stock_data_with_features] 데이터 병합 후 df_merged shape: {df_merged.shape}, NaN 수: {df_merged.isnull().sum().sum()} ({stock_name})")


        # 5. 모든 기술적 지표 계산 (utils.calculate_all_features 사용)
        df_with_all_ta = calculate_all_features(df_merged.copy(), market_name_upper=market_name_upper)
        print(f"[DEBUG][get_latest_stock_data_with_features] TA 계산 후 df_with_all_ta shape: {df_with_all_ta.shape}, NaN 수: {df_with_all_ta.isnull().sum().sum()} ({stock_name})")
        
        # 6. 최종 피처 선택, 순서 맞추기, NaN 처리
        missing_in_df_after_ta = []
        for col in feature_names_for_model_input:
            if col not in df_with_all_ta.columns:
                print(f"[ERROR][get_latest_stock_data_with_features] TA 계산 후에도 필수 피처 '{col}' 컬럼이 누락되었습니다 ({stock_name}). NaN으로 강제 추가합니다. TA 계산 로직 점검 필요.")
                df_with_all_ta[col] = np.nan # 모델 입력 형태를 맞추기 위해 강제 추가
                missing_in_df_after_ta.append(col)
        
        if missing_in_df_after_ta:
             print(f"[ERROR][get_latest_stock_data_with_features] 최종 피처셋 구성 중 누락되어 NaN으로 추가된 피처 목록: {missing_in_df_after_ta} ({stock_name}). 예측 정확도에 심각한 영향이 있을 수 있습니다.")

        df_selected_features = df_with_all_ta[feature_names_for_model_input].copy() # 모델이 기대하는 순서대로 컬럼 선택
        
        # 최종 NaN 처리: ffill 후 bfill. 그래도 남으면 0.0으로 채움.
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

        print(f"[INFO][get_latest_stock_data_with_features] '{stock_name}({stock_code})' 피처 준비 완료. 최종 shape: {recent_features_df.shape}. NaN 없음: {not recent_features_df.isnull().values.any()}")
        return recent_features_df, last_data_date

    except Exception as e:
        print(f"[CRITICAL ERROR][get_latest_stock_data_with_features] 데이터 및 피처 준비 중 심각한 오류 발생 ({stock_name}): {e}")
        traceback.print_exc()
        return None, None

# --- 예측일 계산 함수 (이전과 동일) ---
def get_future_trading_dates_list(start_date_input, num_days):
    if not isinstance(start_date_input, date_type):
        try: start_date_input = pd.to_datetime(start_date_input).date()
        except: start_date_input = datetime.now().date()
    
    kr_holidays_years = list(set([start_date_input.year, start_date_input.year + 1, datetime.now().year + 1])) 
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

# --- Views ---
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
        # 요청 로깅
        print(f"[INFO][predict_stock_price_ajax] POST 요청 수신: {request.POST}")
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type = request.POST.get('analysis_type', 'technical').strip().lower()

        if not PANDAS_TA_AVAILABLE:
             print("[CRITICAL ERROR][predict_stock_price_ajax] pandas_ta 라이브러리 사용 불가. 예측 중단.")
             return JsonResponse({'error': '서버 오류: 기술적 지표 라이브러리(pandas_ta)가 준비되지 않았습니다. 관리자에게 문의하세요.'}, status=500)

        if not stock_input:
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        market_raw, stock_code, stock_name = get_stock_info_for_predict(stock_input)

        if not market_raw or not stock_code:
            return JsonResponse({'error': f"입력하신 '{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요."}, status=400)

        market_name_upper = market_raw.upper()
        if "KOSDAQ GLOBAL" in market_name_upper: market_name_upper = "KOSDAQ"

        if market_name_upper not in ["KOSPI", "KOSDAQ"]:
            print(f"[WARNING][predict_stock_price_ajax] 지원되지 않는 시장: '{market_raw}' (종목: {stock_name})")
            return JsonResponse({'error': f"'{market_raw}' 시장은 현재 예측을 지원하지 않습니다. (지원 시장: KOSPI, KOSDAQ)"}, status=400)

        model_key = f"{market_name_upper.lower()}_{analysis_type}"
        print(f"[INFO][predict_stock_price_ajax] 모델 키 생성: {model_key} (종목: {stock_name})")
        
        try:
            selected_model, selected_scaler_X, selected_scaler_y = load_model_and_scalers(model_key)
            if not all([selected_model, selected_scaler_X, selected_scaler_y]):
                error_msg_model_load = f"{market_name_upper} 시장의 '{analysis_type}' 분석 모델 또는 스케일러를 불러오는 데 실패했습니다. 서버 로그 및 파일 존재 여부를 확인해주세요."
                print(f"[ERROR][predict_stock_price_ajax] {error_msg_model_load}")
                return JsonResponse({'error': error_msg_model_load}, status=500)

            current_market_feature_columns = get_feature_columns_for_market(market_name_upper)
            
            recent_features_df, last_data_date = get_latest_stock_data_with_features(
                stock_code, stock_name, market_name_upper, current_market_feature_columns
            )

            if recent_features_df is None:
                return JsonResponse({'error': f"'{stock_name}({stock_code})'의 예측에 필요한 데이터를 준비하지 못했습니다. 서버 로그를 확인하거나 잠시 후 다시 시도해주세요."}, status=400)

            if len(recent_features_df.columns) != len(current_market_feature_columns):
                error_msg_cols = f"피처 준비 오류: 준비된 피처 수({len(recent_features_df.columns)})와 모델이 기대하는 피처 수({len(current_market_feature_columns)})가 일치하지 않습니다. 서버 설정을 확인하세요."
                print(f"[CRITICAL ERROR][predict_stock_price_ajax] {error_msg_cols} - Prepared: {recent_features_df.columns.tolist()} vs Expected: {current_market_feature_columns}")
                return JsonResponse({'error': error_msg_cols}, status=500)
            
            if len(recent_features_df) != TIME_STEPS:
                error_msg_len = f"'{stock_name}({stock_code})'의 예측에 필요한 시퀀스 길이({TIME_STEPS}일)를 충족하지 못했습니다 (실제 준비된 길이: {len(recent_features_df)}일). 데이터가 충분한지 확인하세요."
                print(f"[ERROR][predict_stock_price_ajax] {error_msg_len}")
                return JsonResponse({'error': error_msg_len}, status=400)
            
            if recent_features_df.isnull().values.any():
                nan_check_df = recent_features_df[recent_features_df.isnull().any(axis=1)]
                print(f"[CRITICAL ERROR][predict_stock_price_ajax] '{stock_name}({stock_code})'의 최종 입력 데이터에 NaN 값이 포함되어 예측을 진행할 수 없습니다. NaN 포함 데이터:\n{nan_check_df}")
                return JsonResponse({'error': f"'{stock_name}({stock_code})'의 최종 입력 데이터에 처리되지 않은 NaN 값이 있습니다. 데이터 정합성을 확인해주세요."}, status=400)

            # 예측 수행
            input_data_for_scaling = recent_features_df[current_market_feature_columns].values
            input_data_scaled = selected_scaler_X.transform(input_data_for_scaling)
            input_data_reshaped = input_data_scaled.reshape(1, TIME_STEPS, len(current_market_feature_columns))
            
            prediction_scaled = selected_model.predict(input_data_reshaped, verbose=0)
            prediction_actual_prices_raw = selected_scaler_y.inverse_transform(prediction_scaled)[0]
            
            future_dates_dt = get_future_trading_dates_list(last_data_date, FUTURE_TARGET_DAYS)
            if len(future_dates_dt) != FUTURE_TARGET_DAYS:
                print(f"[ERROR][predict_stock_price_ajax] 예측일 계산 오류: {len(future_dates_dt)}개 (필요: {FUTURE_TARGET_DAYS}개)")
                return JsonResponse({'error': '예측일 계산에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.'}, status=500)

            predictions_output = []
            current_base_price_for_clipping = recent_features_df['Close'].iloc[-1] 
            
            model_name_or_path = getattr(selected_model, 'name', "") or getattr(selected_model, 'filepath', "")
            model_was_log_trained = "_log_" in model_name_or_path.lower()
            if model_was_log_trained:
                 print(f"[INFO][predict_stock_price_ajax] 모델({model_name_or_path})이 로그 변환으로 학습된 것으로 간주. 예측값에 np.expm1 적용 예정.")

            for i in range(FUTURE_TARGET_DAYS):
                pred_price_day_i_scaled = prediction_actual_prices_raw[i]
                final_pred_price_day_i = np.expm1(pred_price_day_i_scaled) if model_was_log_trained else pred_price_day_i_scaled
                upper_limit = current_base_price_for_clipping * 1.30
                lower_limit = current_base_price_for_clipping * 0.70
                clipped_price = np.clip(final_pred_price_day_i, lower_limit, upper_limit)
                
                predictions_output.append({
                    'date': future_dates_dt[i].strftime('%Y-%m-%d'),
                    'price': round(float(clipped_price))
                })
                current_base_price_for_clipping = clipped_price 

            print(f"[INFO][predict_stock_price_ajax] '{stock_name}({stock_code})' 예측 성공.")
            return JsonResponse({
                'stock_code': stock_code, 'stock_name': stock_name, 'market': market_raw,
                'analysis_type': analysis_type, 'predictions': predictions_output,
                'last_data_date': last_data_date.strftime('%Y-%m-%d') if last_data_date else None,
                'model_log_trained': model_was_log_trained
            })

        except ValueError as ve_main: 
            print(f"[ERROR][predict_stock_price_ajax] 예측 처리 중 ValueError ({stock_name}): {ve_main}")
            traceback.print_exc()
            return JsonResponse({'error': f"예측 처리 중 오류 발생: {ve_main}. 입력 데이터나 스케일러 설정을 확인하세요."}, status=500)
        except Exception as e_main: 
            print(f"[ERROR][predict_stock_price_ajax] AJAX 예측 중 예측할 수 없는 심각한 오류 발생 ({stock_name}): {e_main}")
            traceback.print_exc()
            return JsonResponse({'error': f"예측 처리 중 예기치 않은 서버 오류가 발생했습니다. 관리자에게 문의하세요. (오류: {str(e_main)})"}, status=500)
            
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
            results.append({
                'label': f"{stock_name_val} ({stock_code_val}) - {market_val}", 
                'value': stock_name_val, 
                'code': stock_code_val,  
                'market': market_val    
            })
        if len(results) >= limit: 
            break
    return JsonResponse(results, safe=False)

