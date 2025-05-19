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
from django.core.cache import cache # Django 캐시 사용
# import functools # lru_cache 사용 예시 (여기서는 Django cache 사용) # 주석 처리 또는 삭제

from .utils import calculate_manual_features
from .models import MarketIndex, StockPrice

# --- 상수 정의 ---
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models')

FEATURE_COLUMNS_TRAINING = ['Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'BB_Lower', 'BB_Mid', 'BB_Upper', 'RSI', 'MACD', 'MACD_Hist', 'MACD_Signal']
TIME_STEPS = 10
FUTURE_TARGET_DAYS = 5
MIN_DATA_DAYS_FOR_TA_CALC = 100

# --- 모델 및 스케일러 전역 변수 ---
models = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None
}
scalers_X = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None
}
scalers_y = {
    "kospi_technical": None, "kosdaq_technical": None,
    "kospi_comprehensive": None, "kosdaq_comprehensive": None
}

# --- 모델 및 스케일러 로드 함수 ---
def load_all_models_and_scalers():
    """
    서버 시작 시 모든 모델과 스케일러를 로드합니다.
    """
    markets_config = {
        "kosdaq_technical": {
            "model_file": "kosdaq_technical_model.keras",
            "scaler_x_file": "kosdaq_technical_scaler_X.joblib",
            "scaler_y_file": "kosdaq_technical_scaler_y.joblib"
        },
        "kospi_technical": {
            "model_file": "kospi_technical_model.keras",
            "scaler_x_file": "kospi_technical_scaler_X.joblib",
            "scaler_y_file": "kospi_technical_scaler_y.joblib"
        }
    }

    for model_key, files in markets_config.items():
        try:
            model_path = os.path.join(ML_MODELS_DIR, files["model_file"])
            scaler_x_path = os.path.join(ML_MODELS_DIR, files["scaler_x_file"])
            scaler_y_path = os.path.join(ML_MODELS_DIR, files["scaler_y_file"])

            if os.path.exists(model_path):
                models[model_key] = tf.keras.models.load_model(model_path)
                print(f"[INFO] 모델 로드 성공: {model_path}")
            else:
                print(f"[WARNING] 모델 파일 없음: {model_path} ({model_key})")

            if os.path.exists(scaler_x_path):
                scalers_X[model_key] = joblib.load(scaler_x_path)
                print(f"[INFO] X 스케일러 로드 성공: {scaler_x_path}")
            else:
                print(f"[WARNING] X 스케일러 파일 없음: {scaler_x_path} ({model_key})")

            if os.path.exists(scaler_y_path):
                scalers_y[model_key] = joblib.load(scaler_y_path)
                print(f"[INFO] Y 스케일러 로드 성공: {scaler_y_path}")
            else:
                print(f"[WARNING] Y 스케일러 파일 없음: {scaler_y_path} ({model_key})")
        except Exception as e:
            print(f"[ERROR] {model_key} 모델/스케일러 로드 중 오류: {e}")
            traceback.print_exc()

load_all_models_and_scalers()

# --- 자동완성 및 종목 정보 조회용 헬퍼 함수 ---
def get_krx_stock_list_cached():
    """
    KRX 전체 종목 목록을 DataFrame으로 가져와 캐시합니다. (Django Cache 사용)
    반환 시에는 list of dicts 형태로 변환하여 반환합니다.
    """
    cached_list_of_dicts = cache.get('krx_stock_list_dicts_v2') # 캐시 키 변경
    if cached_list_of_dicts is not None: # None인지 명시적 확인 (빈 리스트도 유효한 캐시 값일 수 있음)
        # print("[INFO] KRX stock list (list of dicts) loaded from cache.")
        return cached_list_of_dicts

    try:
        print("[INFO] Fetching KRX stock list from FDR for autocomplete cache...")
        df_krx_fdr = fdr.StockListing('KRX')
        
        if df_krx_fdr.empty:
            print("[WARNING] FDR StockListing('KRX') returned an empty DataFrame.")
            cache.set('krx_stock_list_dicts_v2', [], timeout=60*10) # 실패 시 짧은 시간 빈 리스트 캐시
            return []

        # 필요한 컬럼만 선택하고 NaN 제거 (원본 DataFrame 컬럼명 사용)
        # FDR의 StockListing 컬럼명은 'Name', 'Symbol' 또는 'Code', 'Market' 등일 수 있음.
        # 일관성을 위해 'Code'를 'Symbol'로 변경하는 경우가 많으므로, 'Symbol'을 우선 확인
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'

        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns:
            print(f"[ERROR] Essential columns ('Name', '{code_col}') not found in FDR StockListing.")
            cache.set('krx_stock_list_dicts_v2', [], timeout=60*10)
            return []
            
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col])

        stock_list_of_dicts = []
        for _, row in df_krx_fdr.iterrows():
            stock_list_of_dicts.append({
                'name': str(row['Name']).strip(),          # 소문자 키 'name'
                'code': str(row[code_col]).strip(),      # 소문자 키 'code'
                'market': str(row.get('Market', '')).strip() # 소문자 키 'market'
            })

        cache.set('krx_stock_list_dicts_v2', stock_list_of_dicts, timeout=60*60*24) # 24시간 캐시
        print(f"[INFO] KRX stock list (list of dicts) fetched and cached. Total: {len(stock_list_of_dicts)} stocks.")
        return stock_list_of_dicts
    except Exception as e:
        print(f"[ERROR] Error fetching or caching KRX stock list: {e}")
        traceback.print_exc()
        cache.set('krx_stock_list_dicts_v2', [], timeout=60*10) # 오류 발생 시 빈 리스트 캐시
        return []


def get_market_info_from_fdr(stock_input):
    """
    캐시된 종목 리스트(list of dicts)를 사용하여 종목 코드/명으로 시장 정보, 코드, 이름을 조회합니다.
    키는 소문자('name', 'code', 'market')를 사용합니다.
    """
    list_of_stock_dicts = get_krx_stock_list_cached()

    if not list_of_stock_dicts: # 캐시된 리스트가 비어있으면 (FDR 조회 실패 포함)
        print(f"[WARNING] KRX stock list is empty. Cannot find stock: '{stock_input}'")
        # 폴백 로직을 여기에 추가하거나, 이 함수 호출 전에 처리하도록 구조화할 수 있음
        # 현재 구조에서는 get_krx_stock_list_cached가 실패 시 빈 리스트를 반환하므로,
        # 여기서 추가적인 FDR 호출 없이 바로 None 반환
        return None, None, None

    found_stock_dict = None
    processed_stock_input = stock_input.strip()
    processed_stock_input_upper = processed_stock_input.upper()

    if processed_stock_input.isdigit() and len(processed_stock_input) == 6:
        # 'code' (소문자) 키로 검색
        found_stock_dict = next((stock_item for stock_item in list_of_stock_dicts if stock_item.get('code') == processed_stock_input), None)
    else:
        # 'name' (소문자) 키로 검색
        found_stock_dict = next((stock_item for stock_item in list_of_stock_dicts
                               if stock_item.get('name') and stock_item.get('name').strip().upper() == processed_stock_input_upper),
                              None)

    if found_stock_dict:
        # 소문자 키로 값 반환
        market_val = found_stock_dict.get('market')
        code_val = found_stock_dict.get('code')
        name_val = found_stock_dict.get('name')
        # print(f"[DEBUG] Found in get_market_info_from_fdr: Name='{name_val}', Code='{code_val}', Market='{market_val}' for input '{stock_input}'")
        return market_val, code_val, name_val
    
    # print(f"[DEBUG] Not found in get_market_info_from_fdr: '{stock_input}'")
    return None, None, None


def get_stock_info_from_db_or_fdr(stock_input_query):
    """DB에서 먼저 종목 정보를 찾고, 없으면 FDR(캐시된 리스트 사용)에서 조회합니다."""
    latest_db_date = StockPrice.objects.order_by('-date').first()
    market, code, name = None, None, None

    if latest_db_date:
        latest_date_val = latest_db_date.date
        stock_db_info = None
        # DB 검색 시 종목 코드 또는 이름으로 조회
        if stock_input_query.isdigit() and len(stock_input_query) == 6:
            stock_db_info = StockPrice.objects.filter(stock_code=stock_input_query, date=latest_date_val).first()
        else:
            # DB의 stock_name 필드와 대소문자 구분 없이 비교
            stock_db_info = StockPrice.objects.filter(stock_name__iexact=stock_input_query.strip(), date=latest_date_val).first()

        if stock_db_info:
            market = stock_db_info.market_name
            code = stock_db_info.stock_code
            name = stock_db_info.stock_name
            # print(f"[DEBUG] Found in DB: {name}({code}) - {market}")
            return market, code, name
    
    # print(f"[DEBUG] Not found in DB for '{stock_input_query}', trying get_market_info_from_fdr...")
    market, code, name = get_market_info_from_fdr(stock_input_query)
    return market, code, name


def get_latest_stock_data_with_features(stock_code, feature_names_for_model_input):
    """특정 종목의 최신 주가 데이터와 기술적 지표를 계산하여 반환합니다."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=MIN_DATA_DAYS_FOR_TA_CALC * 1.7) # 여유분 포함

        df_raw = fdr.DataReader(stock_code, start=start_date, end=end_date)

        if df_raw.empty or len(df_raw) < TIME_STEPS + 20:
            print(f"[WARNING] 데이터 부족 (Raw): {stock_code}, 길이: {len(df_raw)}")
            return None, None

        last_data_date = pd.to_datetime(df_raw.index[-1]).date()
        df_with_ta = calculate_manual_features(df_raw.copy())

        available_cols_in_df = [col for col in feature_names_for_model_input if col in df_with_ta.columns]
        if len(available_cols_in_df) != len(feature_names_for_model_input):
            missing_cols = [col for col in feature_names_for_model_input if col not in available_cols_in_df]
            print(f"[WARNING] 일부 피처가 df_with_ta에 존재하지 않습니다: {missing_cols}. 사용 가능한 피처만 사용합니다.")

        df_selected_features = df_with_ta[available_cols_in_df].ffill().bfill()
        df_processed = df_selected_features.dropna()

        if len(df_processed) < TIME_STEPS:
            print(f"[WARNING] 데이터 부족 (Processed): {stock_code}, 처리 후 길이: {len(df_processed)}")
            return None, None

        recent_features_df = df_processed.tail(TIME_STEPS)
        if not all(col in recent_features_df.columns for col in feature_names_for_model_input):
            final_missing_cols = [col for col in feature_names_for_model_input if col not in recent_features_df.columns]
            print(f"[ERROR] 최종 입력 데이터에 필수 피처 누락: {final_missing_cols}")
            return None, None

        return recent_features_df[feature_names_for_model_input], last_data_date
    except Exception as e:
        print(f"[ERROR] 최신 데이터 가져오기/처리 중 오류 ({stock_code}): {e}")
        traceback.print_exc()
        return None, None

def get_future_trading_dates_list(start_date_input, num_days, country='KR'):
    """주어진 시작일로부터 미래 N일간의 거래일 목록을 반환합니다."""
    if not isinstance(start_date_input, date_type):
        try:
            start_date_input = pd.to_datetime(start_date_input).date()
        except Exception as e_date:
            print(f"[ERROR] 날짜 변환 실패: {start_date_input}. 오류: {e_date}. 오늘 날짜를 기준으로 합니다.")
            start_date_input = datetime.now().date()

    kr_holidays = holidays.KR(years=[start_date_input.year, start_date_input.year + 1])
    future_dates = []
    current_date_pd = pd.Timestamp(start_date_input)

    while len(future_dates) < num_days:
        current_date_pd += BDay(1)
        if current_date_pd.date() not in kr_holidays:
            future_dates.append(current_date_pd.date())
    return future_dates

# --- Views ---
def predict_info_view(request):
    """예측 정보 페이지를 렌더링합니다."""
    context = {
        'stock_name_for_display': '',
        'ticker': '',
        'error_message': None,
        'prediction_indices': [],
        'top5_kospi_gainers': [],
        'top5_kosdaq_gainers': [],
        'prediction_tickers': [],
        'recommended_stocks': [],
        'top_contents': [],
    }

    latest_market_date = MarketIndex.objects.order_by('-date').first()
    if latest_market_date:
        latest_date_val = latest_market_date.date
        kospi_index = MarketIndex.objects.filter(market_name='KOSPI', date=latest_date_val).first()
        kosdaq_index = MarketIndex.objects.filter(market_name='KOSDAQ', date=latest_date_val).first()
        if kospi_index:
            context['prediction_indices'].append({
                'name': '코스피', 'date_display': latest_date_val.strftime('%Y-%m-%d'),
                'close_price': kospi_index.close_price, 'change_value': kospi_index.change_value,
                'change_percent': kospi_index.change_percent
            })
        if kosdaq_index:
            context['prediction_indices'].append({
                'name': '코스닥', 'date_display': latest_date_val.strftime('%Y-%m-%d'),
                'close_price': kosdaq_index.close_price, 'change_value': kosdaq_index.change_value,
                'change_percent': kosdaq_index.change_percent
            })

    latest_stock_date = StockPrice.objects.order_by('-date').first()
    if latest_stock_date:
        latest_date_val = latest_stock_date.date
        kospi_top5 = StockPrice.objects.filter(market_name='KOSPI', date=latest_date_val, change_percent__isnull=False).order_by('-change_percent')[:5]
        kosdaq_top5 = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_date_val, change_percent__isnull=False).order_by('-change_percent')[:5]
        context['top5_kospi_gainers'] = [{'name': s.stock_name, 'code': s.stock_code, 'change': s.change_percent, 'close': s.close_price} for s in kospi_top5]
        context['top5_kosdaq_gainers'] = [{'name': s.stock_name, 'code': s.stock_code, 'change': s.change_percent, 'close': s.close_price} for s in kosdaq_top5]

    return render(request, 'predict_info/predict_info.html', context)


def predict_stock_price_ajax(request):
    """AJAX 요청을 통해 특정 종목의 주가를 예측하여 JSON으로 반환합니다."""
    if request.method == 'POST':
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type = request.POST.get('analysis_type', 'technical').strip().lower()

        if not stock_input:
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        # print(f"[DEBUG] predict_stock_price_ajax: Received stock_input='{stock_input}'")
        market_raw, stock_code, stock_name = get_stock_info_from_db_or_fdr(stock_input)
        # print(f"[DEBUG] predict_stock_price_ajax: From DB/FDR: market='{market_raw}', code='{stock_code}', name='{stock_name}'")


        if not market_raw or not stock_code:
            return JsonResponse({'error': f"'{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다. 정확한 종목명 또는 6자리 코드를 입력해주세요."}, status=400)

        market = market_raw.lower()
        model_key = f"{market}_{analysis_type}"

        selected_model = models.get(model_key)
        selected_scaler_X = scalers_X.get(model_key)
        selected_scaler_y = scalers_y.get(model_key)

        if not all([selected_model, selected_scaler_X, selected_scaler_y]):
            error_msg = f"{market_raw} 시장의 '{analysis_type}' 분석 모델 또는 스케일러를 로드할 수 없습니다. "
            if analysis_type == "comprehensive" and not models.get(f"{market}_comprehensive"):
                error_msg += "종합 분석 모델은 현재 지원되지 않거나 로드되지 않았습니다."
            else:
                error_msg += "서버 설정을 확인해주세요. (모델 파일명, 경로 등)"
            return JsonResponse({'error': error_msg}, status=500)

        recent_features_df, last_data_date = get_latest_stock_data_with_features(stock_code, FEATURE_COLUMNS_TRAINING)

        if recent_features_df is None or len(recent_features_df) != TIME_STEPS:
            return JsonResponse({'error': f"'{stock_name}({stock_code})'의 예측에 필요한 최근 {TIME_STEPS}일치 데이터를 준비하지 못했습니다. 데이터가 충분한지, 피처 생성에 문제가 없는지 확인해주세요."}, status=400)

        try:
            input_data_scaled = selected_scaler_X.transform(recent_features_df.values)
            input_data_reshaped = input_data_scaled.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS_TRAINING))
            prediction_scaled = selected_model.predict(input_data_reshaped, verbose=0)
            prediction_actual_prices = selected_scaler_y.inverse_transform(prediction_scaled)[0]
            future_dates_dt = get_future_trading_dates_list(last_data_date, FUTURE_TARGET_DAYS)

            predictions_output = []
            for i in range(FUTURE_TARGET_DAYS):
                predictions_output.append({
                    'date': future_dates_dt[i].strftime('%Y-%m-%d'),
                    'price': round(float(prediction_actual_prices[i]))
                })

            # print(f"[INFO] AJAX 예측 성공: {stock_name}({stock_code}), 유형: {analysis_type}")
            return JsonResponse({
                'stock_code': stock_code,
                'stock_name': stock_name, # 실제 찾은 종목명으로 응답
                'market': market_raw,
                'analysis_type': analysis_type,
                'predictions': predictions_output,
                'last_data_date': last_data_date.strftime('%Y-%m-%d') if last_data_date else None
            })

        except Exception as e:
            print(f"[ERROR] AJAX 예측 중 오류 ({stock_name}, {analysis_type}): {e}")
            traceback.print_exc()
            return JsonResponse({'error': f"예측 처리 중 서버 오류 발생: {str(e)}"}, status=500)

    return JsonResponse({'error': '잘못된 요청입니다 (POST 요청 필요).'}, status=400)


def search_stocks_ajax(request):
    """AJAX 요청을 통해 종목을 검색하고 자동완성 목록을 JSON으로 반환합니다."""
    term = request.GET.get('term', '').strip()
    limit = int(request.GET.get('limit', 7))

    if not term:
        return JsonResponse([], safe=False)

    all_stocks_list_of_dicts = get_krx_stock_list_cached()
    if not all_stocks_list_of_dicts: # 캐시 목록이 비어있으면 (FDR 조회 실패 등)
        return JsonResponse({'error': '종목 목록을 불러올 수 없습니다. 잠시 후 다시 시도해주세요.'}, status=500)


    results = []
    term_upper = term.upper()

    for stock_dict in all_stocks_list_of_dicts:
        # 'name'과 'code' 키를 사용하여 검색 (모두 소문자 키로 저장되어 있음)
        stock_name_val = stock_dict.get('name', '') 
        stock_code_val = stock_dict.get('code', '')

        match = False
        if term_upper in stock_name_val.upper():
            match = True
        elif term_upper in stock_code_val: # 코드는 보통 숫자이므로 upper 불필요
            match = True
        
        if match:
            results.append({
                'label': f"{stock_name_val} ({stock_code_val}) - {stock_dict.get('market', '')}",
                'value': stock_name_val, # 자동완성 선택 시 input에 들어갈 값
                'code': stock_code_val
            })

        if len(results) >= limit:
            break

    return JsonResponse(results, safe=False)
