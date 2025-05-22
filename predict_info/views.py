# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import pandas as pd
import numpy as np
# import joblib # 함수 내에서 임포트하도록 변경
# import tensorflow as tf # 함수 내에서 임포트하도록 변경
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
from pandas.tseries.offsets import BDay
import os
import traceback
import holidays
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
import json

from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from .models import PredictedStockPrice, StockPrice, FavoriteStock

ML_MODELS_DIR = settings.ML_MODELS_DIR


BASE_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
EXISTING_TA_COLS = [
    'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' 
]
NEW_TA_COLS = ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14']
MACRO_DATA_COLS = ['USD_KRW_Close', 'USD_KRW_Change']
INVESTOR_COLS_MODEL_INPUT = ['Indi', 'Foreign', 'Organ'] 
FUNDAMENTAL_COLS_MODEL_INPUT = ['Marcap', 'PER', 'PBR'] 

def get_feature_columns_for_market(market_name_upper):
    market_specific_cols = []
    if market_name_upper == "KOSPI":
        market_specific_cols = ['KOSPI_Close', 'KOSPI_Change']
    elif market_name_upper == "KOSDAQ":
        market_specific_cols = ['KOSDAQ_Close', 'KOSDAQ_Change']
    else:
        error_message = f"Unsupported market '{market_name_upper}' for feature column definition."
        raise ValueError(error_message) 

    final_columns = (BASE_OHLCV_COLS + EXISTING_TA_COLS + NEW_TA_COLS +
                     market_specific_cols + MACRO_DATA_COLS +
                     INVESTOR_COLS_MODEL_INPUT + FUNDAMENTAL_COLS_MODEL_INPUT)
    
    EXPECTED_FEATURE_COUNT = 30 
    if len(final_columns) != EXPECTED_FEATURE_COUNT:
        error_message = (f"Feature column definition error for {market_name_upper}: "
                         f"{len(final_columns)} features defined, expected {EXPECTED_FEATURE_COUNT}.")
        raise ValueError(error_message) 
    return final_columns

TIME_STEPS = 10 
FUTURE_TARGET_DAYS = 5 
MIN_DATA_DAYS_FOR_PREDICT = 150 

models_cache = {}
scalers_X_cache = {}
scalers_y_cache = {}

def load_model_and_scalers(model_key_name):
    if model_key_name in models_cache:
        print(f"[INFO][load_model_and_scalers] Using cached model/scalers for {model_key_name}.")
        return models_cache[model_key_name], scalers_X_cache[model_key_name], scalers_y_cache[model_key_name]

    market_type = model_key_name.split("_")[0]
    model_file = f"{market_type}_technical_model.keras"
    scaler_x_file = f"{market_type}_technical_scaler_X.joblib"
    scaler_y_file = f"{market_type}_technical_scaler_y.joblib"

    model_path = os.path.join(ML_MODELS_DIR, model_file)
    scaler_x_path = os.path.join(ML_MODELS_DIR, scaler_x_file)
    scaler_y_path = os.path.join(ML_MODELS_DIR, scaler_y_file)

    print(f"[DEBUG][load_model_and_scalers] Attempting to load model from: {model_path}")
    print(f"[DEBUG][load_model_and_scalers] Attempting to load scaler X from: {scaler_x_path}")
    print(f"[DEBUG][load_model_and_scalers] Attempting to load scaler Y from: {scaler_y_path}")

    loaded_model = None
    loaded_scaler_X = None
    loaded_scaler_y = None
    error_messages = [] # 각 파일 로드 실패 메시지를 저장할 리스트

    try:
        # tensorflow와 joblib는 필요할 때만 임포트 (메모리 효율성)
        # 또는 파일 상단에 두어도 무방합니다.
        import tensorflow as tf 
        import joblib

        if os.path.exists(model_path):
            try:
                loaded_model = tf.keras.models.load_model(model_path)
                print(f"[INFO][load_model_and_scalers] Successfully loaded model: {model_file}")
            except Exception as e_model:
                error_messages.append(f"Error loading model file {model_file}: {e_model}")
                print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")
        else:
            error_messages.append(f"Model file not found: {model_path}")
            print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")


        if os.path.exists(scaler_x_path):
            try:
                loaded_scaler_X = joblib.load(scaler_x_path)
                print(f"[INFO][load_model_and_scalers] Successfully loaded scaler X: {scaler_x_file}")
            except Exception as e_scaler_x:
                error_messages.append(f"Error loading scaler X file {scaler_x_file}: {e_scaler_x}")
                print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")
        else:
            error_messages.append(f"Scaler X file not found: {scaler_x_path}")
            print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")

        if os.path.exists(scaler_y_path):
            try:
                loaded_scaler_y = joblib.load(scaler_y_path)
                print(f"[INFO][load_model_and_scalers] Successfully loaded scaler Y: {scaler_y_file}")
            except Exception as e_scaler_y:
                error_messages.append(f"Error loading scaler Y file {scaler_y_file}: {e_scaler_y}")
                print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")
        else:
            error_messages.append(f"Scaler Y file not found: {scaler_y_path}")
            print(f"[ERROR][load_model_and_scalers] {error_messages[-1]}")
        
        # 최종적으로 하나라도 로드되지 않았으면 실패 처리
        if not loaded_model or not loaded_scaler_X or not loaded_scaler_y:
            # 이미 위에서 각 파일별 오류 메시지를 출력했으므로, 여기서는 추가 메시지 없이 None 반환
            # 또는 종합적인 실패 메시지를 한 번 더 출력할 수 있습니다.
            print(f"[ERROR][load_model_and_scalers] One or more components failed to load for {model_key_name}. See specific errors above.")
            return None, None, None
            
        models_cache[model_key_name] = loaded_model
        scalers_X_cache[model_key_name] = loaded_scaler_X
        scalers_y_cache[model_key_name] = loaded_scaler_y
        print(f"[INFO][load_model_and_scalers] Loaded and cached {model_key_name} model and scalers.")
        return loaded_model, loaded_scaler_X, loaded_scaler_y

    except ImportError as ie:
        print(f"[CRITICAL ERROR][load_model_and_scalers] Import error for tensorflow or joblib: {ie}. Please ensure they are installed.")
        traceback.print_exc()
        return None, None, None
    except Exception as e:
        # 예상치 못한 다른 예외 처리
        print(f"[CRITICAL ERROR][load_model_and_scalers] An unexpected exception occurred during loading components for {model_key_name}: {e}")
        traceback.print_exc()
        return None, None, None

# --- 나머지 함수들은 이전과 동일하게 유지 ---

def get_krx_stock_list_predict_cached():
    cache_key = 'krx_stock_list_predict_app_v5_with_market'
    cached_list = cache.get(cache_key)
    if cached_list is not None: return cached_list
    try:
        df_krx_fdr = fdr.StockListing('KRX') 
        if df_krx_fdr.empty: cache.set(cache_key, [], timeout=60*10); return []
        code_col = 'Symbol' if 'Symbol' in df_krx_fdr.columns else 'Code'
        if 'Name' not in df_krx_fdr.columns or code_col not in df_krx_fdr.columns or 'Market' not in df_krx_fdr.columns: 
            cache.set(cache_key, [], timeout=60*10); return []
        df_krx_fdr = df_krx_fdr[['Name', code_col, 'Market']].dropna(subset=['Name', code_col, 'Market'])
        df_krx_fdr = df_krx_fdr.drop_duplicates(subset=[code_col]) 
        stock_list = [{'name': str(r['Name']).strip(), 
                       'code': str(r[code_col]).strip(), 
                       'market': str(r['Market']).strip().upper()} 
                      for _, r in df_krx_fdr.iterrows()]
        cache.set(cache_key, stock_list, timeout=60*60*24) 
        return stock_list
    except Exception as e:
        print(f"[ERROR][get_krx_stock_list_predict_cached] KRX list fetch error: {e}")
        cache.set(cache_key, [], timeout=60*10); return []

def get_stock_info_for_predict(stock_input_query):
    list_of_stock_dicts = get_krx_stock_list_predict_cached()
    if not list_of_stock_dicts: return None, None, None, None 
    found_stock_dict = None
    processed_input = stock_input_query.strip()
    processed_input_upper = processed_input.upper()
    if processed_input.isdigit() and len(processed_input) == 6:
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('code') == processed_input), None)
    else: 
        found_stock_dict = next((s for s in list_of_stock_dicts if s.get('name') and s.get('name').strip().upper() == processed_input_upper), None)
    if found_stock_dict:
        market_name = found_stock_dict.get('market', '') 
        standardized_market = market_name 
        if "KOSDAQ GLOBAL" in standardized_market: standardized_market = "KOSDAQ"
        elif not standardized_market: return None, None, None, None
        return standardized_market, found_stock_dict.get('code'), found_stock_dict.get('name'), market_name 
    return None, None, None, None

def get_future_trading_dates_list(start_date_input, num_days):
    if not isinstance(start_date_input, date_type):
        try: start_date_input = pd.to_datetime(start_date_input).date()
        except: start_date_input = datetime.now().date() 
    kr_holidays_years = list(set([start_date_input.year, start_date_input.year + 1, datetime.now().year + 2])) 
    kr_holidays = holidays.KR(years=kr_holidays_years)
    future_dates = []; current_date_pd = pd.Timestamp(start_date_input)
    current_date_pd += BDay(1) 
    while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays: current_date_pd += BDay(1)
    while len(future_dates) < num_days:
        if current_date_pd.weekday() < 5 and current_date_pd.date() not in kr_holidays: 
            future_dates.append(current_date_pd.date())
        current_date_pd += BDay(1)
        while current_date_pd.weekday() >= 5 or current_date_pd.date() in kr_holidays: current_date_pd += BDay(1)
    return future_dates

def predict_info_view(request):
    context = {'stock_name_for_display': '', 'ticker': '', 'error_message': None, 'is_favorite': False, 'market_name': ''}
    initial_query = request.GET.get('stock_query', '').strip()
    if initial_query:
        market_raw, code, name, original_market_name = get_stock_info_for_predict(initial_query) 
        if code and name: 
            context['stock_name_for_display'] = name
            context['ticker'] = code
            context['market_name'] = original_market_name 
            if request.user.is_authenticated:
                context['is_favorite'] = FavoriteStock.objects.filter(user=request.user, stock_code=code).exists()
        else: 
            context['stock_name_for_display'] = initial_query
            context['error_message'] = f"'{initial_query}'에 대한 종목 정보를 찾을 수 없습니다."
    return render(request, 'predict_info/predict_info.html', context)

def predict_stock_price_ajax(request):
    if request.method == 'POST':
        stock_input = request.POST.get('stock_input', '').strip()
        analysis_type = request.POST.get('analysis_type', 'technical').strip().lower()

        if not stock_input: 
            return JsonResponse({'error': '종목명 또는 종목코드를 입력해주세요.'}, status=400)

        market_raw_standardized, stock_code, stock_name, original_market_name = get_stock_info_for_predict(stock_input)
        
        if not market_raw_standardized or not stock_code: 
            return JsonResponse({'error': f"'{stock_input}'에 해당하는 종목 정보를 찾을 수 없습니다."}, status=400)

        if market_raw_standardized not in ["KOSPI", "KOSDAQ"]: 
            return JsonResponse({'error': f"'{market_raw_standardized}' 시장은 현재 예측을 지원하지 않습니다."}, status=400)

        try:
            latest_prediction_entry = PredictedStockPrice.objects.filter(
                stock_code=stock_code,
                analysis_type=analysis_type
            ).order_by('-prediction_base_date').first()

            if not latest_prediction_entry:
                return JsonResponse({'error': f"'{stock_name}'에 대한 저장된 예측 결과를 찾을 수 없습니다. (배치 작업 미실행 또는 해당 종목 예측 미생성 가능성)"}, status=404)

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
                
                is_favorite_stock = False
                if request.user.is_authenticated:
                    is_favorite_stock = FavoriteStock.objects.filter(user=request.user, stock_code=stock_code).exists()

                return JsonResponse({
                    'stock_code': stock_code, 
                    'stock_name': stock_name, 
                    'market_name': original_market_name, 
                    'analysis_type': analysis_type, 
                    'predictions': predictions_output,
                    'last_data_date': prediction_base_date_from_db.strftime('%Y-%m-%d'), 
                    'is_favorite': is_favorite_stock, 
                    'data_source': 'database_prediction',
                    'is_authenticated': request.user.is_authenticated 
                })
            else:
                return JsonResponse({'error': f"'{stock_name}'에 대한 저장된 예측 데이터 구성에 문제가 있습니다."}, status=404)
        except Exception as e_db_lookup:
            print(f"[ERROR][predict_stock_price_ajax] DB 조회 오류 ({stock_name}): {e_db_lookup}")
            traceback.print_exc()
            return JsonResponse({'error': f"저장된 예측 결과를 가져오는 중 서버 오류 발생."}, status=500)
            
    return JsonResponse({'error': '잘못된 요청입니다 (POST 요청 필요).'}, status=400)

def search_stocks_ajax(request):
    term = request.GET.get('term', '').strip(); limit = int(request.GET.get('limit', 7)) 
    if not term: return JsonResponse([], safe=False)
    all_stocks_list = get_krx_stock_list_predict_cached()
    if not all_stocks_list: 
        return JsonResponse({'error': '종목 목록을 불러오는 데 실패했습니다.'}, status=500)
    results = []; term_upper = term.upper() 
    for item in all_stocks_list:
        stock_name_val = item.get('name',''); stock_code_val = item.get('code',''); market_val = item.get('market','')
        if term_upper in stock_name_val.upper() or term_upper in stock_code_val:
            results.append({'label': f"{stock_name_val} ({stock_code_val}) - {market_val}", 
                            'value': stock_name_val, 'code': stock_code_val, 'market': market_val})
        if len(results) >= limit: break
    return JsonResponse(results, safe=False)

@login_required 
def toggle_favorite_stock_ajax(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            stock_code = data.get('stock_code')
            stock_name = data.get('stock_name')
            market_name = data.get('market_name') 
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': '잘못된 JSON 형식입니다.'}, status=400)

        if not all([stock_code, stock_name, market_name]):
            return JsonResponse({'status': 'error', 'message': '종목 코드, 이름, 시장 정보가 모두 필요합니다.'}, status=400)

        user = request.user
        MAX_FAVORITES = 10

        try:
            favorite_obj, created = FavoriteStock.objects.get_or_create(
                user=user,
                stock_code=stock_code,
                defaults={'stock_name': stock_name, 'market_name': market_name}
            )

            if created: 
                if FavoriteStock.objects.filter(user=user).count() > MAX_FAVORITES:
                    favorite_obj.delete() 
                    return JsonResponse({
                        'status': 'error',
                        'message': f'관심 종목은 최대 {MAX_FAVORITES}개까지 추가할 수 있습니다.',
                        'is_favorite': False 
                    })
                return JsonResponse({
                    'status': 'success', 
                    'message': f"'{stock_name}'을(를) 관심 종목에 추가했습니다.",
                    'is_favorite': True
                })
            else: 
                favorite_obj.delete()
                return JsonResponse({
                    'status': 'success',
                    'message': f"'{stock_name}'을(를) 관심 종목에서 삭제했습니다.",
                    'is_favorite': False
                })
        except Exception as e:
            print(f"[ERROR][toggle_favorite_stock_ajax] 관심종목 처리 중 오류: {e}")
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': '관심 종목 처리 중 오류가 발생했습니다.'}, status=500)

    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다 (POST 방식 필요).'}, status=400)
