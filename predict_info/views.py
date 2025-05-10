# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
# import FinanceDataReader as fdr # 직접 호출 대신 DB 사용 또는 필요시 로컬 임포트
from datetime import datetime, timedelta, date as date_type # date를 date_type으로 임포트하여 모호성 제거
from pandas.tseries.offsets import BDay
import os
import json
from .utils import calculate_manual_features
from .models import MarketIndex, StockPrice # DB 모델 임포트
import re
import traceback
import holidays
# import yfinance as yf # 직접 호출 대신 DB 사용
from django.utils import timezone

# --- 모델 및 스케일러 경로 설정 (기존과 동일) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MARKET = 'KOSDAQ' # 또는 'KOSPI', 학습된 모델에 맞게 설정
MODEL_SAVE_DIR = os.path.join(APP_DIR, 'ml_models')
SCALER_X_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_scaler_x.joblib')
SCALER_Y_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_scaler_y.joblib')
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_lstm_best_model.keras')

# --- 모델 학습 시 사용된 상수 (기존과 동일) ---
SEQUENCE_LENGTH = 60
PREDICT_DAYS = 7
MIN_DATA_DAYS_FOR_FEATURE_CALC = 150
DEFAULT_FEATURE_NAMES = [
    'Open', 'High', 'Low', 'Volume', 'Value', 'Change', 'ATR',
    'BB_Lower', 'BB_Mid', 'BB_Upper', 'RSI', 'MACD', 'MACD_Hist', 'MACD_Signal'
]

# --- 모델 및 스케일러 로드 (기존과 동일) ---
MODEL = None
SCALER_X = None
SCALER_Y_PCT_CHANGE = None
FEATURE_NAMES_FROM_SCALER = None

print(f"--- [DEBUG] predict_info/views.py 모듈 로딩 시작 ---")
try:
    if os.path.exists(BEST_MODEL_PATH):
        MODEL = tf.keras.models.load_model(BEST_MODEL_PATH)
        print(f"[INFO] LSTM 모델 로드 성공: {BEST_MODEL_PATH}")
    else: print(f"[ERROR] 모델 파일 없음: {BEST_MODEL_PATH}")

    if os.path.exists(SCALER_X_PATH):
        SCALER_X = joblib.load(SCALER_X_PATH)
        print(f"[INFO] X 스케일러 로드 성공: {SCALER_X_PATH}")
        try: FEATURE_NAMES_FROM_SCALER = SCALER_X.feature_names_in_.tolist()
        except AttributeError: FEATURE_NAMES_FROM_SCALER = DEFAULT_FEATURE_NAMES; print("[WARNING] X 스케일러 피처 이름 로드 실패. 기본값 사용.")
    else: print(f"[ERROR] X 스케일러 파일 없음: {SCALER_X_PATH}"); FEATURE_NAMES_FROM_SCALER = DEFAULT_FEATURE_NAMES

    if os.path.exists(SCALER_Y_PATH):
        SCALER_Y_PCT_CHANGE = joblib.load(SCALER_Y_PATH)
        print(f"[INFO] Y(변동률) 스케일러 로드 성공: {SCALER_Y_PATH}")
    else: print(f"[ERROR] Y 스케일러 파일 없음: {SCALER_Y_PATH}")

except Exception as e: print(f"[CRITICAL ERROR] 초기 로딩 중 오류: {e}"); traceback.print_exc()


def get_latest_data_date_from_db(model_class):
    latest_entry = model_class.objects.order_by('-date').first()
    return latest_entry.date if latest_entry else None

def get_stock_code_from_query(query_str):
    query_str_cleaned = query_str.strip()
    print(f"[DEBUG get_stock_code_from_query] Received query: '{query_str}', Cleaned query: '{query_str_cleaned}'")
    latest_date = get_latest_data_date_from_db(StockPrice)
    if not latest_date:
        print("[WARNING get_stock_code_from_query] No stock data in DB to search from.")
        return None, None
    stock_info = None
    if re.match(r'^\d{6}$', query_str_cleaned):
        stock_info = StockPrice.objects.filter(stock_code=query_str_cleaned, date=latest_date).first()
        if stock_info:
            return stock_info.stock_code, stock_info.stock_name
    stock_info = StockPrice.objects.filter(stock_name__icontains=query_str_cleaned, date=latest_date).first()
    if stock_info:
        return stock_info.stock_code, stock_info.stock_name
    print(f"[DEBUG get_stock_code_from_query] Name or Code '{query_str_cleaned}' not found in DB for date {latest_date}.")
    return None, None

def get_future_trading_dates(start_date_input, num_days, country='KR'):
    if isinstance(start_date_input, datetime):
        current_check_date = start_date_input.date()
    elif isinstance(start_date_input, date_type):
        current_check_date = start_date_input
    elif hasattr(start_date_input, 'date') and callable(getattr(start_date_input, 'date')):
        current_check_date = start_date_input.date()
        if not isinstance(current_check_date, date_type):
             current_check_date = pd.to_datetime(start_date_input).date()
    else: 
        try:
            current_check_date = pd.to_datetime(start_date_input).date()
        except Exception as e:
            print(f"Error converting start_date_input to date: {e}")
            current_check_date = timezone.now().date()

    kr_holidays = holidays.KR(years=[current_check_date.year, current_check_date.year + 1, current_check_date.year + 2])
    future_dates = []
    current_date_pd = pd.Timestamp(current_check_date)
    
    while len(future_dates) < num_days:
        current_date_pd += BDay(1)
        if current_date_pd.date() not in kr_holidays:
            future_dates.append(current_date_pd.date())
    return future_dates


def predict_info_view(request):
    query = request.GET.get('query', '휴젤') # 기본 검색어 (예: 휴젤)
    print(f"[DEBUG predict_info_view] Received query parameter: '{query}'")
    
    context = {
        'stock_name_for_display': query,
        'ticker': None,
        'error_message': None,
        'initial_predictions': None,
        'prediction_error': None,
        'prediction_indices': [], # 시장 지수 정보
        'top5_kospi_gainers': [], # 코스피 TOP5
        'top5_kosdaq_gainers': [], # 코스닥 TOP5
        # 아래는 기존 플레이스홀더, 필요시 DB 데이터로 교체하거나 제거
        'prediction_tickers': [], # 예: [{'name': '삼성전자', 'price': 'DB조회필요', 'change': 'DB조회필요'}]
        'recommended_stocks': [], # 예: [{'rank': 1, 'name': '추천종목A(DB)'}]
        'top_contents': [],       # 예: [{'rank': 1, 'title': '오늘의 시장 분석(DB)', 'link': '#'}]
    }

    latest_market_date = get_latest_data_date_from_db(MarketIndex)
    latest_stock_date = get_latest_data_date_from_db(StockPrice)

    if latest_market_date:
        kospi_index_data = MarketIndex.objects.filter(market_name='KOSPI', date=latest_market_date).first()
        kosdaq_index_data = MarketIndex.objects.filter(market_name='KOSDAQ', date=latest_market_date).first()

        if kospi_index_data:
            context['prediction_indices'].append({
                'name': '코스피', # '코스피 지수' 대신 간단히
                'date_display': latest_market_date.strftime('%Y-%m-%d'),
                'close_price': kospi_index_data.close_price, # 전일 종가
                'change_value': kospi_index_data.change_value, # 전일 대비 변동폭
                'change_percent': kospi_index_data.change_percent, # 전일 대비 등락률
            })
        if kosdaq_index_data:
            context['prediction_indices'].append({
                'name': '코스닥', # '코스닥 지수' 대신 간단히
                'date_display': latest_market_date.strftime('%Y-%m-%d'),
                'close_price': kosdaq_index_data.close_price, # 전일 종가
                'change_value': kosdaq_index_data.change_value, # 전일 대비 변동폭
                'change_percent': kosdaq_index_data.change_percent, # 전일 대비 등락률
            })
    else:
        # 시장 지수 데이터가 없을 경우, prediction_indices는 빈 리스트로 유지됨
        context['error_message'] = "시장 지수 데이터가 DB에 없습니다. 데이터 수집 스크립트를 실행해주세요."

    if latest_stock_date:
        kospi_gainers = StockPrice.objects.filter(market_name='KOSPI', date=latest_stock_date, change_percent__isnull=False).order_by('-change_percent')[:5]
        context['top5_kospi_gainers'] = [{'name': s.stock_name, 'change': s.change_percent, 'close': s.close_price, 'code': s.stock_code} for s in kospi_gainers]
        
        kosdaq_gainers = StockPrice.objects.filter(market_name='KOSDAQ', date=latest_stock_date, change_percent__isnull=False).order_by('-change_percent')[:5]
        context['top5_kosdaq_gainers'] = [{'name': s.stock_name, 'change': s.change_percent, 'close': s.close_price, 'code': s.stock_code} for s in kosdaq_gainers]
    else:
        # 개별 종목 데이터가 없을 경우, top5_kospi_gainers와 top5_kosdaq_gainers는 빈 리스트로 유지됨
        current_error = context.get('error_message', '')
        context['error_message'] = (current_error + " " if current_error else "") + "개별 종목 데이터가 DB에 없습니다."
        
    stock_code, stock_name = get_stock_code_from_query(query)
    print(f"[DEBUG predict_info_view] Resolved stock_code: {stock_code}, stock_name: {stock_name} from DB")

    if not stock_code:
        context['prediction_error'] = f"종목 '{query}' 정보를 DB에서 찾을 수 없어 초기 예측을 수행할 수 없습니다."
        if stock_name: context['stock_name_for_display'] = stock_name # 이름이라도 있으면 표시
    else:
        context['stock_name_for_display'] = stock_name
        context['ticker'] = stock_code
        
        if MODEL and SCALER_X and SCALER_Y_PCT_CHANGE and FEATURE_NAMES_FROM_SCALER:
            print(f"[INFO] predict_info_view: 종목 {stock_code}({stock_name}) 초기 예측 시작...")
            try:
                import FinanceDataReader as fdr 

                days_to_fetch = SEQUENCE_LENGTH + MIN_DATA_DAYS_FOR_FEATURE_CALC
                end_date_pred = datetime.now() 
                start_date_pred = end_date_pred - timedelta(days=days_to_fetch * 2.5) 
                
                print(f"[DEBUG predict_info_view] Fetching data for prediction from {start_date_pred.strftime('%Y-%m-%d')} to {end_date_pred.strftime('%Y-%m-%d')}")
                df_pred_raw = fdr.DataReader(stock_code, start=start_date_pred, end=end_date_pred)

                if df_pred_raw.empty or len(df_pred_raw) < MIN_DATA_DAYS_FOR_FEATURE_CALC:
                    context['prediction_error'] = f"초기 예측 위한 데이터 부족 (최소 {MIN_DATA_DAYS_FOR_FEATURE_CALC}일 필요, 현재 {len(df_pred_raw)}일)."
                else:
                    last_known_date_for_pred = df_pred_raw.index[-1] 
                    
                    if not isinstance(last_known_date_for_pred, (datetime, date_type, pd.Timestamp)):
                         last_known_date_for_pred = pd.to_datetime(last_known_date_for_pred).date()

                    df_pred_features_calc = calculate_manual_features(df_pred_raw.copy())
                    valid_feature_names = [f for f in FEATURE_NAMES_FROM_SCALER if f in df_pred_features_calc.columns]
                    
                    if len(valid_feature_names) != len(FEATURE_NAMES_FROM_SCALER):
                        missing_f = [f for f in FEATURE_NAMES_FROM_SCALER if f not in valid_feature_names]
                        context['prediction_error'] = f"피처 계산 후 일부 필수 피처 누락: {missing_f}"
                    else:
                        df_for_prediction_input = df_pred_features_calc[valid_feature_names].copy()
                        df_for_prediction_input = df_for_prediction_input.ffill().bfill()
                        
                        if len(df_for_prediction_input) < SEQUENCE_LENGTH:
                            context['prediction_error'] = f"피처 생성 및 NaN 처리 후 데이터 부족 (필요: {SEQUENCE_LENGTH}일)."
                        else:
                            last_sequence_features = df_for_prediction_input.iloc[-SEQUENCE_LENGTH:]
                            if last_sequence_features.isnull().values.any():
                                nan_cols = last_sequence_features.columns[last_sequence_features.isnull().any()].tolist()
                                context['prediction_error'] = f"예측 사용할 마지막 시퀀스에 NaN 포함 (컬럼: {nan_cols})."
                            else:
                                last_actual_close_for_pred = df_pred_raw['Close'].iloc[-1]
                                
                                scaled_features_for_pred = SCALER_X.transform(last_sequence_features)
                                input_sequence_for_pred = np.expand_dims(scaled_features_for_pred, axis=0)
                                predicted_scaled_pct_changes = MODEL.predict(input_sequence_for_pred, verbose=0)[0]
                                predicted_pct_changes = SCALER_Y_PCT_CHANGE.inverse_transform(predicted_scaled_pct_changes.reshape(-1, 1)).flatten()
                                
                                future_prices = []
                                current_price = last_actual_close_for_pred
                                # last_known_date_for_pred는 위에서 이미 타입 처리됨
                                future_dates_dt = get_future_trading_dates(last_known_date_for_pred, PREDICT_DAYS, country='KR')
                                future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
                                
                                for pct_change_val in predicted_pct_changes: # 변수명 변경 (pct_change -> pct_change_val)
                                    current_price = current_price * (1 + pct_change_val / 100)
                                    future_prices.append(round(current_price)) # 정수형으로 변환
                                context['initial_predictions'] = [{'date': date, 'price': price} for date, price in zip(future_dates_str, future_prices)]
                                print(f"[INFO] predict_info_view: 종목 {stock_code}({stock_name}) 초기 예측 완료.")
            except ImportError:
                 context['prediction_error'] = "FinanceDataReader 모듈을 찾을 수 없어 예측 데이터를 로드할 수 없습니다."
            except Exception as e_pred:
                context['prediction_error'] = f"초기 예측 중 오류: {str(e_pred)}"
                print(f"[ERROR] predict_info_view: {context['prediction_error']}"); traceback.print_exc()
        else:
            context['prediction_error'] = "모델/스케일러 미준비로 초기 예측 불가."
            print(f"[WARNING] predict_info_view: {context['prediction_error']}")
            
    return render(request, 'predict_info/predict_info.html', context)


def predict_stock_price_ajax(request):
    if request.method == 'POST':
        stock_code_input = request.POST.get('stock_code')
        if not stock_code_input: return JsonResponse({'error': '종목 코드 또는 이름을 입력해주세요.'}, status=400)
        
        stock_code, stock_name_from_db = get_stock_code_from_query(stock_code_input)
        if not stock_code: return JsonResponse({'error': f"'{stock_code_input}' 종목 정보 없음."}, status=400)
        
        if MODEL is None or SCALER_X is None or SCALER_Y_PCT_CHANGE is None or FEATURE_NAMES_FROM_SCALER is None:
            return JsonResponse({'error': '모델/스케일러 미준비. 서버 로그 확인.'}, status=500)

        try:
            import FinanceDataReader as fdr
            days_to_fetch = SEQUENCE_LENGTH + MIN_DATA_DAYS_FOR_FEATURE_CALC
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_fetch * 2.5)
            
            print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code}({stock_name_from_db}) 데이터 로딩 중...")
            df_raw = fdr.DataReader(stock_code, start_date, end_date)
            
            if df_raw.empty or len(df_raw) < MIN_DATA_DAYS_FOR_FEATURE_CALC:
                return JsonResponse({'error': f'{stock_name_from_db}({stock_code}) 예측 데이터 부족 (최소 {MIN_DATA_DAYS_FOR_FEATURE_CALC}일 필요).'}, status=400)
            
            last_known_date_for_ajax = df_raw.index[-1]
            if not isinstance(last_known_date_for_ajax, (datetime, date_type, pd.Timestamp)):
                last_known_date_for_ajax = pd.to_datetime(last_known_date_for_ajax).date()

        except ImportError:
            return JsonResponse({'error': 'FinanceDataReader 모듈을 찾을 수 없어 예측 데이터를 로드할 수 없습니다.'}, status=500)
        except Exception as e: 
            return JsonResponse({'error': f'{stock_name_from_db}({stock_code}) 데이터 로딩 실패: {str(e)}'}, status=400)

        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 피처 계산 중...")
        df_features_calculated = calculate_manual_features(df_raw.copy())
        try:
            valid_feature_names_for_input = [f for f in FEATURE_NAMES_FROM_SCALER if f in df_features_calculated.columns]
            if len(valid_feature_names_for_input) != len(FEATURE_NAMES_FROM_SCALER):
                missing_cols = [f for f in FEATURE_NAMES_FROM_SCALER if f not in valid_feature_names_for_input]
                return JsonResponse({'error': f'피처 계산 후 일부 피처 누락. 누락: {missing_cols}'}, status=500)
            df_for_prediction_input = df_features_calculated[valid_feature_names_for_input].copy()
            df_for_prediction_input = df_for_prediction_input.ffill().bfill()
        except KeyError as e: return JsonResponse({'error': f'피처 선택 중 오류: {str(e)}'}, status=500)

        if len(df_for_prediction_input) < SEQUENCE_LENGTH:
            return JsonResponse({'error': f'{stock_name_from_db}({stock_code}) 피처 생성 및 NaN 처리 후 예측 데이터 부족.'}, status=400)

        last_sequence_features = df_for_prediction_input.iloc[-SEQUENCE_LENGTH:]
        if last_sequence_features.isnull().values.any():
            nan_cols = last_sequence_features.columns[last_sequence_features.isnull().any()].tolist()
            print(f"[WARNING] predict_stock_price_ajax: 마지막 시퀀스에 NaN 포함 for {stock_code} (컬럼: {nan_cols})")
            return JsonResponse({'error': f'{stock_name_from_db}({stock_code}) 예측용 데이터에 NaN 포함.'}, status=400)

        last_actual_close = df_raw['Close'].iloc[-1]
        
        try: scaled_features = SCALER_X.transform(last_sequence_features)
        except ValueError as e: return JsonResponse({'error': f'피처 스케일링 오류: {str(e)}'}, status=500)
        input_sequence = np.expand_dims(scaled_features, axis=0)

        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 모델 예측 수행...")
        try: predicted_scaled_pct_changes = MODEL.predict(input_sequence, verbose=0)[0]
        except Exception as e: return JsonResponse({'error': f'모델 예측 오류: {str(e)}'}, status=500)

        predicted_pct_changes = SCALER_Y_PCT_CHANGE.inverse_transform(predicted_scaled_pct_changes.reshape(-1, 1)).flatten()
        future_prices = []
        current_price = last_actual_close
        # last_known_date_for_ajax는 위에서 이미 타입 처리됨
        future_dates_dt = get_future_trading_dates(last_known_date_for_ajax, PREDICT_DAYS, country='KR')
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
        
        for pct_change_val in predicted_pct_changes: # 변수명 변경 (pct_change -> pct_change_val)
            current_price = current_price * (1 + pct_change_val / 100)
            future_prices.append(round(current_price)) # 정수형으로 변환
        
        predictions_output = [{'date': date, 'price': price} for date, price in zip(future_dates_str, future_prices)]
        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 예측 완료.")
        return JsonResponse({'stock_code': stock_code, 'stock_name': stock_name_from_db, 'predictions': predictions_output})

    return JsonResponse({'error': '잘못된 요청입니다.'}, status=400)
