# predict_info/views.py
from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import os
import json
from .utils import calculate_manual_features # predict_info/utils.py 가정
import re
import traceback
import holidays # 공휴일 처리
import yfinance as yf

# --- 모델 및 스케일러 경로 설정 ---
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # predict_info 앱 폴더
MARKET = 'KOSDAQ' # 학습된 모델에 맞게 (KOSDAQ 또는 KOSPI)
MODEL_SAVE_DIR = os.path.join(APP_DIR, 'ml_models')

SCALER_X_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_scaler_x.joblib')
SCALER_Y_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_scaler_y.joblib') # 사용자 요청 파일명
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'{MARKET.lower()}_lstm_best_model.keras')

# --- 모델 학습 시 사용된 상수 ---
SEQUENCE_LENGTH = 60
PREDICT_DAYS = 7
# <<< 데이터 로딩 기간 증가: 지표 계산에 필요한 최소 과거 일수 늘리기 >>>
MIN_DATA_DAYS_FOR_FEATURE_CALC = 150 # 기존 90 -> 150으로 증가 (MACD(26) 등 고려)
DEFAULT_FEATURE_NAMES = [
    'Open', 'High', 'Low', 'Volume', 'Value', 'Change', 'ATR',
    'BB_Lower', 'BB_Mid', 'BB_Upper', 'RSI', 'MACD', 'MACD_Hist', 'MACD_Signal'
]

# --- 모델 및 스케일러 로드 (서버 시작 시 한 번만) ---
MODEL = None
SCALER_X = None
SCALER_Y_PCT_CHANGE = None
FEATURE_NAMES_FROM_SCALER = None
KRX_STOCKS_LIST = None
STOCK_CODE_COL_NAME = None

print(f"--- [DEBUG] predict_info/views.py 모듈 로딩 시작 ---")
# ... (기존 디버깅 로그 유지) ...
print(f"[INFO] 모델 및 스케일러 로딩 시도...")
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
    print("[INFO] KRX 전체 종목 목록 로딩 중...")
    KRX_STOCKS_LIST = fdr.StockListing('KRX')
    if KRX_STOCKS_LIST is not None and not KRX_STOCKS_LIST.empty:
        if 'Name' in KRX_STOCKS_LIST.columns: KRX_STOCKS_LIST['Name'] = KRX_STOCKS_LIST['Name'].str.strip()
        if 'Symbol' in KRX_STOCKS_LIST.columns: STOCK_CODE_COL_NAME = 'Symbol'
        elif 'Code' in KRX_STOCKS_LIST.columns: STOCK_CODE_COL_NAME = 'Code'
        else: print("[ERROR] KRX 종목 코드 컬럼 없음.")
        if STOCK_CODE_COL_NAME: print(f"[INFO] KRX 종목 코드 컬럼: '{STOCK_CODE_COL_NAME}'")
        print(f"[INFO] KRX 목록 로드 완료 ({len(KRX_STOCKS_LIST)}개).")
    else: print("[ERROR] KRX_STOCKS_LIST 로딩 실패/비어있음.")
except Exception as e: print(f"[CRITICAL ERROR] 초기 로딩 중 오류: {e}"); traceback.print_exc()

def get_stock_code_from_query(query_str):
    if KRX_STOCKS_LIST is None or KRX_STOCKS_LIST.empty: return None, None
    if STOCK_CODE_COL_NAME is None: return None, None
    query_str_cleaned = query_str.strip()
    print(f"[DEBUG get_stock_code_from_query] Received query: '{query_str}', Cleaned query: '{query_str_cleaned}'")
    if re.match(r'^\d{6}$', query_str_cleaned):
        stock_info = KRX_STOCKS_LIST[KRX_STOCKS_LIST[STOCK_CODE_COL_NAME] == query_str_cleaned]
        if not stock_info.empty: return query_str_cleaned, stock_info['Name'].iloc[0]
    stock_info = KRX_STOCKS_LIST[KRX_STOCKS_LIST['Name'] == query_str_cleaned]
    if not stock_info.empty: return stock_info[STOCK_CODE_COL_NAME].iloc[0], query_str_cleaned
    print(f"[DEBUG get_stock_code_from_query] Name '{query_str_cleaned}' not found.")
    return None, None

def get_future_trading_dates(start_date, num_days, country='KR'):
    if not isinstance(start_date, datetime): start_date = pd.to_datetime(start_date).date()
    kr_holidays = holidays.KR(years=[start_date.year, start_date.year + 1])
    future_dates = []; current_date = pd.Timestamp(start_date)
    while len(future_dates) < num_days:
        current_date += BDay(1)
        if current_date.date() in kr_holidays: continue
        future_dates.append(current_date.date())
    return future_dates

def get_market_realtime_info(symbol, name):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        current = info.get('regularMarketPrice')
        prev = info.get('regularMarketPreviousClose')
        percent = None
        if current is not None and prev:
            try:
                percent = round((current - prev) / prev * 100, 2)
            except Exception:
                percent = None
        return {
            'name': name,
            'current_price': current,
            'previous_close': prev,
            'percent': percent,
        }
    except Exception as e:
        print(f"[ERROR] get_market_realtime_info: {name} {symbol} {e}")
        return {
            'name': name,
            'current_price': None,
            'previous_close': None,
            'percent': None,
        }

def predict_info_view(request):
    query = request.GET.get('query', '휴젤')
    print(f"[DEBUG predict_info_view] Received query parameter: '{query}'")
    kospi_info = get_market_realtime_info('^KS11', '코스피')
    kosdaq_info = get_market_realtime_info('^KQ11', '코스닥')
    context = {
        'stock_name_for_display': query, 'ticker': None, 'error_message': None,
        'initial_predictions': None, 'prediction_error': None,
        'prediction_indices': [
            {'name': f"코스피 예측 지수", 'current_price': kospi_info['current_price'], 'previous_close': kospi_info['previous_close']},
            {'name': f"코스닥 예측 지수", 'current_price': kosdaq_info['current_price'], 'previous_close': kosdaq_info['previous_close']},
        ],
        'prediction_tickers': [{'name': '삼성전자', 'price': '80,000원', 'change': '+1.5%'}, {'name': 'SK하이닉스', 'price': '180,000원', 'change': '-0.8%'}],
        'recommended_stocks': [{'rank': 1, 'name': '추천종목A'}, {'rank': 2, 'name': '추천종목B'}, {'rank': 3, 'name': '추천종목C'}],
        'top_contents': [{'rank': 1, 'title': '오늘의 시장 분석과 내일의 전망', 'link': '#'}, {'rank': 2, 'title': 'AI가 선택한 유망 기술주 TOP 5', 'link': '#'}, {'rank': 3, 'title': '하반기 경제 시나리오별 투자 전략', 'link': '#'}],
    }
    stock_code, stock_name = get_stock_code_from_query(query)
    print(f"[DEBUG predict_info_view] Resolved stock_code: {stock_code}, stock_name: {stock_name}")
    if not stock_code:
        context['error_message'] = f"종목명 또는 코드 '{query}'에 해당하는 정보를 찾을 수 없습니다."
    else:
        context['stock_name_for_display'] = stock_name; context['ticker'] = stock_code
        if MODEL and SCALER_X and SCALER_Y_PCT_CHANGE and FEATURE_NAMES_FROM_SCALER:
            print(f"[INFO] predict_info_view: 종목 {stock_code}({stock_name}) 초기 예측 시작...")
            try:
                # <<< 데이터 로딩 기간 계산 시 MIN_DATA_DAYS_FOR_FEATURE_CALC 사용 >>>
                days_to_fetch = SEQUENCE_LENGTH + MIN_DATA_DAYS_FOR_FEATURE_CALC
                end_date = datetime.now(); start_date = end_date - timedelta(days=days_to_fetch * 2.5) # 주말/공휴일 고려
                print(f"[DEBUG predict_info_view] Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                df_pred_raw = fdr.DataReader(stock_code, start_date, end_date)

                if df_pred_raw.empty or len(df_pred_raw) < MIN_DATA_DAYS_FOR_FEATURE_CALC: # 최소 지표계산일수 확인
                    context['prediction_error'] = f"초기 예측 위한 데이터 부족 (최소 {MIN_DATA_DAYS_FOR_FEATURE_CALC}일 필요, 현재 {len(df_pred_raw)}일)."
                else:
                    df_pred_features_calc = calculate_manual_features(df_pred_raw.copy())
                    valid_feature_names = [f for f in FEATURE_NAMES_FROM_SCALER if f in df_pred_features_calc.columns]
                    if len(valid_feature_names) != len(FEATURE_NAMES_FROM_SCALER):
                        missing_f = [f for f in FEATURE_NAMES_FROM_SCALER if f not in valid_feature_names]
                        context['prediction_error'] = f"피처 계산 후 일부 필수 피처 누락: {missing_f}"
                    else:
                        df_for_prediction_input = df_pred_features_calc[valid_feature_names].copy()
                        df_for_prediction_input = df_for_prediction_input.ffill().bfill() # NaN 채우기
                        if len(df_for_prediction_input) < SEQUENCE_LENGTH:
                            context['prediction_error'] = f"피처 생성 및 NaN 처리 후 데이터 부족 (필요: {SEQUENCE_LENGTH}일)."
                        else:
                            last_sequence_features = df_for_prediction_input.iloc[-SEQUENCE_LENGTH:]
                            if last_sequence_features.isnull().values.any():
                                nan_cols = last_sequence_features.columns[last_sequence_features.isnull().any()].tolist()
                                context['prediction_error'] = f"예측 사용할 마지막 시퀀스에 NaN 포함 (컬럼: {nan_cols})."
                                print(f"[WARNING] predict_info_view: {context['prediction_error']} for {stock_code}")
                            else:
                                last_actual_close_for_pred = df_pred_raw['Close'].iloc[-1]
                                last_known_date_for_pred = last_sequence_features.index[-1]
                                scaled_features_for_pred = SCALER_X.transform(last_sequence_features)
                                input_sequence_for_pred = np.expand_dims(scaled_features_for_pred, axis=0)
                                predicted_scaled_pct_changes = MODEL.predict(input_sequence_for_pred, verbose=0)[0]
                                predicted_pct_changes = SCALER_Y_PCT_CHANGE.inverse_transform(predicted_scaled_pct_changes.reshape(-1, 1)).flatten()
                                future_prices = []
                                current_price = last_actual_close_for_pred
                                future_dates_dt = get_future_trading_dates(last_known_date_for_pred, PREDICT_DAYS, country='KR')
                                future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
                                for pct_change in predicted_pct_changes:
                                    current_price = current_price * (1 + pct_change / 100)
                                    future_prices.append(round(current_price, 2))
                                context['initial_predictions'] = [{'date': date, 'price': price} for date, price in zip(future_dates_str, future_prices)]
                                print(f"[INFO] predict_info_view: 종목 {stock_code}({stock_name}) 초기 예측 완료.")
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
        stock_code, stock_name_from_fdr = get_stock_code_from_query(stock_code_input)
        if not stock_code: return JsonResponse({'error': f"'{stock_code_input}' 종목 정보 없음."}, status=400)
        if MODEL is None or SCALER_X is None or SCALER_Y_PCT_CHANGE is None or FEATURE_NAMES_FROM_SCALER is None:
            return JsonResponse({'error': '모델/스케일러 미준비. 서버 로그 확인.'}, status=500)

        # <<< 데이터 로딩 기간 계산 시 MIN_DATA_DAYS_FOR_FEATURE_CALC 사용 >>>
        days_to_fetch = SEQUENCE_LENGTH + MIN_DATA_DAYS_FOR_FEATURE_CALC
        end_date = datetime.now(); start_date = end_date - timedelta(days=days_to_fetch * 2.5) # 주말/공휴일 고려
        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code}({stock_name_from_fdr}) 데이터 로딩 중...")
        try:
            df_raw = fdr.DataReader(stock_code, start_date, end_date)
            if df_raw.empty or len(df_raw) < MIN_DATA_DAYS_FOR_FEATURE_CALC: # 최소 지표계산일수 확인
                return JsonResponse({'error': f'{stock_name_from_fdr}({stock_code}) 예측 데이터 부족 (최소 {MIN_DATA_DAYS_FOR_FEATURE_CALC}일 필요).'}, status=400)
        except Exception as e: return JsonResponse({'error': f'{stock_name_from_fdr}({stock_code}) 데이터 로딩 실패: {str(e)}'}, status=400)

        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 피처 계산 중...")
        df_features_calculated = calculate_manual_features(df_raw.copy())
        try:
            valid_feature_names_for_input = [f for f in FEATURE_NAMES_FROM_SCALER if f in df_features_calculated.columns]
            if len(valid_feature_names_for_input) != len(FEATURE_NAMES_FROM_SCALER):
                missing_cols = [f for f in FEATURE_NAMES_FROM_SCALER if f not in valid_feature_names_for_input]
                return JsonResponse({'error': f'피처 계산 후 일부 피처 누락. 누락: {missing_cols}'}, status=500)
            df_for_prediction_input = df_features_calculated[valid_feature_names_for_input].copy()
            df_for_prediction_input = df_for_prediction_input.ffill().bfill() # NaN 처리
        except KeyError as e: return JsonResponse({'error': f'피처 선택 중 오류: {str(e)}'}, status=500)

        if len(df_for_prediction_input) < SEQUENCE_LENGTH:
            return JsonResponse({'error': f'{stock_name_from_fdr}({stock_code}) 피처 생성 및 NaN 처리 후 예측 데이터 부족.'}, status=400)

        last_sequence_features = df_for_prediction_input.iloc[-SEQUENCE_LENGTH:]
        # <<< NaN 최종 확인 강화 >>>
        if last_sequence_features.isnull().values.any():
            nan_cols = last_sequence_features.columns[last_sequence_features.isnull().any()].tolist()
            print(f"[WARNING] predict_stock_price_ajax: 마지막 시퀀스에 NaN 포함 for {stock_code} (컬럼: {nan_cols})")
            return JsonResponse({'error': f'{stock_name_from_fdr}({stock_code}) 예측용 데이터에 NaN 포함.'}, status=400)

        last_actual_close = df_raw['Close'].iloc[-1]
        last_known_date = last_sequence_features.index[-1]
        try: scaled_features = SCALER_X.transform(last_sequence_features)
        except ValueError as e: return JsonResponse({'error': f'피처 스케일링 오류: {str(e)}'}, status=500)
        input_sequence = np.expand_dims(scaled_features, axis=0)

        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 모델 예측 수행...")
        try: predicted_scaled_pct_changes = MODEL.predict(input_sequence, verbose=0)[0]
        except Exception as e: return JsonResponse({'error': f'모델 예측 오류: {str(e)}'}, status=500)

        predicted_pct_changes = SCALER_Y_PCT_CHANGE.inverse_transform(predicted_scaled_pct_changes.reshape(-1, 1)).flatten()
        future_prices = []
        current_price = last_actual_close
        future_dates_dt = get_future_trading_dates(last_known_date, PREDICT_DAYS, country='KR')
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates_dt]
        for pct_change in predicted_pct_changes:
            current_price = current_price * (1 + pct_change / 100)
            future_prices.append(round(current_price, 2))
        predictions_output = [{'date': date, 'price': price} for date, price in zip(future_dates_str, future_prices)]
        print(f"[INFO] predict_stock_price_ajax: 종목 {stock_code} 예측 완료.")
        return JsonResponse({'stock_code': stock_code, 'stock_name': stock_name_from_fdr, 'predictions': predictions_output})

    return JsonResponse({'error': '잘못된 요청입니다.'}, status=400)

