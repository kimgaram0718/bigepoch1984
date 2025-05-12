from django.shortcuts import render
from predict.models import StockPrice
from django.db.models import Avg, Q
import datetime
import yfinance as yf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.

def get_index_info(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="2d")  # 최근 2일치
    if len(hist) < 2:
        return None
    today = hist.iloc[-1]
    prev = hist.iloc[-2]
    close = today['Close']
    prev_close = prev['Close']
    change = close - prev_close
    change_rate = (change / prev_close) * 100 if prev_close else 0
    return {
        'date': today.name.date(),
        'close': close,
        'change': change,
        'change_rate': change_rate,
    }

def home(request):
    # 가장 최근 날짜
    try:
        target_date = StockPrice.objects.latest('date').date
    except StockPrice.DoesNotExist:
        target_date = None

    # 전일(이전 거래일) 구하기
    prev_date = None
    if target_date:
        prev_dates = StockPrice.objects.filter(date__lt=target_date).order_by('-date').values_list('date', flat=True)
        prev_date = prev_dates[0] if prev_dates else None

    # 코스피/코스닥 급등 TOP5
    kospi_top5 = StockPrice.objects.filter(
        market='KOSPI', date=target_date
    ).order_by('-change_rate')[:5] if target_date else []
    kosdaq_top5 = StockPrice.objects.filter(
        market='KOSDAQ', date=target_date
    ).order_by('-change_rate')[:5] if target_date else []

    # 코스피/코스닥 실제 지수 정보 (yfinance)
    kospi_info = get_index_info('^KS11')
    kosdaq_info = get_index_info('^KQ11')

    # 선택 종목 정보 보기 기능
    stock_query = request.POST.get('stock_query') if request.method == 'POST' else None
    stock_7days = []
    stock_30days = []
    stock_30days_json = '[]'
    stock_30ma = []
    stock_30ma_json = '[]'
    stock_name = None
    if stock_query:
        stock_qs = StockPrice.objects.filter(
            Q(name__iexact=stock_query) | Q(code__iexact=stock_query)
        ).order_by('-date')
        if stock_qs.exists():
            stock_name = stock_qs.first().name
            stock_7days = list(stock_qs[:7][::-1])  # 최근 7일 (오름차순)
            last_30 = list(stock_qs[:30][::-1])     # 최근 30일 (오름차순)
            stock_30days = [
                {'date': s.date.strftime('%Y-%m-%d'), 'close': s.close, 'volume': s.volume} for s in last_30
            ]
            closes = [s['close'] for s in stock_30days]
            for i in range(len(closes)):
                if i < 29:
                    stock_30ma.append(None)
                else:
                    ma = sum(closes[i-29:i+1]) / 30
                    stock_30ma.append(round(ma, 2))
            stock_30days_json = json.dumps(stock_30days)
            stock_30ma_json = json.dumps(stock_30ma)
    return render(request, 'model.html', {
        'kospi_top5': kospi_top5,
        'kosdaq_top5': kosdaq_top5,
        'kospi_info': kospi_info,
        'kosdaq_info': kosdaq_info,
        'stock_7days': stock_7days,
        'stock_30days': stock_30days,
        'stock_30days_json': stock_30days_json,
        'stock_30ma': stock_30ma,
        'stock_30ma_json': stock_30ma_json,
        'stock_query': stock_query,
        'stock_name': stock_name,
    })

def lstm_predict(stock_code):
    qs = StockPrice.objects.filter(Q(code=stock_code) | Q(name=stock_code)).order_by('-date')[:30][::-1]
    if len(qs) < 10:
        return None, None, None, None, None  # 데이터 부족
    data = np.array([[s.close, s.high, s.low, s.open, s.volume] for s in qs])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(data_scaled) - 3):
        X.append(data_scaled[i:i+3])
        y.append(data_scaled[i+3][0])  # 종가 예측
    X, y = np.array(X), np.array(y)
    X_train, y_train = X[:-3], y[:-3]
    X_test, y_test = X[-3:], y[-3:]
    model = keras.Sequential([
        keras.layers.LSTM(32, input_shape=(3, 5)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=2, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_real = scaler.inverse_transform(
        np.hstack([y_pred, np.zeros((3, 4))])
    )[:, 0]
    y_test_real = scaler.inverse_transform(
        np.hstack([y_test.reshape(-1, 1), np.zeros((3, 4))])
    )[:, 0]
    mae = np.mean(np.abs(y_pred_real - y_test_real))
    # R2(예측률)
    ss_res = np.sum((y_test_real - y_pred_real) ** 2)
    ss_tot = np.sum((y_test_real - np.mean(y_test_real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    # 정확도(여기서는 1 - MAE/실제값평균)
    acc = 1 - (mae / np.mean(y_test_real)) if np.mean(y_test_real) != 0 else 0
    last_seq = data_scaled[-3:]
    future_preds = []
    for _ in range(3):
        pred = model.predict(last_seq.reshape(1, 3, 5))
        pred_real = scaler.inverse_transform(
            np.hstack([pred, np.zeros((1, 4))])
        )[0, 0]
        future_preds.append(pred_real)
        next_row = np.array([pred[0, 0], 0, 0, 0, 0])
        last_seq = np.vstack([last_seq[1:], next_row])
    return list(future_preds), float(mae), list(y_pred_real), float(r2), float(acc)

@csrf_exempt
def ajax_lstm_predict(request):
    code = request.POST.get('stock_code')
    preds, mae, test_preds, r2, acc = lstm_predict(code)
    if preds is None or mae is None or test_preds is None:
        return JsonResponse({
            'future_preds': [],
            'mae': None,
            'test_preds': [],
            'r2': None,
            'accuracy': None,
            'error': '데이터가 부족합니다. 최소 10일 이상의 데이터가 필요합니다.'
        })
    return JsonResponse({
        'future_preds': preds,
        'mae': mae,
        'test_preds': test_preds,
        'r2': r2,
        'accuracy': acc,
    })
