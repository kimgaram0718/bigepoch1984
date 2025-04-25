import logging
from django.shortcuts import render
from django.http import Http404
import json

# 로거 설정
logger = logging.getLogger(__name__)

def chart_view(request):
    # 기본 캔들스틱 데이터
    stock_name = "기본 종목"
    ticker = "DEFAULT"
    candle_dates = ["2025-04-16", "2025-04-17", "2025-04-18", "2025-04-21", "2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27"]
    open_prices = [10000, 10100, 10200, 10150, 10300, 10250, 10400, 10350, 10500, 10450]
    high_prices = [10100, 10200, 10300, 10250, 10400, 10350, 10500, 10450, 10600, 10550]
    low_prices = [9900, 10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400, 10350]
    close_prices = [10050, 10150, 10180, 10200, 10280, 10300, 10420, 10400, 10480, 10500]
    
    # 예측 데이터
    pred_dates = ["2025-04-28", "2025-04-29", "2025-04-30"]
    pred_prices = [10550, 10600, 10680]

    # 이동평균선 계산 (5일선, 20일선)
    ma5 = []
    ma20 = []
    for i in range(len(close_prices)):
        if i >= 4:  # 5일선은 최소 5일 데이터 필요
            ma5.append(sum(close_prices[i-4:i+1]) / 5)
        else:
            ma5.append(None)
        if i >= 19:  # 20일선은 최소 20일 데이터 필요
            ma20.append(sum(close_prices[i-19:i+1]) / 20)
        else:
            ma20.append(None)

    context = {
        'markets': [
            {'name': '국가 증시 A', 'value': '2,541.40', 'change': '+34.76 (1.40%)', 'status': 'text-success'},
            {'name': '국가 증시 B', 'value': '3,412.56', 'change': '-21.31 (-0.62%)', 'status': 'text-danger'},
        ],
        'news': [
            '1. 미국 고용지표 발표 예정',
            '2. 연준 기준금리 발표 대기',
            '3. 비트코인 상승률 2% 돌파',
        ],
        'stocks': [
            {'id': 1, 'name': '삼성전자', 'change': '+4.52%', 'status': 'text-success'},
            {'id': 2, 'name': '카카오', 'change': '-3.21%', 'status': 'text-danger'},
            {'id': 3, 'name': 'SK하이닉스', 'change': '+2.80%', 'status': 'text-success'},
            {'id': 4, 'name': 'LG전자', 'change': '-1.90%', 'status': 'text-danger'},
            {'id': 5, 'name': '네이버', 'change': '+1.75%', 'status': 'text-success'},
        ],
        'stock_name': stock_name,
        'ticker': ticker,
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'pred_dates': json.dumps(pred_dates),
        'pred_prices': json.dumps(pred_prices),
        'ma5': json.dumps(ma5),
        'ma20': json.dumps(ma20),
    }
    return render(request, 'chart.html', context)

def chart_detail_view(request, stock_id):
    # stocks 데이터를 하드코딩으로 정의
    stocks = [
        {'id': 1, 'name': '삼성전자', 'ticker': '005930', 'change': '+4.52%', 'status': 'text-success'},
        {'id': 2, 'name': '카카오', 'ticker': '035720', 'change': '-3.21%', 'status': 'text-danger'},
        {'id': 3, 'name': 'SK하이닉스', 'ticker': '000660', 'change': '+2.80%', 'status': 'text-success'},
        {'id': 4, 'name': 'LG전자', 'ticker': '066570', 'change': '-1.90%', 'status': 'text-danger'},
        {'id': 5, 'name': '네이버', 'ticker': '035420', 'change': '+1.75%', 'status': 'text-success'},
    ]
    
    # stock_id에 해당하는 종목 찾기
    stock = next((s for s in stocks if s['id'] == stock_id), None)
    if stock is None:
        raise Http404(f"Stock with id {stock_id} not found")

    # 캔들스틱 데이터
    candle_dates = ["2025-04-16", "2025-04-17", "2025-04-18", "2025-04-21", "2025-04-22", "2025-04-23", "2025-04-24", "2025-04-25", "2025-04-26", "2025-04-27"]
    open_prices = [73900, 74100, 74000, 74200, 73800, 73700, 73900, 74100, 74000, 74200]
    high_prices = [74200, 74400, 74300, 74500, 74100, 74000, 74200, 74400, 74300, 74500]
    low_prices = [73700, 73900, 73800, 74000, 73600, 73500, 73700, 73900, 73800, 74000]
    close_prices = [74100, 74000, 74200, 73900, 73700, 73900, 74100, 74000, 74200, 74100]
    
    # 예측 데이터
    pred_dates = ["2025-04-28", "2025-04-29", "2025-04-30"]
    pred_prices = [74300, 74500, 74600]

    # 이동평균선 계산 (5일선, 20일선)
    ma5 = []
    ma20 = []
    for i in range(len(close_prices)):
        if i >= 4:
            ma5.append(sum(close_prices[i-4:i+1]) / 5)
        else:
            ma5.append(None)
        if i >= 19:
            ma20.append(sum(close_prices[i-19:i+1]) / 20)
        else:
            ma20.append(None)

    # Django 로그로 출력
    logger.info(f"Chart Detail Accessed - Stock ID: {stock_id}, Stock: {stock}")

    context = {
        'markets': [
            {'name': '국가 증시 A', 'value': '2,541.40', 'change': '+34.76 (1.40%)', 'status': 'text-success'},
            {'name': '국가 증시 B', 'value': '3,412.56', 'change': '-21.31 (-0.62%)', 'status': 'text-danger'},
        ],
        'news': [
            '1. 미국 고용지표 발표 예정',
            '2. 연준 기준금리 발표 대기',
            '3. 비트코인 상승률 2% 돌파',
        ],
        'stocks': stocks,
        'stock_name': stock['name'],
        'ticker': stock['ticker'],
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'pred_dates': json.dumps(pred_dates),
        'pred_prices': json.dumps(pred_prices),
        'ma5': json.dumps(ma5),
        'ma20': json.dumps(ma20),
    }
    return render(request, 'chart_detail.html', context)