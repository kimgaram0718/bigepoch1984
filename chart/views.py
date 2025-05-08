import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta
import yfinance as yf
from django.http import JsonResponse

SYMBOL_MAP = {
    '삼성전자': '005930.KS',
    '카카오': '035720.KS',
    '네이버': '035420.KS',
    # 필요시 추가
}

def get_stock_price(symbol):
    try:
        # 한글명 입력 시 종목코드로 변환
        symbol = SYMBOL_MAP.get(symbol, symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'current_price': info.get('regularMarketPrice', 0),
            'previous_close': info.get('regularMarketPreviousClose', 0),
            'change': info.get('regularMarketChange', 0),
            'change_percent': info.get('regularMarketChangePercent', 0),
            'name': info.get('longName', symbol)
        }
    except Exception as e:
        print(f"Error fetching stock data for symbol: {symbol}, Exception: {e}")
        return None

def chart_view(request):
    query = request.GET.get('query', '삼성전자')
    
    # 실시간 주가 정보 가져오기
    stock_info = get_stock_price(query)
    
    # CSV 데이터 로드
    file_map = {
        '삼성전자': '삼성전자_5min_50month.csv',
        '카카오': '카카오_5min_50month.csv',
        '네이버': '네이버_5min_50month.csv',
    }
    file_name = file_map.get(query, file_map['삼성전자'])
    df = pd.read_csv(f'chart/data/{file_name}', parse_dates=['datetime'])
    df = df.sort_values('datetime')

    # 실시간 가격을 마지막에 추가
    if stock_info and stock_info['current_price']:
        now = pd.Timestamp.now().floor('min')
        new_row = {
            'datetime': now,
            'open': stock_info['current_price'],
            'high': stock_info['current_price'],
            'low': stock_info['current_price'],
            'close': stock_info['current_price']
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    future_date = df['datetime'].max() + timedelta(days=14)
    candle_dates = df['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    open_prices = df['open'].tolist()
    high_prices = df['high'].tolist()
    low_prices = df['low'].tolist()
    close_prices = df['close'].tolist()
    ma5 = df['close'].rolling(window=5).mean().fillna(0).tolist()
    ma20 = df['close'].rolling(window=20).mean().fillna(0).tolist()

    # 최근 1년(52주)간의 최고가/최저가 계산
    one_year_ago = pd.Timestamp.now() - pd.Timedelta(weeks=52)
    recent_df = df[df['datetime'] >= one_year_ago]
    fifty_two_week_high = recent_df['high'].max() if not recent_df.empty else None
    fifty_two_week_low = recent_df['low'].min() if not recent_df.empty else None

    context = {
        'stock_name': query,
        'ticker': '',
        'candle_dates': json.dumps(candle_dates),
        'open_prices': json.dumps(open_prices),
        'high_prices': json.dumps(high_prices),
        'low_prices': json.dumps(low_prices),
        'close_prices': json.dumps(close_prices),
        'ma5': json.dumps(ma5),
        'ma20': json.dumps(ma20),
        'future_date': future_date.strftime('%Y-%m-%d %H:%M'),
        'stock_info': stock_info,
        'fifty_two_week_high': fifty_two_week_high,
        'fifty_two_week_low': fifty_two_week_low,
    }
    return render(request, 'chart.html', context)

def get_realtime_price(request):
    symbol = request.GET.get('symbol', '')
    if not symbol:
        return JsonResponse({'error': 'Symbol is required'}, status=400)
    
    stock_info = get_stock_price(symbol)
    if stock_info:
        return JsonResponse(stock_info)
    return JsonResponse({'error': 'Failed to fetch stock data'}, status=500)
