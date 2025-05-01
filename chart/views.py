import pandas as pd
from django.shortcuts import render
import json
from datetime import timedelta

def chart_view(request):
    query = request.GET.get('query', '삼성전자')
    file_map = {
        '삼성전자': '삼성전자_5min_50month.csv',
        '카카오': '카카오_5min_50month.csv',
        '네이버': '네이버_5min_50month.csv',
    }
    file_name = file_map.get(query, file_map['삼성전자'])
    df = pd.read_csv(f'chart/data/{file_name}', parse_dates=['datetime'])
    df = df.sort_values('datetime')

    future_date = df['datetime'].max() + timedelta(days=14)
    candle_dates = df['datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    open_prices = df['open'].tolist()
    high_prices = df['high'].tolist()
    low_prices = df['low'].tolist()
    close_prices = df['close'].tolist()
    ma5 = df['close'].rolling(window=5).mean().tolist()
    ma20 = df['close'].rolling(window=20).mean().tolist()

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
    }
    return render(request, 'chart.html', context)
