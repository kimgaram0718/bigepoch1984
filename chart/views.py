import logging
from django.shortcuts import render
from django.http import Http404

# 로거 설정
logger = logging.getLogger(__name__)

def chart_view(request):
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
    }
    return render(request, 'chart.html', context)

def chart_detail_view(request, stock_id):
    # stocks 데이터를 하드코딩으로 정의
    stocks = [
        {'id': 1, 'name': '삼성전자', 'change': '+4.52%', 'status': 'text-success'},
        {'id': 2, 'name': '카카오', 'change': '-3.21%', 'status': 'text-danger'},
        {'id': 3, 'name': 'SK하이닉스', 'change': '+2.80%', 'status': 'text-success'},
        {'id': 4, 'name': 'LG전자', 'change': '-1.90%', 'status': 'text-danger'},
        {'id': 5, 'name': '네이버', 'change': '+1.75%', 'status': 'text-success'},
    ]
    
    # stock_id에 해당하는 종목 찾기
    stock = next((s for s in stocks if s['id'] == stock_id), None)
    if stock is None:
        raise Http404(f"Stock with id {stock_id} not found")

    # Django 로그로 출력
    logger.info(f"Chart Detail Accessed - Stock ID: {stock_id}, Stock: {stock}")

    context = {
        'stock_id': stock_id,
        'stock': stock,
    }
    return render(request, 'chart_detail.html', context)