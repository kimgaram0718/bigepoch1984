from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
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
            {'name': '삼성전자', 'change': '+4.52%', 'status': 'text-success'},
            {'name': '카카오', 'change': '-3.21%', 'status': 'text-danger'},
            {'name': 'SK하이닉스', 'change': '+2.80%', 'status': 'text-success'},
            {'name': 'LG전자', 'change': '-1.90%', 'status': 'text-danger'},
            {'name': '네이버', 'change': '+1.75%', 'status': 'text-success'},
        ],
    }
    return render(request, 'chart.html', context)
