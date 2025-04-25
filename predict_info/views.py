from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def predict_info_view(request):
    context = {
        'prediction_indices': [
            {'name': '예측 지수 A'},
            {'name': '예측 지수 B'},
        ],
        'prediction_tickers': [
            {'name': '삼성전자', 'price': '75,000원', 'change': '+2.5%'},
            {'name': '네이버', 'price': '260,000원', 'change': '+1.3%'},
        ],
        'recommended_stocks': [
            {'rank': 1, 'name': '삼성전자'},
            {'rank': 2, 'name': '카카오'},
            {'rank': 3, 'name': '현대차'},
        ],
        'top_contents': [
            {'rank': 1, 'title': '다음 주 주목할 업종은?', 'link': '#'},
            {'rank': 2, 'title': 'AI로 본 시장 예측', 'link': '#'},
            {'rank': 3, 'title': '경제지표와 코스피 방향', 'link': '#'},
        ],
    }
    return render(request, 'predict_info.html', context)
