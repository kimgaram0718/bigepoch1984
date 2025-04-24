from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def mypage_view(request):
    context = {
        'user': {
            'nickname': '닉네임',
            'greeting_message': '인사 메시지를 입력하세요',
        },
        'prediction_items': [
            {'name': '삼성전자', 'price': '82,000원', 'change': '+1.20%'},
            {'name': '비트코인', 'price': '125,000,000원', 'change': '+0.80%'},
        ],
        'watchlist': [
            {'name': '이더리움'},
            {'name': '애플'},
        ],
        'user_posts': [
            {'title': 'LOOM, ARK, HIFI 상승 예상 분석'},
            {'title': '테더 스테이블 코인 동향'},
        ],
    }
    return render(request, 'mypage.html', context)

def edit_profile_view(request):
    return render(request, 'mypage.html')