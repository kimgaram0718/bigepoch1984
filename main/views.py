# main/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
import requests
import json
from django.views.decorators.http import require_http_methods
from community.models import AdminBoard  # AdminBoard 모델 임포트
from datetime import datetime, timezone, timedelta

#add1
def main_coalition(request):
    return render(request, 'main_coalition.html')
def main_faq(request):
    return render(request, 'main_faq.html')
def main_advertise(request):
    return render(request, 'main_advertise.html')

def admin_board_detail(request, pk):
    """
    AdminBoard의 상세 페이지 뷰.
    pk로 특정 게시글을 조회하고, is_visible=True인 경우에만 표시.
    """
    post = get_object_or_404(AdminBoard, pk=pk, is_visible=True)
    context = {
        'post': post,
    }
    return render(request, 'community_admin_content.html', context)  # 템플릿 변경
#add2

def main(request):
    # is_visible=True인 AdminBoard 데이터 가져오기
    admin_posts = AdminBoard.objects.filter(is_visible=True).select_related('user')[:3]
    context = {
        'admin_posts': admin_posts,
    }
    return render(request, 'main.html', context)

def get_naver_news(request):
    client_id = "OCRZok3QLNl9VF2e0Uo_"
    client_secret = "djBL9xrZIM"
    query = "경제"
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": query,
        "display": 30,  # 더 많이 받아오기
        "sort": "date"
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        news_data = response.json()
        news_list = []
        for item in news_data.get('items', []):
            title = item['title'].replace('<b>', '').replace('</b>', '')
            description = item['description'].replace('<b>', '').replace('</b>', '')
            pub_date_str = item['pubDate']
            pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
            news_list.append({
                'title': title,
                'link': item['link'],
                'description': description,
                'pubDate': pub_date_str,
                'pubDateObj': pub_date
            })
        # 오늘 날짜만 필터링
        if news_list:
            today = datetime.now(timezone.utc).astimezone(news_list[0]['pubDateObj'].tzinfo).date()
        else:
            today = datetime.now().date()
        today_news = [n for n in news_list if n['pubDateObj'].date() == today]
        # 오늘 뉴스가 5개 미만이면, 최신순으로 5개까지 채움
        today_news.sort(key=lambda x: x['pubDateObj'], reverse=True)
        if len(today_news) < 5:
            news_list.sort(key=lambda x: x['pubDateObj'], reverse=True)
            extra = [n for n in news_list if n not in today_news]
            today_news += extra[:5-len(today_news)]
        # pubDateObj 제거
        for news in today_news:
            del news['pubDateObj']
        return JsonResponse({'news': today_news[:5]})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)