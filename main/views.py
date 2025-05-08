from django.shortcuts import render, redirect
from django.http import JsonResponse
import requests
import json
from django.views.decorators.http import require_http_methods

# Create your views here.
#250430_main 외 새 브랜치 생성 후 
# 그 브랜치에서 작업하기 테스트 중입니다
#17시37분에 django01 이란 브랜치생성 후 진입했어요
def main(request):
    return render(request, 'main.html')

def get_naver_news(request):
    client_id = "OCRZok3QLNl9VF2e0Uo_"
    client_secret = "djBL9xrZIM"
    
    # 증시 관련 검색어
    query = "증시 OR 주식 OR 코스피 OR 코스닥"
    
    # 네이버 뉴스 검색 API 호출
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": query,
        "display": 10,  # 10개의 뉴스만 가져옴
        "sort": "date"  # 최신순 정렬
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # 에러 발생시 예외 발생
        news_data = response.json()
        
        # 뉴스 데이터 가공
        news_list = []
        for item in news_data.get('items', []):
            news_list.append({
                'title': item['title'].replace('<b>', '').replace('</b>', ''),
                'link': item['link'],
                'description': item['description'].replace('<b>', '').replace('</b>', ''),
                'pubDate': item['pubDate']
            })
        
        return JsonResponse({'news': news_list})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)