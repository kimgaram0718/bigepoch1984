# predict_info/urls.py
from django.urls import path
from . import views

app_name = 'predict_info' # 앱 네임스페이스 설정

urlpatterns = [
    # predict_info.html을 렌더링하고 초기 예측을 수행하는 뷰
    path('', views.predict_info_view, name='predict_info_page'),
    # AJAX를 통해 특정 종목의 주가를 예측하는 API 엔드포인트
    path('predict_stock_ajax/', views.predict_stock_price_ajax, name='predict_stock_price_ajax'),
    # 자동완성 검색을 위한 API 엔드포인트 추가
    path('search_stocks_ajax/', views.search_stocks_ajax, name='search_stocks_ajax'),
]
