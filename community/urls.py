from django.urls import path
from . import views

app_name = 'community'

urlpatterns = [
    path('/community', views.community_view, name='community'),
    path('/write/', views.write_view, name='write'),  # 글쓰기 뷰 추가
]