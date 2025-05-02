from django.urls import path
from . import views

app_name = 'community'

urlpatterns = [
    path('', views.community_view, name='community'),
    path('news/', views.news_view, name='news'),
    path('write/', views.write_view, name='write'),
    path('<int:post_id>/', views.community_detail_view, name='detail'),
    path('<int:post_id>/comment/', views.comment_create, name='comment_create'),
    path('<int:post_id>/like/', views.like_post, name='like_post'),
    path('<int:post_id>/edit/', views.edit_view, name='edit'),
    path('<int:post_id>/delete/', views.delete_view, name='delete'),
]