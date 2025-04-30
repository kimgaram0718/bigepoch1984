from django.urls import path
from . import views

app_name = 'community'

urlpatterns = [
    path('', views.community_view, name='community'),
    path('write/', views.write_view, name='write'),
    path('<int:post_id>/', views.community_detail_view, name='detail'),
    path('edit/<int:post_id>/', views.edit_view, name='edit'),
    path('delete/<int:post_id>/', views.delete_view, name='delete'),
]