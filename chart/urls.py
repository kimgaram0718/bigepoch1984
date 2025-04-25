from django.urls import path
from . import views

app_name = 'chart'

urlpatterns = [
    path('', views.chart_view, name='chart'),
    path('detail/<int:stock_id>/', views.chart_detail_view, name='chart_detail'),
]