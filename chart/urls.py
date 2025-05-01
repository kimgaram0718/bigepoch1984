from django.urls import path
from .views import chart_view

app_name = 'chart'

urlpatterns = [
    path('', chart_view, name='chart'),
]
