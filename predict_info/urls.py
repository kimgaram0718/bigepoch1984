from django.urls import path
from . import views

app_name = 'predict_info'

urlpatterns = [
    path('', views.predict_info_view, name='predict_info'),
]