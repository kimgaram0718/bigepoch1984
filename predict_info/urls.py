from django.urls import path
from . import views

app_name = 'predict_info'

urlpatterns = [
    path('/predict_info', views.predict_info_view, name='predict_info'),
]