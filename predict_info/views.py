from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def predict_info_view(request):
    return render(request, 'predict_info.html')
