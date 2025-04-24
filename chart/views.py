from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def chart_view(request):
    return render(request, 'chart.html')
