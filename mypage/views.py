from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
def mypage_view(request):
    return render(request, 'mypage.html')