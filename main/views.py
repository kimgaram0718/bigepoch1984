from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
#250430_main 외 새 브랜치 생성 후 
# 그 브랜치에서 작업하기 테스트 중입니다
def main(request):
    return render(request, 'main.html')