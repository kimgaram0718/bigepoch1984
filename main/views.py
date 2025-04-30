from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.
#250430_main 외 새 브랜치 생성 후 
# 그 브랜치에서 작업하기 테스트 중입니다
#17시37분에 django01 이란 브랜치생성 후 진입했어요
def main(request):
    return render(request, 'main.html')