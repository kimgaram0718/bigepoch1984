from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from .models import User
from django.http import JsonResponse
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def logout_view(request):
    username = request.user.login_id if request.user.is_authenticated else "Anonymous"
    logger.info(f"User {username} logged out")
    logout(request)  # Django의 logout 함수로 세션 정리
    return render(request, 'login.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # Django 인증 시스템으로 사용자 검증
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Django 세션에 사용자 로그인
            logger.info(f"User {username} logged in successfully")
            return redirect('main:main')
        else:
            logger.warning(f"Failed login attempt for {username}: Incorrect username or password")
            return render(request, 'login.html', {'error': '잘못된 아이디 또는 비밀번호입니다.'})

    return render(request, 'login.html')

def signup_view(request):
    if request.method == 'POST':
        login_id = request.POST['login_id']
        password = request.POST['pwd']
        pwd_confirm = request.POST['pwd_confirm']
        nickname = request.POST['nickname']
        email = request.POST['email']

        # 비밀번호 일치 확인
        if password != pwd_confirm:
            return render(request, 'signup.html', {'error': '비밀번호가 일치하지 않습니다.'})

        # 중복 체크
        if User.objects.filter(login_id=login_id).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 아이디입니다.'})
        if User.objects.filter(nickname=nickname).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 닉네임입니다.'})
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 이메일입니다.'})

        # 사용자 생성 (UserManager 사용)
        user = User.objects.create_user(
            login_id=login_id,
            email=email,
            nickname=nickname,
            password=password  # 비밀번호는 set_password로 자동 해싱
        )
        logger.info(f"User {login_id} signed up successfully")
        return redirect('account:login')
    return render(request, 'signup.html')