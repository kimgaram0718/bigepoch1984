from django.shortcuts import render, redirect
from django.contrib.auth import login
from .models import User
from django.http import JsonResponse
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def logout_view(request):
    if 'user_id' in request.session:
        username = User.objects.get(user_id=request.session['user_id']).login_id
        logger.info(f"User {username} logged out")
        request.session.flush()  # 세션 전체 삭제
    return render(request, 'login.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        try:
            # 커스텀 User 모델에서 사용자 조회
            user = User.objects.get(login_id=username)
            # 비밀번호 확인 (평문 비밀번호 가정)
            if user.pwd == password:
                # Django 세션에 사용자 로그인
                request.session['user_id'] = user.user_id
                request.session['user_nickname'] = user.nickname
                logger.info(f"User {username} logged in successfully")
                return redirect('main:main')
            else:
                logger.warning(f"Failed login attempt for {username}: Incorrect password")
                return render(request, 'login.html', {'error': '잘못된 아이디 또는 비밀번호입니다.'})
        except User.DoesNotExist:
            logger.warning(f"Failed login attempt: User {username} does not exist")
            return render(request, 'login.html', {'error': '잘못된 아이디 또는 비밀번호입니다.'})

    return render(request, 'login.html')

def signup_view(request):
    if request.method == 'POST':
        login_id = request.POST['login_id']
        pwd = request.POST['pwd']
        pwd_confirm = request.POST['pwd_confirm']
        nickname = request.POST['nickname']
        email = request.POST['email']

        # 비밀번호 일치 확인
        if pwd != pwd_confirm:
            return render(request, 'signup.html', {'error': '비밀번호가 일치하지 않습니다.'})

        # 중복 체크
        if User.objects.filter(login_id=login_id).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 아이디입니다.'})
        if User.objects.filter(nickname=nickname).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 닉네임입니다.'})
        if User.objects.filter(email=email).exists():
            return render(request, 'signup.html', {'error': '이미 사용 중인 이메일입니다.'})

        # 사용자 생성
        user = User(login_id=login_id, pwd=pwd, nickname=nickname, email=email)
        user.save()
        logger.info(f"User {login_id} signed up successfully")
        return redirect('account:login')
    return render(request, 'signup.html')

# 주석: 비밀번호 해시 처리로 전환하려면 아래와 같이 수정
"""
from django.contrib.auth.hashers import make_password, check_password

# signup_view에서 비밀번호 해시 저장
user = User(login_id=login_id, pwd=make_password(pwd), nickname=nickname, email=email)
user.save()

# login_view에서 비밀번호 확인
if check_password(password, user.pwd):
    request.session['user_id'] = user.user_id
    request.session['user_nickname'] = user.nickname
    return redirect('main:main')
"""