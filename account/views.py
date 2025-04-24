from django.shortcuts import render, redirect
from .models import User
from django.http import JsonResponse

# Create your views here.
def login_view(request):
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
        return redirect('account:login')
    return render(request, 'signup.html')