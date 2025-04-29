from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from account.models import User
from community.models import FreeBoard
from django.contrib import messages
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def mypage_view(request):
    # 인증 확인
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to mypage_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/')

    user = request.user
    # 사용자 게시물 조회
    user_posts = FreeBoard.objects.filter(user=user, is_deleted=False).values('title')[:5]  # 최대 5개

    context = {
        'user': {
            'nickname': user.nickname,
            'auth_id': user.auth_id,  # 운영자/일반회원 구분
            'greeting_message': '인사 메시지를 입력하세요',  # 추후 사용자 모델에 추가 가능
        },
        'prediction_items': [
            {'name': '삼성전자', 'price': '82,000원', 'change': '+1.20%'},
            {'name': '비트코인', 'price': '125,000,000원', 'change': '+0.80%'},
        ],
        'watchlist': [
            {'name': '이더리움'},
            {'name': '애플'},
        ],
        'user_posts': user_posts,  # 실제 게시물
    }
    return render(request, 'mypage.html', context)

def edit_profile_view(request):
    # 인증 확인
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to edit_profile_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    user = request.user

    if request.method == 'POST':
        nickname = request.POST.get('nickname')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')

        # 닉네임 검증
        if nickname and len(nickname) < 2:
            messages.error(request, '닉네임은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user})

        # 닉네임 중복 체크 (현재 사용자의 닉네임 제외)
        if nickname != user.nickname and User.objects.filter(nickname=nickname).exists():
            messages.error(request, '이미 사용 중인 닉네임입니다.')
            return render(request, 'edit_profile.html', {'user': user})

        # 비밀번호 검증
        if password:
            if len(password) < 8:
                messages.error(request, '비밀번호는 최소 8자 이상이어야 합니다.')
                return render(request, 'edit_profile.html', {'user': user})
            if password != password_confirm:
                messages.error(request, '비밀번호가 일치하지 않습니다.')
                return render(request, 'edit_profile.html', {'user': user})

        # 사용자 정보 업데이트
        user.nickname = nickname
        if password:
            user.set_password(password)  # 비밀번호 해시 처리
        user.save()

        logger.info(f"User {user.login_id} updated profile: nickname={user.nickname}")
        messages.success(request, '프로필이 성공적으로 수정되었습니다.')
        return redirect('mypage:mypage')

    return render(request, 'edit_profile.html', {'user': user})