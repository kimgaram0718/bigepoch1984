from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from account.models import User
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def mypage_view(request):
    # 세션에서 사용자 정보 확인
    if 'user_id' not in request.session:
        logger.warning("Unauthorized access to mypage_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/mypage/')

    user_id = request.session['user_id']
    try:
        user = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        logger.warning(f"User with ID {user_id} not found - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/mypage/')

    context = {
        'user': {
            'nickname': user.nickname,
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
        'user_posts': [
            {'title': 'LOOM, ARK, HIFI 상승 예상 분석'},
            {'title': '테더 스테이블 코인 동향'},
        ],
    }
    return render(request, 'mypage.html', context)

def edit_profile_view(request):
    # 세션에서 사용자 정보 확인
    if 'user_id' not in request.session:
        logger.warning("Unauthorized access to edit_profile_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    user_id = request.session['user_id']
    try:
        user = User.objects.get(user_id=user_id)
    except User.DoesNotExist:
        logger.warning(f"User with ID {user_id} not found - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    if request.method == 'POST':
        nickname = request.POST.get('nickname')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')

        # 닉네임 검증
        if nickname and len(nickname) < 2:
            return render(request, 'edit_profile.html', {
                'user': user,
                'error': '닉네임은 최소 2자 이상이어야 합니다.'
            })

        # 닉네임 중복 체크 (현재 사용자의 닉네임 제외)
        if nickname != user.nickname and User.objects.filter(nickname=nickname).exists():
            return render(request, 'edit_profile.html', {
                'user': user,
                'error': '이미 사용 중인 닉네임입니다.'
            })

        # 비밀번호 검증
        if password:
            if len(password) < 8:
                return render(request, 'edit_profile.html', {
                    'user': user,
                    'error': '비밀번호는 최소 8자 이상이어야 합니다.'
                })
            if password != password_confirm:
                return render(request, 'edit_profile.html', {
                    'user': user,
                    'error': '비밀번호가 일치하지 않습니다.'
                })

        # 사용자 정보 업데이트
        user.nickname = nickname
        if password:
            user.pwd = password  # 평문 비밀번호 저장 (account와 동일)
        user.save()

        # 세션 업데이트
        request.session['user_nickname'] = user.nickname
        logger.info(f"User {user.login_id} updated profile: nickname={user.nickname}")
        return redirect('mypage:mypage')

    return render(request, 'edit_profile.html', {'user': user})