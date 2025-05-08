from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from account.models import User
from community.models import FreeBoard
from django.contrib import messages
import logging
from django.core.files.storage import default_storage
from django.contrib.auth import login

# 로거 설정
logger = logging.getLogger(__name__)

def mypage_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to mypage_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/')
    user = request.user
    user_posts = FreeBoard.objects.filter(user=user, is_deleted=False).values('title')[:5]
    context = {
        'user': {
            'nickname': user.nickname,
            'auth_id': user.auth_id,
            'greeting_message': '인사 메시지를 입력하세요',
        },
        'prediction_items': [
            {'name': '삼성전자', 'price': '82,000원', 'change': '+1.20%'},
            {'name': '비트코인', 'price': '125,000,000원', 'change': '+0.80%'},
        ],
        'watchlist': [
            {'name': '이더리움'},
            {'name': '애플'},
        ],
        'user_posts': user_posts,
    }
    return render(request, 'mypage.html', context)

def edit_profile_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to edit_profile_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    user = request.user

    if request.method == 'POST':
        nickname = request.POST.get('nickname')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        profile_image = request.FILES.get('profile_image')

        logger.debug(f"Received POST data: {request.POST}")
        logger.debug(f"Received FILES data: {request.FILES}")

        if nickname and len(nickname) < 2:
            messages.error(request, '닉네임은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user})

        if nickname != user.nickname and User.objects.filter(nickname=nickname).exists():
            messages.error(request, '이미 사용 중인 닉네임입니다.')
            return render(request, 'edit_profile.html', {'user': user})

        if password:
            if len(password) < 8:
                messages.error(request, '비밀번호는 최소 8자 이상이어야 합니다.')
                return render(request, 'edit_profile.html', {'user': user})
            if password != password_confirm:
                messages.error(request, '비밀번호가 일치하지 않습니다.')
                return render(request, 'edit_profile.html', {'user': user})

        user.nickname = nickname
        password_changed = False
        if password:
            user.set_password(password)
            password_changed = True
        if profile_image:
            if user.profile_image and user.profile_image.name != 'profile_images/default.jpg':
                if default_storage.exists(user.profile_image.name):
                    default_storage.delete(user.profile_image.name)
                    logger.info(f"Deleted old profile image: {user.profile_image.name}")
            user.profile_image = profile_image
            logger.info(f"Updated profile image to: {user.profile_image.name}")
        else:
            logger.info("No new profile image uploaded, keeping existing image")
        user.save()

        if password_changed:
            login(request, user)  # 비밀번호 변경 시 세션 갱신
        logger.info(f"User {user.login_id} updated profile: nickname={user.nickname}, profile_image={user.profile_image.url if user.profile_image else 'default'}")
        messages.success(request, '프로필이 성공적으로 수정되었습니다.')
        return redirect('mypage:mypage')

    return render(request, 'edit_profile.html', {'user': user})