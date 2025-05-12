from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from account.models import User
from community.models import FreeBoard
from django.contrib import messages
import logging
from django.core.files.storage import default_storage
from django.contrib.auth import login
from django.utils import timezone
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def mypage(request):
    return render(request, 'mypage/mypage.html')


# 로거 설정
logger = logging.getLogger(__name__)

def mypage_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to mypage_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/')
    user = request.user
    user_posts = FreeBoard.objects.filter(user=user, is_deleted=False).values('title')[:5]
    context = {
        'user': user,  # 딕셔너리 대신 user 객체 자체를 전달
        'prediction_items': [
            {'name': '삼성전자', 'price': '82,000원', 'change': '+1.20%'},
            {'name': '비트코인', 'price': '125,000,000원', 'change': '+0.80%'},
        ],
        'watchlist': [
            {'name': '이더리움'},
            {'name': '애플'},
        ],
        'user_posts': user_posts,
        'now': timezone.now(),
    }
    return render(request, 'mypage.html', context)

def edit_profile_view(request):
    if not request.user.is_authenticated:
        logger.warning("Unauthorized access to edit_profile_view - redirecting to login")
        return HttpResponseRedirect('/account/login/?next=/mypage/edit/')

    user = request.user

    if request.method == 'POST':
        logger.debug(f"Received POST data: {request.POST}")
        logger.debug(f"Received FILES data: {request.FILES}")

        nickname = request.POST.get('nickname')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        profile_image = request.FILES.get('profile_image')

        # 닉네임 유효성 검사
        if not nickname or len(nickname) < 2:
            logger.warning(f"Invalid nickname: {nickname}")
            messages.error(request, '닉네임은 최소 2자 이상이어야 합니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if nickname != user.nickname and User.objects.filter(nickname=nickname).exists():
            logger.warning(f"Nickname already exists: {nickname}")
            messages.error(request, '이미 사용 중인 닉네임입니다.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        # 비밀번호 유효성 검사
        if password:
            if len(password) < 8:
                logger.warning(f"Password too short: {len(password)} characters")
                messages.error(request, '비밀번호는 최소 8자 이상이어야 합니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})
            if password != password_confirm:
                logger.warning("Password confirmation does not match")
                messages.error(request, '비밀번호가 일치하지 않습니다.')
                return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        # 사용자 정보 업데이트
        user.nickname = nickname
        password_changed = False
        if password:
            user.set_password(password)
            password_changed = True
            logger.info(f"Password updated for user {user.login_id}")

        # 프로필 이미지 처리
        if profile_image:
            logger.debug("Profile image received, processing upload")
            # 기존 이미지 삭제 (기본 이미지가 아닌 경우)
            if user.profile_image and user.profile_image.name != 'profile_images/default.jpg':
                if default_storage.exists(user.profile_image.name):
                    try:
                        default_storage.delete(user.profile_image.name)
                        logger.info(f"Deleted old profile image: {user.profile_image.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete old profile image: {str(e)}")
            user.profile_image = profile_image
            logger.info(f"Assigned new profile image: {user.profile_image.name}")
        else:
            logger.debug("No new profile image uploaded, keeping existing image")

        try:
            user.save()
            logger.info(f"User {user.login_id} updated profile: nickname={user.nickname}, profile_image={user.profile_image.url if user.profile_image else 'default'}")
        except Exception as e:
            logger.error(f"Failed to save user profile: {str(e)}")
            messages.error(request, '프로필 저장 중 오류가 발생했습니다. 다시 시도해주세요.')
            return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})

        if password_changed:
            login(request, user)  # 비밀번호 변경 시 세션 갱신
        messages.success(request, '프로필이 성공적으로 수정되었습니다.')
        return redirect('mypage:mypage')

    return render(request, 'edit_profile.html', {'user': user, 'now': timezone.now()})