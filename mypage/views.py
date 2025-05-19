from django.shortcuts import render, redirect, get_object_or_404
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
from django.http import JsonResponse
from account.models import ReportedUser
from account.models import BlockedUser
from django.views.decorators.http import require_http_methods

from account.models import BlockedUser
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


@login_required
def unblock_user(request, blocked_id):
    """
    현재 로그인된 사용자가 특정 사용자를 차단 해제하는 기능
    """
    if request.method == 'POST':
        logger.debug(f"Unblock request received for blocked_id: {blocked_id}")
        blocker = request.user
        blocked_user = get_object_or_404(User, user_id=blocked_id)
        block_relation = BlockedUser.objects.filter(blocker=blocker, blocked=blocked_user).first()
        if not block_relation:
            logger.warning(f"No block relation found for blocker: {blocker.user_id}, blocked: {blocked_id}")
            return redirect('mypage:mypage')
        block_relation.delete()
        logger.info(f"Blocked user {blocked_user.nickname} (user_id: {blocked_id}) unblocked by {blocker.nickname}")
        return redirect('mypage:mypage')
    logger.debug(f"Non-POST request to unblock_user: {request.method}")
    return redirect('mypage:mypage')

@login_required
def mypage(request):
    return render(request, 'mypage/mypage.html')

# 로거 설정
logger = logging.getLogger(__name__)

@login_required
def mypage_view(request):
    #add1
    # 내가 쓴 글 목록 (삭제되지 않은 것만)
    my_posts_qs = FreeBoard.objects.filter(user=request.user, is_deleted=False).order_by('-reg_dt')
    post_paginator = Paginator(my_posts_qs, 5)  # 5개씩
    post_page_number = request.GET.get('my_posts_page', 1)
    try:
        my_posts_page_obj = post_paginator.page(post_page_number)
    except (EmptyPage, PageNotAnInteger):
        my_posts_page_obj = post_paginator.page(1)

    my_posts_qs = FreeBoard.objects.filter(user=request.user, is_deleted=False).order_by('-reg_dt')
    post_paginator = Paginator(my_posts_qs, 5)
    post_page_number = request.GET.get('my_posts_page', 1)
    try:
        my_posts_page_obj = post_paginator.page(post_page_number)
    except (EmptyPage, PageNotAnInteger):
        my_posts_page_obj = post_paginator.page(1)

    # 페이지네이션 범위 계산
    total_pages = post_paginator.num_pages
    current = my_posts_page_obj.number
    start_page = max(current - 5, 1)
    end_page = min(start_page + 9, total_pages)
    if end_page - start_page < 9:
        start_page = max(end_page - 9, 1)
    #add2

    # BlockedUser 모델에서 blocker가 현재 유저인 것만 조회
    blocked_users = BlockedUser.objects.filter(blocker=request.user).select_related('blocked').order_by('-created_at')
    
    # 차단한 유저 정보 리스트 생성
    blocked_users_data = []
    for blocked in blocked_users:
        blocked_user = blocked.blocked
        if blocked_user:
            blocked_users_data.append({
                'id': blocked_user.user_id,
                'nickname': getattr(blocked_user, 'nickname', None) or getattr(blocked_user, 'username', '알 수 없음'),
                'profile_image_url': getattr(blocked_user, 'profile_image', None).url if getattr(blocked_user, 'profile_image', None) else None,
            })

    paginator = Paginator(blocked_users_data, 5)
    page_number = request.GET.get('page', 1)
    try:
        page_obj = paginator.page(page_number)
    except (EmptyPage, PageNotAnInteger):
        page_obj = paginator.page(1)

    context = {
        'page_obj': page_obj,
        'my_posts_page_obj': my_posts_page_obj,
        'page_range': range(start_page, end_page + 1),
    }
    return render(request, 'mypage.html', context)

@login_required
def update_greeting_message(request):
    """
    사용자의 인사 메시지를 저장하거나 업데이트하는 뷰입니다.
    """
    if request.method == 'POST':
        greeting_message = request.POST.get('greeting_message', '').strip()
        logger.debug(f"Received greeting message: {greeting_message}")

        # 유효성 검사
        if len(greeting_message) > 100:
            logger.warning(f"Greeting message too long: {len(greeting_message)} characters")
            return JsonResponse({'success': False, 'message': '인사 메시지는 최대 100자까지 가능합니다.'}, status=400)

        try:
            user = request.user
            user.greeting_message = greeting_message
            user.save(update_fields=['greeting_message'])
            logger.info(f"Greeting message updated for user {user.login_id}: {greeting_message}")
            return JsonResponse({'success': True, 'message': '인사 메시지가 성공적으로 저장되었습니다.'})
        except Exception as e:
            logger.error(f"Failed to update greeting message for user {request.user.login_id}: {str(e)}")
            return JsonResponse({'success': False, 'message': '저장 중 오류가 발생했습니다.'}, status=500)

    return JsonResponse({'success': False, 'message': '잘못된 요청입니다.'}, status=400)

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