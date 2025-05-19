from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.urls import reverse
from .models import FreeBoard, FreeBoardComment, FreeBoardLike, DartDisclosure, NewsArticle, Notification
from django.utils import timezone as django_timezone
from django.contrib import messages
from django.db import transaction
import logging
from datetime import datetime, timedelta
from django.db.models import Q
import re

from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger, Page
from account.models import BlockedUser
from django.views.decorators.http import require_POST

logger = logging.getLogger(__name__)

#add1
@login_required
@require_POST
def unblock_user(request):
    blocked_id = request.POST.get('blocked_id')
    if not blocked_id:
        return JsonResponse({'error': '차단 해제할 유저 정보가 없습니다.'}, status=400)
    try:
        blocked_user_obj = BlockedUser.objects.get(blocker=request.user, blocked_id=blocked_id)
        blocked_user_obj.delete()
        return JsonResponse({'status': 'success'})
    except BlockedUser.DoesNotExist:
        return JsonResponse({'error': '이미 차단 해제된 유저입니다.'}, status=404)
#add2

@login_required
def report_user(request, post_id):
    # 현재는 기본 템플릿을 렌더링하지만, 실제 신고 로직 구현 필요
    post = get_object_or_404(FreeBoard, id=post_id)
    # 신고 관련 로직 (예: Report 모델에 저장)
    messages.info(request, f"'{post.title}' 게시글에 대한 신고가 접수되었습니다. (실제 처리 로직 필요)")
    return redirect('community:detail', post_id=post_id)

@login_required
def block_user(request, post_id):
    if request.method != 'POST':
        messages.error(request, '잘못된 요청입니다.')
        return redirect('community:detail', post_id=post_id)

    if not request.user.is_authenticated:
        messages.error(request, '로그인이 필요합니다.')
        return redirect('account:login')

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    target_user = post.user

    if request.user == target_user:
        messages.error(request, '자기 자신을 차단할 수 없습니다.')
        return redirect('community:detail', post_id=post_id)

    blocked_user_obj, created = BlockedUser.objects.get_or_create(
        blocker=request.user,
        blocked=target_user
    )

    if created:
        messages.success(request, f'{target_user.nickname if hasattr(target_user, "nickname") and target_user.nickname else target_user.username}님을 차단했습니다.')
    else:
        messages.info(request, f'{target_user.nickname if hasattr(target_user, "nickname") and target_user.nickname else target_user.username}님은 이미 차단된 상태입니다.')

    return redirect('community:community')


def notifications_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인이 필요합니다.'}, status=401)

    notifications_qs = Notification.objects.filter(
        recipient=request.user
    ).select_related('sender', 'comment__free_board').order_by('-created_at')[:3] 

    notification_data = []
    now = django_timezone.now()
    for notif in notifications_qs:
        time_diff = now - notif.created_at
        time_ago = "방금 전"
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"

        sender_name = notif.sender.nickname if hasattr(notif.sender, 'nickname') and notif.sender.nickname else notif.sender.username
        
        preview_message = notif.message 
        if notif.comment and not preview_message: 
            preview_message = notif.comment.content[:30] + '...' if len(notif.comment.content) > 30 else notif.comment.content

        notification_data.append({
            'user': sender_name,
            'preview': preview_message,
            'time': time_ago,
            'link': notif.get_absolute_url(),
            'is_read': notif.is_read,
            'id': notif.id 
        })

    return JsonResponse(notification_data, safe=False)


def extract_api_disclosure_info(content_str):
    company_name = "정보 없음"
    report_link = "#"
    if isinstance(content_str, str): 
        company_match = re.search(r"회사명: (.*?)\n", content_str)
        if company_match:
            company_name = company_match.group(1).strip()
        link_match = re.search(r"공시 원문 보기: (https?://[^\s]+)", content_str)
        if link_match:
            report_link = link_match.group(1).strip()
    return company_name, report_link

# 나머지 뷰는 이전과 동일
def community_view(request):
    tab = request.GET.get('tab', 'community')
    subtab = request.GET.get('subtab', '')

    blocked_user_ids = []
    if request.user.is_authenticated:
        blocked_user_ids = BlockedUser.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)

    disclosure_posts_for_carousel_qs = FreeBoard.objects.filter(
        Q(category='API공시') | Q(category='수동공시'),
        is_deleted=False
    ).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('-reg_dt')[:5]

    processed_carousel_disclosure_list = []
    for post in disclosure_posts_for_carousel_qs:
        if not post.user:
            logger.warning(f"Post {post.id} has no associated user")
            continue
        item_data = {
            'obj': post,
            'display_title': post.title,
            'display_company_name': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
            'display_date': post.reg_dt,
            'display_category': post.get_category_display(),
            'link': post.get_absolute_url(),
            'is_api': (post.category == 'API공시')
        }
        if post.category == 'API공시':
            company_name, report_link = extract_api_disclosure_info(post.content)
            item_data['display_company_name'] = company_name
            item_data['link'] = report_link
        processed_carousel_disclosure_list.append(item_data)

    paginator_for_carousel = Paginator(processed_carousel_disclosure_list, 5)
    try:
        carousel_page_obj = paginator_for_carousel.page(1)
    except EmptyPage:
        carousel_page_obj = Page([], 1, paginator_for_carousel)

    common_context_data = {
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
        'ticker_message': '코박 예측 정보: 비트코인 단기 상승 전망!',
    }

    if tab == 'news':
        active_subtab = subtab if subtab in ['realtime', 'disclosure'] else 'realtime'
        if active_subtab == 'realtime':
            realtime_posts_qs = FreeBoard.objects.filter(
                category='실시간뉴스',
                is_deleted=False
            ).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('-reg_dt')
            
            paginator_realtime = Paginator(realtime_posts_qs, 10)
            page_number_realtime = request.GET.get('page', 1)
            try:
                realtime_page_obj = paginator_realtime.page(page_number_realtime)
            except (EmptyPage, PageNotAnInteger):
                realtime_page_obj = paginator_realtime.page(1)
            
            posts_for_template = realtime_page_obj
            page_obj_for_template = realtime_page_obj
        else:
            posts_for_template = None
            page_obj_for_template = None

        disclosure_posts_qs = FreeBoard.objects.filter(
            Q(category='수동공시') | Q(category='API공시'),
            is_deleted=False
        ).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('-reg_dt')

        processed_disclosure_list = []
        for post in disclosure_posts_qs:
            if not post.user:
                logger.warning(f"Post {post.id} has no associated user")
                continue
            item_data = {
                'obj': post,
                'display_title': post.title,
                'display_company_name': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
                'display_date': post.reg_dt,
                'display_category': post.get_category_display(),
                'link': post.get_absolute_url(),
                'is_api': (post.category == 'API공시')
            }
            if post.category == 'API공시':
                company_name, report_link = extract_api_disclosure_info(post.content)
                item_data['display_company_name'] = company_name
                item_data['link'] = report_link
            processed_disclosure_list.append(item_data)

        paginator_disclosure = Paginator(processed_disclosure_list, 10)
        page_number_disclosure = request.GET.get('page', 1)
        try:
            disclosure_list_page_obj = paginator_disclosure.page(page_number_disclosure)
        except (EmptyPage, PageNotAnInteger):
            disclosure_list_page_obj = paginator_disclosure.page(1)

        context = {
            **common_context_data,
            'disclosures': carousel_page_obj,
            'active_tab': 'news',
            'active_subtab': active_subtab,
            'posts': posts_for_template,
            'page_obj': page_obj_for_template,
            'disclosure_list_page_obj': disclosure_list_page_obj,
        }
        return render(request, 'community_news.html', context)

    else:  # tab == 'community'
        period = request.GET.get('period', '한달')
        sort = request.GET.get('sort', '최신순')
        search_query = request.GET.get('q', '').strip()
        now = django_timezone.now()

        all_posts_queryset = FreeBoard.objects.filter(
            is_deleted=False,
            category='잡담',
            reg_dt__gte=now - timedelta(days={'하루': 1, '일주일': 7, '한달': 30, '반년': 180}.get(period, 30))
        ).exclude(user_id__in=blocked_user_ids).select_related('user')

        # 검색어 필터링
        if search_query:
            all_posts_queryset = all_posts_queryset.filter(
                Q(title__icontains=search_query) | Q(content__icontains=search_query)
            )

        # 정렬
        sort_fields = {
            '최신순': '-reg_dt',
            '조회수순': '-view_count',
            '중요순': '-likes_count',
            '걱정순': '-worried_count',
        }
        all_posts_queryset = all_posts_queryset.order_by(sort_fields.get(sort, '-reg_dt'))

        paginator_community = Paginator(all_posts_queryset, 10)
        page_number_community = request.GET.get('page', 1)
        try:
            community_page_obj = paginator_community.page(page_number_community)
        except (EmptyPage, PageNotAnInteger):
            community_page_obj = paginator_community.page(1)

        processed_post_list = []
        liked_post_ids = set()
        if request.user.is_authenticated:
            post_ids = [post.id for post in community_page_obj.object_list]
            liked_post_ids = set(FreeBoardLike.objects.filter(
                user=request.user,
                free_board_id__in=post_ids,
                is_liked=True
            ).values_list('free_board_id', flat=True))

        for post in community_page_obj.object_list:
            if not post.user:
                logger.warning(f"Post {post.id} has no associated user")
                continue
            time_diff = now - post.reg_dt
            days_ago = time_diff.days
            time_ago_str = "방금 전"
            if days_ago == 0:
                if time_diff.seconds < 60: time_ago_str = "방금 전"
                elif time_diff.seconds // 60 < 60: time_ago_str = f"{time_diff.seconds // 60}분 전"
                else: time_ago_str = f"{time_diff.seconds // 3600}시간 전"
            else: time_ago_str = f"{days_ago}일 전"

            processed_post_list.append({
                'id': post.id,
                'user': post.user,
                'username': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
                'auth_id': getattr(post.user, 'auth_id', ''),
                'category': getattr(post, 'category', '잡담'),
                'time_ago': time_ago_str,
                'days_ago': days_ago,
                'title': post.title,
                'content': post.content,
                'likes_count': post.likes_count,
                'comments_count': post.comments_count,
                'view_count': post.view_count,
                'worried_count': post.worried_count,
                'reg_dt': post.reg_dt,
                'is_liked': post.id in liked_post_ids,
                'get_absolute_url': post.get_absolute_url(),
                'thumbnail': getattr(post, 'thumbnail', None),
                'image': post.image if hasattr(post, 'image') else None,
            })

        context = {
            **common_context_data,
            'disclosures_for_carousel': carousel_page_obj,
            'posts': processed_post_list,
            'page_obj': community_page_obj,
            'period': period,
            'sort': sort,
            'search_query': search_query,
            'active_tab': 'community',
            'active_subtab': '',
        }
        return render(request, 'community.html', context)

@login_required
def write_view(request):
    if not request.user.is_authenticated: 
        next_url = request.path
        if request.GET:
            next_url += '?' + request.GET.urlencode()
        return HttpResponseRedirect(f"{reverse('account:login')}?next={next_url}")

    board_type = request.GET.get('board_type', 'freeboard') 

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip() 
        captcha_answer = request.POST.get('captcha_answer', '').strip() 

        error_message = None
        if not title or not content:
            error_message = '제목과 내용을 모두 입력해주세요.'
        if not captcha_value.isdigit() or len(captcha_value) != 4:
             error_message = '자동 입력 방지 문자를 올바르게 입력해주세요. (숫자 4자리)'
        elif captcha_answer != captcha_value : 
             error_message = '자동 입력 방지 문자가 일치하지 않습니다.'


        if error_message:
            messages.error(request, error_message)
            return render(request, 'community_write.html', {
                'title': title,
                'content': content,
                'board_type': board_type,
                'error_message': error_message, 
                'is_edit': False
            })

        post_category = '잡담' 
        redirect_url_name = 'community:community'
        redirect_params = {}

        if board_type == 'realtime_news':
            messages.info(request, "실시간 뉴스 게시물은 현재 관리자만 작성 가능합니다. (예시)")
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            post_category = '수동공시'
            redirect_url_name = 'community:community' 
            redirect_params = {'tab': 'news', 'subtab': 'disclosure'}
        
        new_post = FreeBoard.objects.create(
            user=request.user,
            title=title,
            content=content,
            category=post_category,
            image=request.FILES.get('image') # 이미지 업로드 처리
        )        
        if redirect_params:
            query_string = '&'.join([f"{k}={v}" for k, v in redirect_params.items()])
            return redirect(f"{reverse(redirect_url_name)}?{query_string}")
        return redirect(redirect_url_name)
 
    return render(request, 'community_write.html', {
        'board_type': board_type,
        'is_edit': False
    })

def community_detail_view(request, post_id):
    blocked_user_ids = []
    if request.user.is_authenticated:
        blocked_user_ids = BlockedUser.objects.filter(blocker=request.user).values_list('blocked_id', flat=True)

    post_obj = get_object_or_404(FreeBoard.objects.select_related('user').exclude(user_id__in=blocked_user_ids), id=post_id, is_deleted=False)

    session_key_viewed = f'viewed_post_{post_obj.id}'
    if not request.session.get(session_key_viewed, False):
        with transaction.atomic(): 
            post_for_update = FreeBoard.objects.select_for_update().get(id=post_obj.id)
            post_for_update.view_count += 1
            post_for_update.save(update_fields=['view_count'])
            post_obj.view_count = post_for_update.view_count 
        request.session[session_key_viewed] = True
        request.session.set_expiry(timedelta(days=1)) 

    comments_qs = FreeBoardComment.objects.filter(
        free_board=post_obj,
        is_deleted=False
    ).exclude(user_id__in=blocked_user_ids).select_related('user').order_by('reg_dt')
    
    comments_data = []
    for comment in comments_qs:
        comment_time_diff = django_timezone.now() - comment.reg_dt
        comment_time_ago = "방금 전"
        if comment_time_diff.days > 0: comment_time_ago = f"{comment_time_diff.days}일 전"
        elif comment_time_diff.seconds // 3600 > 0: comment_time_ago = f"{comment_time_diff.seconds // 3600}시간 전"
        elif comment_time_diff.seconds // 60 > 0: comment_time_ago = f"{comment_time_diff.seconds // 60}분 전"
        
        comments_data.append({
            'id': comment.id,
            'user': comment.user, 
            'username': comment.user.nickname if hasattr(comment.user, 'nickname') and comment.user.nickname else comment.user.username,
            'auth_id': comment.user.auth_id if hasattr(comment.user, 'auth_id') else '',
            'content': comment.content,
            'reg_dt_formatted': comment.reg_dt.strftime('%Y.%m.%d %H:%M'),
            'time_ago': comment_time_ago,
            'is_author': request.user == comment.user,
            'profile_image_url': comment.user.profile_image.url if hasattr(comment.user, 'profile_image') and comment.user.profile_image else None,
        })


    time_diff = django_timezone.now() - post_obj.reg_dt
    time_ago = "방금 전"
    if time_diff.days > 0: time_ago = f"{time_diff.days}일 전"
    elif time_diff.seconds // 3600 > 0: time_ago = f"{time_diff.seconds // 3600}시간 전"
    elif time_diff.seconds // 60 > 0: time_ago = f"{time_diff.seconds // 60}분 전"

    is_liked = False
    is_worried = False
    if request.user.is_authenticated:
        like_obj = FreeBoardLike.objects.filter(free_board=post_obj, user=request.user).first()
        if like_obj:
            is_liked = like_obj.is_liked
            is_worried = like_obj.is_worried

    dart_link_for_detail = None
    company_name_for_detail = None
    if post_obj.category == 'API공시':
        company_name_for_detail, dart_link_for_detail = extract_api_disclosure_info(post_obj.content)

    # Replace "[사진]" with the image tag if an image exists
    content_with_image = post_obj.content.replace("[사진]", f'<img src="{post_obj.image.url}" alt="Uploaded Image">' if post_obj.image else "")
    
    post_data = {
        'id': post_obj.id,
        'user': post_obj.user, 
        'username': post_obj.user.nickname if hasattr(post_obj.user, 'nickname') and post_obj.user.nickname else post_obj.user.get_username(),
        'auth_id': post_obj.user.auth_id if hasattr(post_obj.user, 'auth_id') else '',
        'profile_image_url': post_obj.user.profile_image.url if hasattr(post_obj.user, 'profile_image') and post_obj.user.profile_image else None,
        'title': post_obj.title,
        'content': content_with_image,  # Updated content with image
        'image_url': post_obj.image.url if post_obj.image else None, # 이미지 URL 추가
        'time_ago': time_ago,
        'reg_dt_formatted': post_obj.reg_dt.strftime('%Y.%m.%d %H:%M'),
        'likes_count': post_obj.likes_count,
        'worried_count': post_obj.worried_count,
        'comments_count': post_obj.comments.filter(is_deleted=False).exclude(user_id__in=blocked_user_ids).count(), 
        'view_count': post_obj.view_count,
        'is_liked_by_user': is_liked, 
        'is_worried_by_user': is_worried, 
        'is_author': request.user == post_obj.user,
        'category': post_obj.category, 
        'category_display': post_obj.get_category_display(),
        'dart_link': dart_link_for_detail,
        'company_name_for_api_disclosure': company_name_for_detail,
    }
    return render(request, 'community_detail.html', {'post': post_data, 'comments': comments_data})


@login_required
def like_post(request, post_id):
    if not request.user.is_authenticated: 
        return JsonResponse({'error': '로그인이 필요합니다.'}, status=401)

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if post.user == request.user: 
        return JsonResponse({'error': '자신의 게시글에는 반응할 수 없습니다.'}, status=400)

    action = request.POST.get('action') 
    if action not in ['like', 'worry']:
        return JsonResponse({'error': '잘못된 요청입니다: action 값이 필요합니다.'}, status=400)

    response_data = {'status': 'success'}
    with transaction.atomic():
        like_obj, created = FreeBoardLike.objects.get_or_create(
            free_board=post,
            user=request.user,
            defaults={'is_liked': False, 'is_worried': False}
        )
        
        post_for_update = FreeBoard.objects.select_for_update().get(id=post.id)

        if action == 'like':
            if like_obj.is_liked: 
                like_obj.is_liked = False
                post_for_update.likes_count = max(0, post_for_update.likes_count - 1)
            else: 
                like_obj.is_liked = True
                post_for_update.likes_count += 1
                if like_obj.is_worried: 
                    like_obj.is_worried = False
                    post_for_update.worried_count = max(0, post_for_update.worried_count - 1)
        
        elif action == 'worry':
            if like_obj.is_worried: 
                like_obj.is_worried = False
                post_for_update.worried_count = max(0, post_for_update.worried_count - 1)
            else: 
                like_obj.is_worried = True
                post_for_update.worried_count += 1
                if like_obj.is_liked: 
                    like_obj.is_liked = False
                    post_for_update.likes_count = max(0, post_for_update.likes_count - 1)
        
        like_obj.save()
        post_for_update.save(update_fields=['likes_count', 'worried_count'])

        if not like_obj.is_liked and not like_obj.is_worried:
            like_obj.delete()

        response_data.update({
            'likes_count': post_for_update.likes_count,
            'worried_count': post_for_update.worried_count,
            'is_liked': like_obj.is_liked if FreeBoardLike.objects.filter(pk=like_obj.pk).exists() else False,
            'is_worried': like_obj.is_worried if FreeBoardLike.objects.filter(pk=like_obj.pk).exists() else False,
        })

    return JsonResponse(response_data)

@login_required
@require_POST
def comment_create(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    content = request.POST.get('content', '').strip()
    if not content:
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'error': '댓글 내용을 입력해주세요.'}, status=400)
        else:
            messages.error(request, '댓글 내용을 입력해주세요.')
            return redirect('community:detail', post_id=post_id)

    with transaction.atomic():
        comment = FreeBoardComment.objects.create(free_board=post, user=request.user, content=content)
        post_for_update = FreeBoard.objects.select_for_update().get(id=post.id)
        post_for_update.comments_count += 1
        post_for_update.save(update_fields=['comments_count'])

        if post.user != request.user:
            notification_message = f"{request.user.nickname or request.user.username}님이 회원님의 게시글에 댓글을 남겼습니다: \"{content[:20]}...\""
            if len(content) <= 20:
                notification_message = f"{request.user.nickname or request.user.username}님이 회원님의 게시글에 댓글을 남겼습니다: \"{content}\""
            Notification.objects.create(
                recipient=post.user,
                sender=request.user,
                comment=comment,
                message=notification_message
            )

    # AJAX 요청이면 JSON 반환, 아니면 리다이렉트
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        comment_time_ago = "방금 전"
        comment_data = {
            'id': comment.id,
            'user': {
                'nickname': request.user.nickname or request.user.get_username(),
                'auth_id': getattr(request.user, 'auth_id', ''),
                'profile_image_url': request.user.profile_image.url if hasattr(request.user, 'profile_image') and request.user.profile_image else None,
            },
            'content': comment.content,
            'reg_dt_formatted': comment.reg_dt.strftime('%Y.%m.%d %H:%M'),
            'time_ago': comment_time_ago,
            'is_author': True,
        }
        return JsonResponse({
            'status': 'success',
            'comment': comment_data,
            'comments_count': post_for_update.comments_count
        })
    else:
        return redirect('community:detail', post_id=post_id)

@login_required
def edit_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)

    if post.category == '수동공시':
        board_type = 'disclosure_manual'
    elif post.category == '실시간뉴스': 
        board_type = 'realtime_news'
    else: 
        board_type = 'freeboard'


    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip()
        captcha_answer = request.POST.get('captcha_answer', '').strip() 

        error_message = None
        if not title or not content:
            error_message = '제목과 내용을 모두 입력해주세요.'
        if not captcha_value.isdigit() or len(captcha_value) != 4:
             error_message = '자동 입력 방지 문자를 올바르게 입력해주세요. (숫자 4자리)'
        elif captcha_answer != captcha_value :
             error_message = '자동 입력 방지 문자가 일치하지 않습니다.'

        if error_message:
            messages.error(request, error_message)
            return render(request, 'community_write.html', {
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
                'error_message': error_message,
                'image_url': post.image.url if post.image else None, # 기존 이미지 URL
            })

        post.title = title
        post.content = content
        if request.FILES.get('image'): # 새 이미지 파일이 있으면 업데이트
            post.image = request.FILES.get('image')
        elif request.POST.get('remove_image') == 'on': # 이미지 삭제 체크박스
            post.image = None

        post.save(update_fields=['title', 'content', 'up_dt', 'image']) 
        return redirect('community:detail', post_id=post_id)
 
    return render(request, 'community_write.html', {
        'title': post.title,
        'content': post.content,
        'post_id': post_id,
        'is_edit': True,
        'board_type': board_type, 
        'image_url': post.image.url if post.image else None, # 기존 이미지 URL
    })


@login_required
def delete_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.id)

    if request.method == 'POST': 
        post.is_deleted = True
        post.save(update_fields=['is_deleted'])
        original_category = getattr(post, 'category', '잡담')
        if original_category == '실시간뉴스':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif original_category in ['수동공시', 'API공시']:
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        return redirect('community:community') 

    return render(request, 'community_delete.html', {'post': post})


@login_required
def comment_edit(request, pk): 
    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        messages.error(request, "본인이 작성한 댓글만 수정할 수 있습니다.")
        return redirect('community:detail', post_id=comment.free_board.id)

    if request.method == "POST":
        content = request.POST.get('content', '').strip()
        if content:
            comment.content = content
            comment.save(update_fields=['content', 'up_dt']) 
            messages.success(request, "댓글이 수정되었습니다.")
        else:
            messages.error(request, "댓글 내용을 입력해 주세요.")
        return redirect(f"{comment.free_board.get_absolute_url()}#comment-{comment.id}") 

    return redirect('community:detail', post_id=comment.free_board.id)


@login_required
def comment_delete(request, pk): 
    if not request.user.is_authenticated: 
        return JsonResponse({'error': '로그인이 필요합니다.'}, status=401)

    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        return JsonResponse({'error': '삭제 권한이 없습니다.'}, status=403)

    if request.method == 'POST': 
        with transaction.atomic():
            comment.is_deleted = True
            comment.save(update_fields=['is_deleted'])
            
            post = comment.free_board
            post_for_update = FreeBoard.objects.select_for_update().get(id=post.id)
            post_for_update.comments_count = max(0, post_for_update.comments_count - 1)
            post_for_update.save(update_fields=['comments_count'])

        return JsonResponse({
            'status': 'success',
            'comment_id': pk,
            'comments_count': post_for_update.comments_count
        })

    return JsonResponse({'error': '잘못된 요청입니다. (POST 요청 필요)'}, status=400)
