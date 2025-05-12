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

logger = logging.getLogger(__name__)

#add1
def notifications_view(request):
    """
    로그인된 사용자의 알림 데이터를 JSON 형식으로 반환합니다.
    최신 3개만 반환하며, 읽음 여부도 포함.
    """
    if not request.user.is_authenticated:
        return JsonResponse({'error': '로그인 필요'}, status=401)

    notifications = Notification.objects.filter(
        recipient=request.user
    ).select_related('sender', 'comment__free_board')[:3]  # 최신 3개만

    notification_data = []
    for notif in notifications:
        time_diff = django_timezone.now() - notif.created_at
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"
        else:
            time_ago = "방금 전"

        sender_name = notif.sender.nickname if hasattr(notif.sender, 'nickname') and notif.sender.nickname else notif.sender.username
        notification_data.append({
            'user': sender_name,
            'preview': notif.message,  # message 필드에 댓글 내용만 포함
            'time': time_ago,
            'link': notif.get_absolute_url(),
            'is_read': notif.is_read
        })

    return JsonResponse(notification_data, safe=False)
#add2

def extract_api_disclosure_info(content_str):
    """
    API 공시 내용에서 회사명과 원문 링크를 추출합니다.
    """
    company_name = "정보 없음"
    report_link = "#"
    company_match = re.search(r"회사명: (.*?)\n", content_str)
    if company_match:
        company_name = company_match.group(1).strip()
    link_match = re.search(r"공시 원문 보기: (https?://[^\s]+)", content_str)
    if link_match:
        report_link = link_match.group(1).strip()
    return company_name, report_link

def community_view(request):
    """
    커뮤니티 메인 페이지 및 뉴스 페이지를 처리하는 뷰입니다.
    탭(community, news)과 서브탭(realtime, disclosure)에 따라 다른 컨텐츠를 보여줍니다.
    """
    tab = request.GET.get('tab', 'community')
    subtab = request.GET.get('subtab', '') 

    context = {
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
        'active_tab': tab,
        'active_subtab': subtab if subtab else ('realtime' if tab == 'news' else ''),
    }

    # 커뮤니티 페이지 캐러셀을 위한 공시 데이터 준비 (최대 5개)
    disclosure_posts_for_carousel_qs = FreeBoard.objects.filter(
        Q(category='API공시') | Q(category='수동공시'),
        is_deleted=False
    ).select_related('user').order_by('-reg_dt')[:5] # 최근 5개 항목만 가져옴

    processed_carousel_disclosure_list = []
    for post in disclosure_posts_for_carousel_qs:
        item_data = {
            'obj': post, 'display_title': post.title,
            'display_company_name': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
            'display_date': post.reg_dt, 'display_category': post.get_category_display(),
            'link': post.get_absolute_url(), 'is_api': False
        }
        if post.category == 'API공시':
            company_name, report_link = extract_api_disclosure_info(post.content)
            item_data['display_company_name'] = company_name
            item_data['link'] = report_link 
            item_data['is_api'] = True
        processed_carousel_disclosure_list.append(item_data)

    paginator_for_carousel = Paginator(processed_carousel_disclosure_list, 5) 
    try:
        page_for_community_carousel = paginator_for_carousel.page(1)
    except EmptyPage:
        page_for_community_carousel = Page([], 1, paginator_for_carousel)


    if tab == 'news':
        page_param_from_url = request.GET.get('page', '1')
        try:
            current_page_for_active_tab = int(page_param_from_url)
        except ValueError:
            current_page_for_active_tab = 1

        url_active_subtab = subtab if subtab else 'realtime'

        news_article_list = NewsArticle.objects.all().order_by('-pub_date')
        paginator_realtime = Paginator(news_article_list, 10)
        page_num_for_realtime = current_page_for_active_tab if url_active_subtab == 'realtime' else 1
        try:
            realtime_page_obj = paginator_realtime.page(page_num_for_realtime)
        except (EmptyPage, PageNotAnInteger):
            realtime_page_obj = paginator_realtime.page(1)

        disclosure_posts_qs = FreeBoard.objects.filter(
            Q(category='API공시') | Q(category='수동공시'),
            is_deleted=False
        ).select_related('user').order_by('-reg_dt')
        
        processed_disclosure_list = []
        for post in disclosure_posts_qs:
            item_data = {
                'obj': post, 'display_title': post.title,
                'display_company_name': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
                'display_date': post.reg_dt, 'display_category': post.get_category_display(),
                'link': post.get_absolute_url(), 'is_api': False
            }
            if post.category == 'API공시':
                company_name, report_link = extract_api_disclosure_info(post.content)
                item_data['display_company_name'] = company_name
                item_data['link'] = report_link 
                item_data['is_api'] = True
            processed_disclosure_list.append(item_data)

        paginator_disclosure = Paginator(processed_disclosure_list, 10) 
        page_num_for_disclosure = current_page_for_active_tab if url_active_subtab == 'disclosure' else 1
        try:
            disclosure_page_obj_final = paginator_disclosure.page(page_num_for_disclosure) 
        except (EmptyPage, PageNotAnInteger):
            disclosure_page_obj_final = paginator_disclosure.page(1)
        
        context.update({
            'realtime_posts': realtime_page_obj, 
            'realtime_page_obj': realtime_page_obj, 
            'disclosures': disclosure_page_obj_final, 
            'disclosure_page_obj': disclosure_page_obj_final, 
        })
        return render(request, 'community_news.html', context)

    # tab == 'community' 인 경우의 로직
    period = request.GET.get('period', '한달')
    sort = request.GET.get('sort', '최신순')
    # '잡담' 카테고리의 게시글만 필터링 (기존 로직 유지)
    all_posts_queryset = FreeBoard.objects.filter(is_deleted=False, category='잡담').select_related('user')
    
    processed_post_list = []
    for post_obj in all_posts_queryset:
        time_diff = django_timezone.now() - post_obj.reg_dt
        days_ago = time_diff.days
        if days_ago > 0: time_ago = f"{days_ago}일 전"
        elif time_diff.seconds // 3600 > 0: time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0: time_ago = f"{time_diff.seconds // 60}분 전"
        else: time_ago = "방금 전"
        
        is_liked = False
        if request.user.is_authenticated:
            is_liked = FreeBoardLike.objects.filter(free_board=post_obj, user=request.user).exists()
            
        processed_post_list.append({
            'id': post_obj.id,
            'user': post_obj.user,
            'username': post_obj.user.nickname if hasattr(post_obj.user, 'nickname') and post_obj.user.nickname else post_obj.user.get_username(),
            'auth_id': post_obj.user.auth_id if hasattr(post_obj.user, 'auth_id') else '',
            'category': getattr(post_obj, 'category', '잡담'),
            'time_ago': time_ago,
            'days_ago': days_ago,
            'title': post_obj.title,
            'content': post_obj.content,
            'likes_count': post_obj.likes_count,
            'comments_count': post_obj.comments_count,
            'view_count': getattr(post_obj, 'view_count', 0),
            'reg_dt': post_obj.reg_dt,
            'important': getattr(post_obj, 'important', 5),
            'is_liked': is_liked,
            'get_absolute_url': post_obj.get_absolute_url() if hasattr(post_obj, 'get_absolute_url') else f"/community/{post_obj.id}/",
            'tags': post_obj.tags.all() if hasattr(post_obj, 'tags') else [],
        })

    if period == '하루':
        processed_post_list = [p for p in processed_post_list if p['days_ago'] <= 1]
    elif period == '일주일':
        processed_post_list = [p for p in processed_post_list if p['days_ago'] <= 7]
    elif period == '한달': 
        processed_post_list = [p for p in processed_post_list if p['days_ago'] <= 30]
    elif period == '반년':
        processed_post_list = [p for p in processed_post_list if p['days_ago'] <= 180]
        
    if sort == '최신순':
        processed_post_list.sort(key=lambda x: x['reg_dt'], reverse=True)
    elif sort == '인기순':
        # 조회수 높은 순, 같으면 최신순
        processed_post_list.sort(key=lambda x: (-x['view_count'], -x['reg_dt'].timestamp()))  

    paginator_community = Paginator(processed_post_list, 10)
    page_number_community = request.GET.get('page', 1)
    try:
        community_page_obj = paginator_community.page(page_number_community)
    except (EmptyPage, PageNotAnInteger):
        community_page_obj = paginator_community.page(1)

    # 커뮤니티 탭의 컨텍스트에 캐러셀용 공시 데이터를 'disclosures_for_carousel'로 전달
    context['disclosures_for_carousel'] = page_for_community_carousel

    context.update({
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
        'posts': community_page_obj,
        'page_obj': community_page_obj,
        'period': period,
        'sort': sort,
    })
    return render(request, 'community.html', context)

def write_view(request):
    """
    새 게시글 작성 뷰입니다.
    로그인된 사용자만 접근 가능하며, 캡차 검증을 포함합니다.
    """
    if not request.user.is_authenticated:
        # 로그인 페이지로 리다이렉트하면서 'next' 파라미터로 현재 경로 전달
        next_url = request.path
        if request.GET: # GET 파라미터가 있다면 함께 전달
            next_url += '?' + request.GET.urlencode()
        return HttpResponseRedirect(f"{reverse('account:login')}?next={next_url}") 

    board_type = request.GET.get('board_type', 'freeboard') # 'freeboard', 'realtime_news', 'disclosure_manual'

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip() # 사용자가 입력한 캡차 이미지의 숫자
        captcha_answer = request.POST.get('captcha_answer', '').strip() # 사용자가 입력한 캡차 정답

        logger.info(f"write_view POST: title='{title}', content_len={len(content)}, captcha_value='{captcha_value}', captcha_answer='{captcha_answer}', board_type='{board_type}'")

        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', {
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title, # 기존 입력값 유지
                'content': content, # 기존 입력값 유지
                'board_type': board_type,
            })

        # 캡차 값이 4자리 숫자인지 확인 (단순 검증 예시)
        if not captcha_value.isdigit() or len(captcha_value) != 4:
            logger.warning(f"Invalid captcha_value provided: {captcha_value}")
            messages.error(request, '잘못 입력했습니다.') # 캡차 오류 메시지는 동일하게 유지
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })

        # 캡차 답변 검증 (실제로는 세션 등에 저장된 값과 비교해야 함)
        # 여기서는 captcha_value (이미지에 표시된 값)와 captcha_answer (사용자 입력값)가 같아야 함
        if captcha_answer != captcha_value:
            logger.warning(f"Captcha mismatch: answer='{captcha_answer}', expected_value='{captcha_value}'")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })

        post_category = '잡담' # 기본값
        if board_type == 'realtime_news':
            # 실시간 뉴스는 현재 직접 작성 기능을 비활성화하거나 다른 로직을 적용할 수 있음
            messages.info(request, "실시간 뉴스는 자동 수집됩니다. 직접 작성 기능은 현재 비활성화되어 있습니다.")
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            post_category = '수동공시'

        # 'realtime_news'가 아닐 경우에만 게시물 생성
        if board_type != 'realtime_news': 
            new_post = FreeBoard.objects.create(
                user=request.user,
                title=title,
                content=content,
                category=post_category # board_type에 따라 카테고리 설정
            )
            messages.success(request, '게시물이 성공적으로 등록되었습니다.')
            logger.info(f"Post created successfully: id={new_post.id}, title='{new_post.title}', category='{post_category}'")
        
        # 게시물 유형에 따라 리다이렉트 경로 결정
        if board_type == 'realtime_news': # 이 경우는 위에서 이미 처리됨
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        else: # 'freeboard' (잡담)
            return redirect('community:community') # 커뮤니티 메인으로 리다이렉트

    # GET 요청 시 글쓰기 폼을 보여줌
    return render(request, 'community_write.html', {
        'board_type': board_type # 템플릿에서 board_type에 따라 UI 변경 가능
    })

def community_detail_view(request, post_id):
    """
    게시글 상세 페이지 뷰입니다.
    조회수 증가 로직 (세션 기반 중복 방지) 및 댓글 목록을 포함합니다.
    """
    post = get_object_or_404(FreeBoard.objects.select_related('user'), id=post_id, is_deleted=False)
    comments = FreeBoardComment.objects.filter(free_board=post, is_deleted=False).select_related('user').order_by('reg_dt')
    
    # 조회수 증가 로직 (세션 또는 IP 기반)
    # 로그인 사용자: 세션 키에 post_id 사용
    if request.user.is_authenticated:
        viewed_posts_key = f'viewed_post_{post_id}'
        if not request.session.get(viewed_posts_key, False):
            with transaction.atomic(): # 동시성 문제 방지를 위해 트랜잭션 사용
                post.view_count += 1
                post.save(update_fields=['view_count'])
                request.session[viewed_posts_key] = True
                request.session.set_expiry(86400) # 세션 만료 시간 (예: 24시간)
    else: # 비로그인 사용자: 세션 키에 post_id와 IP 주소 조합 사용
        ip_address = request.META.get('REMOTE_ADDR')
        viewed_ips_key = f'viewed_ip_{post_id}_{ip_address}'
        if not request.session.get(viewed_ips_key, False):
            with transaction.atomic():
                post.view_count += 1
                post.save(update_fields=['view_count'])
                request.session[viewed_ips_key] = True
                request.session.set_expiry(86400)

    # 시간 표시 형식 변경
    time_diff = django_timezone.now() - post.reg_dt
    if time_diff.days > 0: time_ago = f"{time_diff.days}일 전"
    elif time_diff.seconds // 3600 > 0: time_ago = f"{time_diff.seconds // 3600}시간 전"
    elif time_diff.seconds // 60 > 0: time_ago = f"{time_diff.seconds // 60}분 전"
    else: time_ago = "방금 전"
    
    is_liked = False
    if request.user.is_authenticated:
        is_liked = FreeBoardLike.objects.filter(free_board=post, user=request.user).exists()
        
    dart_link_for_detail = None
    company_name_for_detail = None
    if post.category == 'API공시':
        company_name_for_detail, dart_link_for_detail = extract_api_disclosure_info(post.content)

    post_data = {
        'id': post.id,
        'user': post.user,
        'username': post.user.nickname if hasattr(post.user, 'nickname') and post.user.nickname else post.user.get_username(),
        'auth_id': post.user.auth_id if hasattr(post.user, 'auth_id') else '',
        'title': post.title,
        'content': post.content, # 원본 내용을 전달 (템플릿에서 linebreaksbr 필터 사용 가능)
        'time_ago': time_ago,
        'reg_dt': post.reg_dt,
        'likes_count': post.likes_count,
        'comments_count': post.comments_count,
        'view_count': post.view_count,
        'is_liked': is_liked,
        'is_author': request.user == post.user, # 현재 사용자가 글쓴이인지 여부
        'category': getattr(post, 'category', '잡담'),
        'dart_link': dart_link_for_detail, # API공시일 경우 DART 링크
        'company_name_for_api_disclosure': company_name_for_detail, # API공시일 경우 회사명
    }
    return render(request, 'community_detail.html', {
        'post': post_data,
        'comments': comments,
    })

def like_post(request, post_id):
    """
    게시글 좋아요 처리 뷰입니다.
    로그인된 사용자만 가능하며, 본인 글에는 좋아요를 할 수 없습니다.
    """
    logger.info(f"like_post called for post_id={post_id}, user={request.user}")
    if not request.user.is_authenticated:
        logger.warning(f"Unauthorized like attempt for post_id={post_id}")
        messages.error(request, '로그인 후 좋아요를 누를 수 있습니다.')
        return redirect('community:detail', post_id=post_id) # 상세 페이지로 리다이렉트

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) # 삭제되지 않은 게시물만
    if post.user == request.user: # 본인 게시글 확인
        logger.warning(f"Self-like attempt by user={request.user} on post_id={post_id}")
        messages.error(request, '본인 게시글에는 좋아요를 누를 수 없습니다.')
        return redirect('community:detail', post_id=post_id)

    if request.method == 'POST': # POST 요청일 때만 처리
        with transaction.atomic(): # 데이터 일관성을 위해 트랜잭션 사용
            like, created = FreeBoardLike.objects.get_or_create(free_board=post, user=request.user)
            if created:
                post.likes_count += 1
                messages.success(request, '좋아요를 눌렀습니다.')
            else:
                like.delete()
                post.likes_count = max(0, post.likes_count - 1) # 음수 방지
                messages.success(request, '좋아요를 취소했습니다.')
            post.save(update_fields=['likes_count'])
            logger.info(f"Like {'added' if created else 'removed'} for post_id={post_id}, likes_count={post.likes_count}")
        return redirect('community:detail', post_id=post_id) # 처리 후 상세 페이지로 리다이렉트
    
    # POST 요청이 아닌 경우 (예: URL 직접 입력)
    messages.error(request, '잘못된 요청입니다.')
    return redirect('community:detail', post_id=post_id)

def comment_create(request, post_id):
    """
    댓글 작성 처리 뷰입니다.
    댓글이 작성되면 게시글 작성자에게 알림을 생성합니다.
    """
    if not request.user.is_authenticated:
        return HttpResponseRedirect(f"{reverse('account:login')}?next={reverse('community:detail', args=[post_id])}")

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if not content:
            messages.error(request, '댓글 내용을 입력해주세요.')
        else:
            with transaction.atomic():
                comment = FreeBoardComment.objects.create(
                    free_board=post,
                    user=request.user,
                    content=content,
                )
                post.comments_count += 1
                post.save(update_fields=['comments_count'])

                # 게시글 작성자에게 알림 생성 (본인 댓글이 아닌 경우)
                if post.user != request.user:
                    # 댓글 내용만 포함, 30자 초과 시 ... 추가
                    preview_content = (content[:30] + '...') if len(content) > 30 else content
                    Notification.objects.create(
                        recipient=post.user,
                        sender=request.user,
                        comment=comment,
                        message=preview_content
                    )

            messages.success(request, '댓글이 성공적으로 작성되었습니다.')
        return redirect('community:detail', post_id=post_id)
    
    return redirect('community:detail', post_id=post_id)

def edit_view(request, post_id):
    """
    게시글 수정 뷰입니다.
    게시글 작성자만 수정 가능하며, 캡차 검증을 포함합니다.
    """
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) # 삭제되지 않은 게시물만 수정 가능
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)

    # 게시물의 기존 카테고리를 board_type으로 사용 (예: '잡담', '수동공시')
    board_type = getattr(post, 'category', 'freeboard') # category 속성이 없을 경우 기본값 'freeboard'

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        captcha_value = request.POST.get('captcha_value', '').strip()
        captcha_answer = request.POST.get('captcha_answer', '').strip()

        logger.info(f"edit_view POST: post_id={post_id}, title='{title}', content_len={len(content)}, captcha_value='{captcha_value}', captcha_answer='{captcha_answer}'")

        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', { # 글쓰기 폼 재활용
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'post_id': post_id, # 수정임을 알리기 위해 post_id 전달
                'is_edit': True,    # 수정 모드임을 명시
                'board_type': board_type, # 원래 게시물의 board_type 전달
            })

        if not captcha_value.isdigit() or len(captcha_value) != 4:
            logger.warning(f"Invalid captcha_value during edit: {captcha_value}")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
            })

        if captcha_answer != captcha_value:
            logger.warning(f"Captcha mismatch during edit: answer='{captcha_answer}', value='{captcha_value}'")
            messages.error(request, '잘못 입력했습니다.')
            return render(request, 'community_write.html', {
                'error_message': '잘못 입력했습니다.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
                'board_type': board_type,
            })

        post.title = title
        post.content = content
        # 카테고리는 수정 시 변경하지 않는 것으로 가정. 필요하다면 로직 추가.
        post.save()
        messages.success(request, '게시물이 성공적으로 수정되었습니다.')
        logger.info(f"Post edited successfully: id={post.id}, title='{post.title}'")
        return redirect('community:detail', post_id=post_id)
    
    # GET 요청 시, 기존 게시물 내용으로 채워진 글쓰기 폼을 보여줌
    return render(request, 'community_write.html', {
        'title': post.title,
        'content': post.content,
        'post_id': post_id,
        'is_edit': True, # 수정 모드임을 명시
        'board_type': board_type, # 원래 게시물의 board_type 전달
    })

def delete_view(request, post_id):
    """
    게시글 삭제 뷰입니다. (논리적 삭제: is_deleted=True)
    게시글 작성자만 삭제 가능합니다.
    """
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) # 아직 삭제되지 않은 게시물 대상
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.id)

    if request.method == 'POST': # POST 요청으로만 삭제 처리 (CSRF 보호)
        post.is_deleted = True
        post.save(update_fields=['is_deleted'])
        messages.success(request, '게시물이 성공적으로 삭제되었습니다.')
        
        # 삭제 후 리다이렉트: 원래 게시물의 카테고리에 따라 분기
        original_category = getattr(post, 'category', 'freeboard')
        if original_category == '실시간뉴스': # 이 카테고리는 현재 사용되지 않을 수 있음
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif original_category == '수동공시' or original_category == 'API공시':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        # 기본적으로 커뮤니티 메인으로 리다이렉트
        return redirect('community:community') 
    
    # GET 요청 시 삭제 확인 페이지를 보여줌 (community_delete.html 템플릿 필요)
    return render(request, 'community_delete.html', {'post': post})

@login_required # 로그인 필수 데코레이터
def comment_edit(request, pk): # 댓글의 pk를 받음
    """
    댓글 수정 뷰입니다.
    댓글 작성자만 수정 가능합니다.
    """
    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        messages.error(request, "본인이 작성한 댓글만 수정할 수 있습니다.")
        return redirect('community:detail', post_id=comment.free_board.id)
    
    if request.method == "POST":
        content = request.POST.get('content', '').strip() # 수정할 내용
        if content:
            comment.content = content
            comment.save(update_fields=['content'])
            messages.success(request, "댓글이 수정되었습니다.")
        else:
            messages.error(request, "댓글 내용을 입력해 주세요.")
        return redirect('community:detail', post_id=comment.free_board.id)
    
    # POST 요청이 아니면 (예: URL 직접 접근) 상세 페이지로 리다이렉트
    # 또는 별도의 수정 폼을 제공할 수도 있음 (현재는 리다이렉트)
    return redirect('community:detail', post_id=comment.free_board.id)

@login_required # 로그인 필수 데코레이터
def comment_delete(request, pk): # 댓글의 pk를 받음
    """
    댓글 삭제 뷰입니다. (논리적 삭제: is_deleted=True)
    댓글 작성자만 삭제 가능합니다.
    """
    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        messages.error(request, "본인이 작성한 댓글만 삭제할 수 있습니다.")
        return redirect('community:detail', post_id=comment.free_board.id)
    
    if request.method == "POST": # POST 요청으로만 삭제 처리
        with transaction.atomic(): # 댓글 수 업데이트와 함께 처리
            comment.is_deleted = True
            comment.save(update_fields=['is_deleted'])
            # 게시물의 댓글 수 감소
            comment.free_board.comments_count = max(0, comment.free_board.comments_count - 1)
            comment.free_board.save(update_fields=['comments_count'])
        messages.success(request, "댓글이 삭제되었습니다.")
    
    return redirect('community:detail', post_id=comment.free_board.id)
