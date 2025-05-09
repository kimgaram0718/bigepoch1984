# community/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from .models import FreeBoard, FreeBoardComment, FreeBoardLike, DartDisclosure, NewsArticle # NewsArticle 임포트
from django.utils import timezone as django_timezone
from django.contrib import messages
from django.db import transaction
import logging
from datetime import datetime, timedelta
from django.db.models import Q
import re 

from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

logger = logging.getLogger(__name__)

def extract_api_disclosure_info(content_str):
    company_name = "정보 없음"; report_link = "#"
    company_match = re.search(r"회사명: (.*?)\n", content_str)
    if company_match: company_name = company_match.group(1).strip()
    link_match = re.search(r"공시 원문 보기: (https?://[^\s]+)", content_str)
    if link_match: report_link = link_match.group(1).strip()
    return company_name, report_link

def community_view(request):
    # print(f"\n--- [VIEWS.PY DEBUG] community_view 함수 호출됨 ---") # 디버깅용
    tab = request.GET.get('tab', 'community')
    subtab = request.GET.get('subtab', '') 
    # print(f"--- [VIEWS.PY DEBUG] 요청 파라미터: tab='{tab}', subtab='{subtab}' ---") # 디버깅용

    context = {
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
        'active_tab': tab,
        'active_subtab': subtab if subtab else ('realtime' if tab == 'news' else ''),
    }

    if tab == 'news':
        # print(f"--- [VIEWS.PY DEBUG] 뉴스 탭 처리 시작 (URL subtab: '{subtab}') ---") # 디버깅용
        
        # URL의 'page' 파라미터는 현재 활성화된 subtab에만 적용됩니다.
        # 다른 subtab은 기본적으로 1페이지를 로드합니다.
        page_param_from_url = request.GET.get('page', '1')
        try:
            current_page_for_active_tab = int(page_param_from_url)
        except ValueError:
            current_page_for_active_tab = 1

        url_active_subtab = subtab if subtab else 'realtime' # URL 기준으로 활성화된 서브탭

        # 1. 실시간 뉴스 데이터 항상 준비
        # print("--- [VIEWS.PY DEBUG] '실시간 뉴스' 데이터 준비 중 ---") # 디버깅용
        news_article_list = NewsArticle.objects.all().order_by('-pub_date')
        paginator_realtime = Paginator(news_article_list, 10)
        # '실시간 뉴스' 탭이 URL에서 활성화된 경우 해당 페이지 번호 사용, 아니면 1페이지
        page_num_for_realtime = current_page_for_active_tab if url_active_subtab == 'realtime' else 1
        try:
            realtime_page_obj = paginator_realtime.page(page_num_for_realtime)
        except (EmptyPage, PageNotAnInteger):
            realtime_page_obj = paginator_realtime.page(1)
        # print(f"--- [VIEWS.PY DEBUG] '실시간 뉴스' Page 객체: {realtime_page_obj}, 항목 수: {len(realtime_page_obj.object_list) if realtime_page_obj else 0} (요청 페이지: {page_num_for_realtime}) ---") # 디버깅용

        # 2. 거래소 공시 데이터 항상 준비
        # print("--- [VIEWS.PY DEBUG] '거래소 공시' 데이터 준비 중 ---") # 디버깅용
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
        # '거래소 공시' 탭이 URL에서 활성화된 경우 해당 페이지 번호 사용, 아니면 1페이지
        page_num_for_disclosure = current_page_for_active_tab if url_active_subtab == 'disclosure' else 1
        try:
            disclosure_page_obj_final = paginator_disclosure.page(page_num_for_disclosure) 
        except (EmptyPage, PageNotAnInteger):
            disclosure_page_obj_final = paginator_disclosure.page(1)
        
        # print(f"--- [VIEWS.PY DEBUG] '거래소 공시' Page 객체: {disclosure_page_obj_final}, 항목 수: {len(disclosure_page_obj_final.object_list) if disclosure_page_obj_final else 0} (요청 페이지: {page_num_for_disclosure}) ---") # 디버깅용
        
        context.update({
            'realtime_posts': realtime_page_obj, 
            'realtime_page_obj': realtime_page_obj, 
            'disclosures': disclosure_page_obj_final, 
            'disclosure_page_obj': disclosure_page_obj_final, 
        })
        # print(f"--- [VIEWS.PY DEBUG] community_news.html 렌더링 준비 완료. Context 'realtime_posts' 항목 수: {len(context['realtime_posts'].object_list)}, Context 'disclosures' 항목 수: {len(context['disclosures'].object_list)} ---") # 디버깅용
        return render(request, 'community_news.html', context)

    # 기존 커뮤니티 탭 로직
    # ... (이하 코드는 이전과 동일하게 유지) ...
    period = request.GET.get('period', '한달')
    sort = request.GET.get('sort', '최신순')
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
        processed_post_list.sort(key=lambda x: x['likes_count'], reverse=True)

    paginator_community = Paginator(processed_post_list, 10)
    page_number_community = request.GET.get('page', 1)
    try:
        community_page_obj = paginator_community.page(page_number_community)
    except (EmptyPage, PageNotAnInteger):
        community_page_obj = paginator_community.page(1)

    context.update({
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
        'posts': community_page_obj,
        'page_obj': community_page_obj,
        'period': period,
        'sort': sort,
    })
    # print("--- [DEBUG] community.html 렌더링 시도 ---") # 디버깅용
    return render(request, 'community.html', context)

# write_view, community_detail_view 등 나머지 뷰 함수는 이전 버전과 동일하게 유지
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
        
        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', {
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'board_type': board_type,
            })
        
        post_category = '잡담' 
        if board_type == 'realtime_news':
            messages.info(request, "실시간 뉴스는 자동 수집됩니다. 직접 작성 기능은 현재 비활성화되어 있습니다.")
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            post_category = '수동공시'

        if board_type != 'realtime_news': 
            new_post = FreeBoard.objects.create(
                user=request.user,
                title=title,
                content=content,
                category=post_category 
            )
            messages.success(request, '게시물이 성공적으로 등록되었습니다.')
        
        if board_type == 'realtime_news':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif board_type == 'disclosure_manual':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=disclosure")
        else: 
            return redirect('community:community')

    return render(request, 'community_write.html', {
        'board_type': board_type 
    })

def community_detail_view(request, post_id):
    post = get_object_or_404(FreeBoard.objects.select_related('user'), id=post_id, is_deleted=False)
    comments = FreeBoardComment.objects.filter(free_board=post, is_deleted=False).select_related('user').order_by('reg_dt')
    
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
        'content': post.content, 
        'time_ago': time_ago,
        'reg_dt': post.reg_dt,
        'likes_count': post.likes_count,
        'comments_count': post.comments_count,
        'view_count': getattr(post, 'view_count', 0),
        'is_liked': is_liked,
        'is_author': request.user == post.user,
        'category': getattr(post, 'category', '잡담'),
        'dart_link': dart_link_for_detail, 
        'company_name_for_api_disclosure': company_name_for_detail, 
    }
    return render(request, 'community_detail.html', {
        'post': post_data,
        'comments': comments,
    })

# like_post, comment_create, edit_view, delete_view, comment_edit, comment_delete 함수는 이전과 동일하게 유지
def like_post(request, post_id):
    logger.info(f"like_post called for post_id={post_id}, user={request.user}")
    if not request.user.is_authenticated:
        logger.warning(f"Unauthorized like attempt for post_id={post_id}")
        messages.error(request, '로그인 후 좋아요를 누를 수 있습니다.')
        return redirect('community:detail', post_id=post_id) 

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) 
    if post.user == request.user:
        logger.warning(f"Self-like attempt by user={request.user} on post_id={post_id}")
        messages.error(request, '본인 게시글에는 좋아요를 누를 수 없습니다.')
        return redirect('community:detail', post_id=post_id)

    if request.method == 'POST':
        with transaction.atomic():
            like, created = FreeBoardLike.objects.get_or_create(free_board=post, user=request.user)
            if created:
                post.likes_count += 1
                messages.success(request, '좋아요를 눌렀습니다.')
            else:
                like.delete()
                post.likes_count = max(0, post.likes_count - 1) 
                messages.success(request, '좋아요를 취소했습니다.')
            post.save(update_fields=['likes_count'])
            logger.info(f"Like {'added' if created else 'removed'} for post_id={post_id}, likes_count={post.likes_count}")
        return redirect('community:detail', post_id=post_id)
    
    messages.error(request, '잘못된 요청입니다.')
    return redirect('community:detail', post_id=post_id)


def comment_create(request, post_id):
    if not request.user.is_authenticated:
        return HttpResponseRedirect(f"{reverse('account:login')}?next={reverse('community:detail', args=[post_id])}")

    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) 
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if not content:
            messages.error(request, '댓글 내용을 입력해주세요.')
        else:
            with transaction.atomic():
                FreeBoardComment.objects.create(
                    free_board=post,
                    user=request.user,
                    content=content,
                )
                post.comments_count += 1
                post.save(update_fields=['comments_count'])
            messages.success(request, '댓글이 성공적으로 작성되었습니다.')
        return redirect('community:detail', post_id=post_id) 
    
    return redirect('community:detail', post_id=post_id)


def edit_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) 
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)

    board_type = getattr(post, 'category', 'freeboard') 

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        if not title or not content:
            messages.error(request, '제목과 내용을 모두 입력해주세요.')
            return render(request, 'community_write.html', { 
                'error_message': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'post_id': post_id, 
                'is_edit': True,
                'board_type': board_type,
            })
        post.title = title
        post.content = content
        post.save()
        messages.success(request, '게시물이 성공적으로 수정되었습니다.')
        return redirect('community:detail', post_id=post_id)
    
    return render(request, 'community_write.html', {
        'title': post.title,
        'content': post.content,
        'post_id': post_id,
        'is_edit': True,
        'board_type': board_type, 
    })

def delete_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False) 
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.id)

    if request.method == 'POST': 
        post.is_deleted = True
        post.save(update_fields=['is_deleted'])
        messages.success(request, '게시물이 성공적으로 삭제되었습니다.')
        
        original_category = getattr(post, 'category', 'freeboard')
        if original_category == '실시간뉴스':
            return redirect(f"{reverse('community:community')}?tab=news&subtab=realtime")
        elif original_category == '수동공시' or original_category == 'API공시':
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
            comment.save(update_fields=['content'])
            messages.success(request, "댓글이 수정되었습니다.")
        else:
            messages.error(request, "댓글 내용을 입력해 주세요.")
        return redirect('community:detail', post_id=comment.free_board.id)
    
    return redirect('community:detail', post_id=comment.free_board.id)


@login_required
def comment_delete(request, pk): 
    comment = get_object_or_404(FreeBoardComment, pk=pk, is_deleted=False)
    if request.user != comment.user:
        messages.error(request, "본인이 작성한 댓글만 삭제할 수 있습니다.")
        return redirect('community:detail', post_id=comment.free_board.id)
    
    if request.method == "POST": 
        with transaction.atomic():
            comment.is_deleted = True
            comment.save(update_fields=['is_deleted'])
            comment.free_board.comments_count = max(0, comment.free_board.comments_count - 1)
            comment.free_board.save(update_fields=['comments_count'])
        messages.success(request, "댓글이 삭제되었습니다.")
    
    return redirect('community:detail', post_id=comment.free_board.id)
