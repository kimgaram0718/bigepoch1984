from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
from .models import FreeBoard, FreeBoardComment, FreeBoardLike
from django.utils import timezone
from django.contrib import messages
from django.db import transaction
import logging

logger = logging.getLogger(__name__)

def community_view(request):
    period = request.GET.get('period', '한달')
    sort = request.GET.get('sort', '최신순')
    posts = FreeBoard.objects.filter(is_deleted=False).select_related('user')
    post_list = []
    for post in posts:
        time_diff = timezone.now() - post.reg_dt
        days_ago = time_diff.days
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"
        else:
            time_ago = "방금 전"
        is_liked = False
        if request.user.is_authenticated:
            is_liked = FreeBoardLike.objects.filter(free_board=post, user=request.user).exists()
        post_list.append({
            'id': post.id,
            'username': post.user.nickname,
            'auth_id': post.user.auth_id,
            'category': '잡담',
            'time_ago': time_ago,
            'days_ago': days_ago,
            'title': post.title,
            'content': post.content,
            'likes_count': post.likes_count,
            'comments_count': post.comments_count,
            'important': 5,
            'is_liked': is_liked,
        })
    if period == '하루':
        post_list = [post for post in post_list if post['days_ago'] <= 1]
    elif period == '일주일':
        post_list = [post for post in post_list if post['days_ago'] <= 7]
    elif period == '한달':
        post_list = [post for post in post_list if post['days_ago'] <= 30]
    elif period == '반년':
        post_list = [post for post in post_list if post['days_ago'] <= 180]
    if sort == '최신순':
        post_list.sort(key=lambda x: x['days_ago'])
    elif sort == '인기순':
        post_list.sort(key=lambda x: x['likes_count'], reverse=True)
    elif sort == '중요순':
        post_list.sort(key=lambda x: x['important'], reverse=True)
    elif sort == '걱정순':
        post_list.sort(key=lambda x: x['comments_count'], reverse=True)
    context = {
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
        'posts': post_list,
        'period': period,
        'sort': sort,
    }
    return render(request, 'community.html', context)

def news_view(request):
    context = {
        'community_menus': [{'name': '커뮤니티'}, {'name': '뉴스'}, {'name': '종목'}, {'name': '예측'}, {'name': '공지'}],
    }
    return render(request, 'news.html', context)

def write_view(request):
    if not request.user.is_authenticated:
        return HttpResponseRedirect('/account/login/?next=/community/write/')
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        if not title or not content:
            return render(request, 'community_write.html', {
                'error': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
            })
        FreeBoard.objects.create(
            user=request.user,
            title=title,
            content=content,
        )
        messages.success(request, '게시물이 성공적으로 등록되었습니다.')
        return redirect('community:community')
    return render(request, 'community_write.html')

def community_detail_view(request, post_id):
    post = get_object_or_404(FreeBoard.objects.select_related('user'), id=post_id, is_deleted=False)
    comments = FreeBoardComment.objects.filter(free_board=post, is_deleted=False).select_related('user')
    time_diff = timezone.now() - post.reg_dt
    if time_diff.days > 0:
        time_ago = f"{time_diff.days}일 전"
    elif time_diff.seconds // 3600 > 0:
        time_ago = f"{time_diff.seconds // 3600}시간 전"
    elif time_diff.seconds // 60 > 0:
        time_ago = f"{time_diff.seconds // 60}분 전"
    else:
        time_ago = "방금 전"
    is_liked = False
    if request.user.is_authenticated:
        is_liked = FreeBoardLike.objects.filter(free_board=post, user=request.user).exists()
    post_data = {
        'id': post.id,
        'username': post.user.nickname,
        'auth_id': post.user.auth_id,
        'title': post.title,
        'content': post.content,
        'time_ago': time_ago,
        'likes_count': post.likes_count,
        'comments_count': post.comments_count,
        'is_liked': is_liked,
    }
    return render(request, 'community_detail.html', {
        'post': post_data,
        'comments': comments,
        'messages': messages.get_messages(request),  # 메시지 컨텍스트
    })

def like_post(request, post_id):
    logger.info(f"like_post called for post_id={post_id}, user={request.user}")
    if not request.user.is_authenticated:
        logger.warning(f"Unauthorized like attempt for post_id={post_id}")
        messages.error(request, '로그인 후 좋아요를 누를 수 있습니다.')
        return redirect('community:detail', post_id=post_id)
    post = get_object_or_404(FreeBoard.objects.select_related('user'), id=post_id, is_deleted=False)
    if post.user == request.user:
        logger.warning(f"Self-like attempt by user={request.user} on post_id={post_id}")
        messages.error(request, '본인 게시글에는 좋아요를 누를 수 없습니다.')
        return redirect('community:detail', post_id=post_id)
    if request.method == 'POST':
        with transaction.atomic():
            like, created = FreeBoardLike.objects.get_or_create(free_board=post, user=request.user)
            if created:
                post.likes_count += 1
                post.save()
                logger.info(f"Like added for post_id={post_id}, likes_count={post.likes_count}")
                messages.success(request, '좋아요를 눌렀습니다.')
            else:
                like.delete()
                post.likes_count = max(0, post.likes_count - 1)
                post.save()
                logger.info(f"Like removed for post_id={post_id}, likes_count={post.likes_count}")
                messages.success(request, '좋아요를 취소했습니다.')
        return redirect('community:detail', post_id=post_id)
    messages.error(request, '잘못된 요청입니다.')
    return redirect('community:detail', post_id=post_id)

def comment_create(request, post_id):
    if not request.user.is_authenticated:
        return HttpResponseRedirect(f'/account/login/?next=/community/{post_id}/')
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        if not content:
            messages.error(request, '댓글 내용을 입력해주세요.')
            return redirect('community:detail', post_id=post_id)
        with transaction.atomic():
            FreeBoardComment.objects.create(
                free_board=post,
                user=request.user,
                content=content,
            )
            post.comments_count += 1
            post.save()
        messages.success(request, '댓글이 성공적으로 작성되었습니다.')
        return redirect('community:detail', post_id=post_id)
    return redirect('community:detail', post_id=post_id)

def edit_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        if not title or not content:
            return render(request, 'community_write.html', {
                'error': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
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
    })

def delete_view(request, post_id):
    post = get_object_or_404(FreeBoard, id=post_id, is_deleted=False)
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.id)
    if request.method == 'POST':
        post.is_deleted = True
        post.save()
        messages.success(request, '게시물이 성공적으로 삭제되었습니다.')
        return redirect('community:community')
    return render(request, 'community_delete.html', {'post': post})