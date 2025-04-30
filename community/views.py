from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
from .models import FreeBoard
from django.utils import timezone
from django.contrib import messages

def community_view(request):
    # FreeBoard에서 삭제되지 않은 게시물 가져오기
    posts = FreeBoard.objects.filter(is_deleted=False).select_related('user')

    # 게시물 데이터를 템플릿에 맞게 변환
    post_list = []
    for post in posts:
        # 시간 차이 계산 (예: "5분 전")
        time_diff = timezone.now() - post.reg_dt
        if time_diff.days > 0:
            time_ago = f"{time_diff.days}일 전"
        elif time_diff.seconds // 3600 > 0:
            time_ago = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds // 60 > 0:
            time_ago = f"{time_diff.seconds // 60}분 전"
        else:
            time_ago = "방금 전"

        post_list.append({
            'id': post.free_board_id,
            'username': post.user.nickname,
            'category': '잡담',
            'time_ago': time_ago,
            'title': post.title,
            'content': post.content,
            'likes': 0,
            'comments': 0,
        })

    context = {
        'community_menus': [
            {'name': '커뮤니티'},
            {'name': '뉴스'},
            {'name': '종목'},
            {'name': '예측'},
            {'name': '공지'},
        ],
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
        'posts': post_list,
    }
    return render(request, 'community.html', context)

def write_view(request):
    if not request.user.is_authenticated:
        return HttpResponseRedirect('/account/login/?next=/community/write/')
    
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        
        # 입력값 검증
        if not title or not content:
            return render(request, 'community_write.html', {
                'error': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
            })
        
        # 게시물 저장
        FreeBoard.objects.create(
            user=request.user,
            title=title,
            content=content,
        )
        messages.success(request, '게시물이 성공적으로 등록되었습니다.')
        return redirect('community:community')
    
    return render(request, 'community_write.html')

def community_detail_view(request, post_id):
    # 게시물 조회
    post = get_object_or_404(FreeBoard, free_board_id=post_id, is_deleted=False)

    # 시간 차이 계산
    time_diff = timezone.now() - post.reg_dt
    if time_diff.days > 0:
        time_ago = f"{time_diff.days}일 전"
    elif time_diff.seconds // 3600 > 0:
        time_ago = f"{time_diff.seconds // 3600}시간 전"
    elif time_diff.seconds // 60 > 0:
        time_ago = f"{time_diff.seconds // 60}분 전"
    else:
        time_ago = "방금 전"

    # 템플릿에 전달할 데이터
    post_data = {
        'id': post.free_board_id,  # 올바르게 free_board_id 사용
        'user': post.user,
        'title': post.title,
        'content': post.content,
        'time_ago': time_ago,
        'likes': 0,
        'wows': 0,
        'sads': 0,
    }

    return render(request, 'community_detail.html', {'post': post_data})

def edit_view(request, post_id):
    # 게시물 조회
    post = get_object_or_404(FreeBoard, free_board_id=post_id, is_deleted=False)

    # 권한 검증: 작성자 본인만 수정 가능
    if request.user != post.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('community:detail', post_id=post_id)

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()

        # 입력값 검증
        if not title or not content:
            return render(request, 'community_write.html', {
                'error': '제목과 내용을 모두 입력해주세요.',
                'title': title,
                'content': content,
                'post_id': post_id,
                'is_edit': True,
            })

        # 게시물 업데이트
        post.title = title
        post.content = content
        post.save()

        messages.success(request, '게시물이 성공적으로 수정되었습니다.')
        return redirect('community:detail', post_id=post_id)

    # GET 요청: 수정 페이지 표시
    return render(request, 'community_write.html', {
        'title': post.title,
        'content': post.content,
        'post_id': post_id,
        'is_edit': True,
    })

def delete_view(request, post_id):
    # 게시물 조회
    post = get_object_or_404(FreeBoard, free_board_id=post_id, is_deleted=False)

    # 권한 검증: 작성자 본인만 삭제 가능
    if request.user != post.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('community:detail', post_id=post.free_board_id)

    if request.method == 'POST':
        # 논리적 삭제: is_deleted를 True로 변경
        post.is_deleted = True
        post.save()

        messages.success(request, '게시물이 성공적으로 삭제되었습니다.')
        return redirect('community:community')

    # GET 요청: 삭제 확인 페이지 표시
    return render(request, 'community_delete.html', {
        'post': post,
    })