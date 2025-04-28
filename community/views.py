from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect

def community_view(request):
    # 로그인 여부 확인
    is_authenticated = 'user_id' in request.session

    context = {
        'community_menus': [
            {'name': '커뮤니티'},
            {'name': '뉴스'},
            {'name': '종목'},
            {'name': '예측'},
            {'name': '공지'},
        ],
        'ticker_message': '예측 정보 티커 영역 예시: 비트코인 1억 돌파 예측 중!',
        'posts': [
            {
                'username': '코알라3B2',
                'category': '코인한마디',
                'time_ago': '5분 전',
                'title': 'LOOM, ARK, HIFI',
                'content': '비트는 현재 1.21억입니다. 오늘은 2자리 상승률을 가진 코인이 2개 있습니다...',
                'likes': 3,
                'comments': 0,
            },
            {
                'username': '코인불장기원',
                'category': '잡담',
                'time_ago': '10분 전',
                'title': '오늘 코인 시장 분위기 어떰?',
                'content': '다들 오늘 장 어때요? 흐름이 좋아 보이긴 하는데...',
                'likes': 7,
                'comments': 2,
            },
        ],
        'is_authenticated': is_authenticated,
    }
    return render(request, 'community.html', context)

def write_view(request):
    # 로그인 여부 확인
    if 'user_id' not in request.session:
        return HttpResponseRedirect('/account/login/?next=/community/write/')
    
    if request.method == 'POST':
        # TODO: 게시글 저장 로직 (모델 필요)
        return redirect('community:community')
    
    return render(request, 'community_write.html')