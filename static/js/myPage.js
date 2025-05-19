document.addEventListener('DOMContentLoaded', () => {
    const isAuthenticated = document.body.dataset.isAuthenticated === 'true';

    if (!isAuthenticated) {
        sessionStorage.setItem('prevPage', 'mypage');
        window.location.href = '/account/login/';
        return;
    }

    const dropdownBtn = document.getElementById('dropdownMenuBtn');
    const dropdownMenu = document.querySelector('#mypageDropdown .dropdown-menu');
    const currentLabel = document.getElementById('currentMenuLabel');
    const profileViewBtn = document.querySelector('.profile-view-btn');
    const nicknameEl = document.getElementById('nickname');
    const scrollTopBtn = document.getElementById('scrollTopBtn');

    const contentMap = {
        '마이페이지': 'content-mypage',
        '예측 종목': 'content-profile',
        '내가 쓴 글': 'content-space',
        '차단 계정': 'content-security'
    };

    dropdownBtn?.addEventListener('click', () => {
        dropdownMenu.style.display =
            dropdownMenu.style.display === 'none' || dropdownMenu.style.display === ''
                ? 'block'
                : 'none';
        console.log('Dropdown toggled:', dropdownMenu.style.display);
    });

    dropdownMenu?.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const label = item.getAttribute('data-label');
            console.log('Dropdown item clicked:', label);
            currentLabel.textContent = label;
            dropdownMenu.style.display = 'none';

            Object.values(contentMap).forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });

            const selected = contentMap[label];
            if (selected) {
                const el = document.getElementById(selected);
                if (el) {
                    el.style.display = 'block';
                    console.log('Content displayed:', selected);
                } else {
                    console.error('Element not found:', selected);
                }
            }
        });
    });

    document.addEventListener('click', (e) => {
        if (!document.getElementById('mypageDropdown')?.contains(e.target)) {
            dropdownMenu.style.display = 'none';
        }
    });

    const userData = sessionStorage.getItem('user');
    if (userData && nicknameEl) {
        try {
            const user = JSON.parse(userData);
            nicknameEl.textContent = user.nickname || '사용자';
        } catch (e) {
            console.warn('닉네임 파싱 실패:', e);
        }
    }

    profileViewBtn?.addEventListener('click', () => {
        Object.values(contentMap).forEach(id => {
            const el = document.getElementById(id);
            if (el) el.style.display = 'none';
        });
        const profileEl = document.getElementById(contentMap['예측 종목']);
        if (profileEl) profileEl.style.display = 'block';
        currentLabel.textContent = '예측 종목';
    });

    window.addEventListener('scroll', () => {
        if (window.scrollY > 200) {
            scrollTopBtn.style.display = 'flex';
        } else {
            scrollTopBtn.style.display = 'none';
        }
    });

    scrollTopBtn?.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    //add1
    document.querySelectorAll('.unblock-btn').forEach(function(btn) {
    btn.disabled = false;
    btn.addEventListener('click', function() {
      const blockedId = btn.getAttribute('data-blocked-id');
      if (!blockedId) return;
      if (!confirm('정말로 차단을 해제하시겠습니까?')) return;
      fetch('/community/unblock_user/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': getCookie('csrftoken')
        },
        body: 'blocked_id=' + encodeURIComponent(blockedId)
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          btn.closest('.blocked-user-card').remove();
        } else {
          alert(data.error || '차단 해제에 실패했습니다.');
        }
      });
    });
  });

  // URL에서 tab 파라미터 읽기
  const params = new URLSearchParams(window.location.search);
  const tab = params.get('tab');
  if (tab === 'security') {
    document.querySelectorAll('.content-block').forEach(el => el.style.display = 'none');
    document.getElementById('content-security').style.display = '';
    const label = document.getElementById('currentMenuLabel');
    if (label) label.textContent = '차단 계정';
  } else if (tab === 'space') {
    document.querySelectorAll('.content-block').forEach(el => el.style.display = 'none');
    document.getElementById('content-space').style.display = '';
    const label = document.getElementById('currentMenuLabel');
    if (label) label.textContent = '내가 쓴 글';
  }
  //add2
});