document.addEventListener('DOMContentLoaded', () => {
  // 로그인 여부 판단 (서버에서 넘긴 값 기준)
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
  const openGreetingInputBtn = document.getElementById('openGreetingInputBtn');
  const greetingInputContainer = document.getElementById('greetingInputContainer');
  const cancelGreetingInputBtn = document.getElementById('cancelGreetingInputBtn');
  const confirmGreetingBtn = document.getElementById('confirmGreetingBtn');
  const scrollTopBtn = document.getElementById('scrollTopBtn');

  const contentMap = {
    '마이페이지': 'content-mypage',
    '프로필': 'content-profile',
    '내 스페이스': 'content-space',
    '계정 및 보안': 'content-security'
  };

  // 디버깅 로그
  console.log('Elements:', {
    greetingInputContainer,
    openGreetingInputBtn,
    cancelGreetingInputBtn,
    confirmGreetingBtn
  });

  // 드롭다운 토글
  dropdownBtn?.addEventListener('click', () => {
    dropdownMenu.style.display =
      dropdownMenu.style.display === 'none' || dropdownMenu.style.display === ''
        ? 'block'
        : 'none';
  });

  // 메뉴 클릭 시 콘텐츠 전환
  dropdownMenu?.querySelectorAll('.dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const label = item.getAttribute('data-label');
      currentLabel.textContent = label;
      dropdownMenu.style.display = 'none';

      Object.values(contentMap).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
      });

      const selected = contentMap[label];
      if (selected) {
        const el = document.getElementById(selected);
        if (el) el.style.display = 'block';
      }
    });
  });

  // 외부 클릭 시 드롭다운 닫기
  document.addEventListener('click', (e) => {
    if (!document.getElementById('mypageDropdown')?.contains(e.target)) {
      dropdownMenu.style.display = 'none';
    }
  });

  // 닉네임 출력
  const userData = sessionStorage.getItem('user');
  if (userData && nicknameEl) {
    try {
      const user = JSON.parse(userData);
      nicknameEl.textContent = user.nickname || '사용자';
    } catch (e) {
      console.warn('닉네임 파싱 실패:', e);
    }
  }

  // 프로필 보기 버튼 → 전환
  profileViewBtn?.addEventListener('click', () => {
    Object.values(contentMap).forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = 'none';
    });
    const profileEl = document.getElementById(contentMap['프로필']);
    if (profileEl) profileEl.style.display = 'block';
    currentLabel.textContent = '프로필';
  });

  // 스크롤 버튼 동작
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

  // 인사 메시지 입력창 열기
  openGreetingInputBtn?.addEventListener('click', () => {
    if (greetingInputContainer) {
      console.log('Opening greeting input');
      greetingInputContainer.classList.add('show');
    } else {
      console.error('Greeting input container not found');
    }
  });

  // 인사 메시지 입력창 닫기 (취소 버튼)
  cancelGreetingInputBtn?.addEventListener('click', () => {
    if (greetingInputContainer) {
      greetingInputContainer.classList.remove('show');
    }
  });

  // 인사 메시지 저장 (확인 버튼)
  confirmGreetingBtn?.addEventListener('click', async () => {
    const greetingInput = document.getElementById('greetingInput')?.value || '';
    const url = openGreetingInputBtn?.dataset.url || '{% url "mypage:update_greeting_message" %}';
    const token = openGreetingInputBtn?.dataset.csrf || '{{ csrf_token }}';

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-CSRFToken': token,
        },
        body: new URLSearchParams({
          'greeting_message': greetingInput,
        }),
      });

      const data = await response.json();
      if (data.success) {
        alert(data.message);
        if (greetingInputContainer) {
          greetingInputContainer.classList.remove('show');
          location.reload(); // 페이지 새로고침으로 메시지 반영
        }
      } else {
        alert(data.message);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('서버와의 통신 중 오류가 발생했습니다.');
    }
  });
});