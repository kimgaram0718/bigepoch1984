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

  const contentMap = {
    '마이페이지': 'content-mypage',
    '프로필': 'content-profile',
    '내 스페이스': 'content-space',
    '계정 및 보안': 'content-security'
  };

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
      e.preventDefault();  // ✅ 이게 있어야 href="#" 이동 방지됨
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
  const scrollTopBtn = document.getElementById('scrollTopBtn');
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
});
