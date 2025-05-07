// script.js

// HTML 문서가 완전히 로드되고 파싱되었을 때 실행될 함수들을 등록합니다.
document.addEventListener('DOMContentLoaded', () => {
  // 1. fetch('./layout/header.html') 부분 제거
  // Django에서는 main_header.html이 직접 포함되므로, JS로 헤더를 로드하지 않습니다.

  // 2. setHeaderContent() 함수 호출 또는 헤더 관련 이벤트 리스너 직접 등록
  // setHeaderContent 함수는 더 이상 헤더의 전체 HTML을 만들지 않고,
  // 이미 존재하는 요소(main_header.html에 의해 렌더링된)에 이벤트 리스너를 추가하는 역할 등을 합니다.
  initializeHeaderFeatures(); // 함수 이름을 변경하거나 setHeaderContent를 아래와 같이 수정

  // Footer 관련 로직 (기존 코드 유지)
  fetchFooter(); // 함수명 변경 예시

  // 페이지 로딩 시 초기 필터값 표시 (기존 코드 유지)
  if (typeof toggleMainTab === 'function' && document.getElementById('btn-main-popular')) { // 함수 및 요소 존재 여부 확인
      toggleMainTab('popular');
  }

  // 스크롤 최상단 이동 버튼 (기존 코드 유지)
  initializeScrollToTopButton(); // 함수명 변경 예시

  // 배너 (기존 코드 유지)
  initializeBanners(); // 함수명 변경 예시

  // 실시간 인기 검색 (기존 코드 유지)
  initializeRealtimeSearch(); // 함수명 변경 예시

  // Originals 리스트 (기존 코드 유지)
  if (typeof renderOriginalsList === 'function' && document.getElementById('originals-list')) {
      renderOriginalsList();
  }

  // NOW Headlines (기존 코드 유지)
  initializeHeadlines(); // 함수명 변경 예시

  // Investing Insight (기존 코드 유지)
  if (typeof renderCoinInfoList === 'function' && document.getElementById('coin-info-list')) {
      renderCoinInfoList();
  }

  // 공지사항 (기존 코드 유지)
  if (typeof renderNoticeList === 'function' && document.getElementById('notice-list')) {
      renderNoticeList();
  }

  // 거래소 리스트 (기존 코드 유지)
  initializeExchangeList(); // 함수명 변경 예시

  // 코인 시세 조회 (기존 코드 유지)
  if (typeof renderCoinList === 'function' && document.getElementById('coin-list')) {
      renderCoinList();
  }

  // 글쓰기 버튼 (기존 코드 유지)
  initializeWriteButton(); // 함수명 변경 예시
});

// 헤더 기능 초기화 함수 (기존 setHeaderContent 역할 일부 대체 및 수정)
function initializeHeaderFeatures() {
  // main_header.html에 의해 헤더가 이미 렌더링되었다고 가정합니다.
  // #header-content div는 Django 템플릿에 의해 채워져 있을 것입니다.

  // 팝업 및 패널 관련 이벤트 리스너 등록
  const alertIcon = document.getElementById('alertIcon');
  const profileIcon = document.getElementById('profileIcon');
  const mainPanelButton = document.getElementById('mainPanelIcon'); // 헤더의 햄버거 버튼 ID

  // 로그인 상태는 Django 템플릿에 의해 #profileIcon 등이 렌더링 되었는지로 판단 가능
  const isLoggedIn = !!profileIcon; // profileIcon이 있으면 로그인된 것으로 간주 (main_header_html_updated 기준)

  if (isLoggedIn) {
      // 로그인된 경우의 아이콘 이벤트 리스너
      if (alertIcon) {
          alertIcon.addEventListener('click', (e) => {
              e.stopPropagation();
              togglePopup('alert-popup');
          });
      }

      if (profileIcon) {
          profileIcon.addEventListener('click', (e) => {
              e.stopPropagation();
              setupAndToggleProfilePopup(); // 프로필 팝업 내용 설정 및 토글
          });
      }
       // 팝업 HTML (alert-popup, profile-popup)은 main.html 또는 base.html에 미리 정의되어 있어야 합니다.
       // 또는 여기서 동적으로 생성할 수 있지만, Django URL 및 static 처리에 주의해야 합니다.
       // 예시: ensurePopupsExist(); // 팝업 HTML이 없으면 동적으로 추가하는 함수
  } else {
      // 비로그인 상태의 헤더에는 Django 템플릿에 의해 '로그인' 버튼이 이미 존재하고 링크도 설정되어 있을 것입니다.
      // JavaScript로 추가적인 로그인 버튼 클릭 로직이 필요하다면 여기에 작성합니다.
      // 예: const loginButton = document.querySelector('a[href="{% url \'account:login\' %}"] button');
      // if (loginButton) { /* 이벤트 리스너 추가 */ }
  }

  // 햄버거 메뉴 (메인 패널) 버튼 이벤트 리스너
  if (mainPanelButton) {
      mainPanelButton.addEventListener('click', (event) => {
          event.stopPropagation();
          openLoginPanel(); // 로그인 패널 열기 함수 호출
      });
  } else {
      console.error('Error: mainPanelIcon element not found in the HTML.');
  }

  // 공통: 팝업 외부 클릭 시 닫기 (팝업이 여러 개일 경우 주의해서 관리)
  document.addEventListener('click', (event) => {
      // 현재 열려있는 팝업들을 식별하여, 클릭된 대상이 해당 팝업 외부인지 확인 후 닫습니다.
      // 예를 들어, data 속성 등으로 현재 열린 팝업을 표시하고 관리할 수 있습니다.
      const activePopups = document.querySelectorAll('.popup-box:not(.d-none)');
      activePopups.forEach(popup => {
          // 아이콘 클릭으로 팝업이 열리는 경우, 해당 아이콘을 클릭했을 때는 이 로직으로 닫히면 안됨.
          // 각 아이콘 클릭 핸들러에서 e.stopPropagation()을 사용하는 것이 중요.
          if (!popup.contains(event.target)) {
               // 클릭된 대상이 팝업을 여는 아이콘 자체도 아닌지 확인해야 함
               let clickedOnToggleButton = false;
               if (popup.id === 'alert-popup' && alertIcon && alertIcon.contains(event.target)) clickedOnToggleButton = true;
               if (popup.id === 'profile-popup' && profileIcon && profileIcon.contains(event.target)) clickedOnToggleButton = true;

               if(!clickedOnToggleButton) {
                  popup.classList.add('d-none');
               }
          }
      });
  });
}

function setupAndToggleProfilePopup() {
  let profilePopup = document.getElementById('profile-popup');
  if (!profilePopup) {
      // 프로필 _팝업_HTML_생성_및_삽입 (Django URL 및 static 경로 주의)
      // 이 부분은 Django 템플릿에 미리 구조를 만들어두는 것이 더 좋습니다.
      const userData = JSON.parse(sessionStorage.getItem('user')); // sessionStorage는 Django 인증과 별개임.
                                                                // Django 인증 정보를 사용해야 함.
      const nickname = userData?.nickname || "{{ request.user.nickname|default:'사용자' }}"; // Django 템플릿 값 사용 (HTML data 속성에서 가져와야 함)

      const popupHtml = `
      <div id="profile-popup" class="popup-box d-none" style="
          position: absolute; top: 60px; right: 20px; width: 200px;
          background: white; border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          z-index: 9999; font-size: 14px;">
          <div class="px-3 py-2 d-flex align-items-center">
              <i class="bi bi-person-circle me-2" style="font-size: 20px; color: #9376e0;"></i>
              <span class="fw-normal text-dark" style="font-weight: 500;">${nickname}</span>
          </div>
          <ul class="list-unstyled m-0">
              <li class="px-3 py-2 hover-bg text-muted" onclick="location.href='profile.html'">마이페이지 이동</li> {/* Django URL 사용 */}
              <li class="px-3 py-2 hover-bg text-muted" onclick="location.href='space.html'">스페이스 이동</li> {/* Django URL 사용 */}
              <li class="px-3 py-2 hover-bg text-danger" onclick="performLogout()">로그아웃</li> {/* Django 로그아웃 URL로 이동 */}
          </ul>
      </div>`;
      document.body.insertAdjacentHTML('beforeend', popupHtml);
      profilePopup = document.getElementById('profile-popup');
  }

  if (profilePopup) {
      // 닉네임 등 동적 내용 업데이트 (필요시)
      // const nicknameSpan = profilePopup.querySelector('.fw-normal.text-dark');
      // if (nicknameSpan) nicknameSpan.textContent = getCurrentUserNickname(); // 현재 사용자 닉네임 가져오는 함수

      profilePopup.classList.toggle('d-none');
  }
}

function performLogout() {
  // Django 로그아웃 URL로 이동하거나, form submit 방식으로 로그아웃 처리
  window.location.href = "{% url 'account:logout' %}"; // Django URL name에 맞게 수정
}


function togglePopup(id) {
  const popup = document.getElementById(id);
  if (popup) {
      popup.classList.toggle('d-none');
  }
}

function closePopup(id) { // 이 함수는 document 클릭 리스너에서 직접 사용되거나, 특정 팝업 닫기 로직에 사용될 수 있음
  const popup = document.getElementById(id);
  if (popup && !popup.classList.contains('d-none')) {
      popup.classList.add('d-none');
  }
}

// 로그인 패널 열기 (기존 함수 구조 유지, 내부 로직은 Django 상황에 맞게 조정)
function openLoginPanel() {
  const panel = document.getElementById('login-panel');
  if (panel) {
      console.log('Login panel found. Attempting to show.');
      setupLoginPanelContent(); // 패널 내용 설정
      panel.classList.add('show');
  } else {
      console.error('Error: login-panel element not found!');
      return;
  }
  document.body.classList.add('no-scroll');
  setTimeout(() => {
      document.addEventListener('click', handleOutsideClickForPanel, { once: true });
  }, 0);
}

// 로그인 패널 외부 클릭 시 닫기 (기존 함수 구조 유지)
function handleOutsideClickForPanel(e) {
  const panel = document.getElementById('login-panel');
  const mainPanelButton = document.getElementById('mainPanelIcon');
  if (panel && !panel.contains(e.target) && mainPanelButton && !mainPanelButton.contains(e.target)) {
      closeLoginPanel();
  } else if (panel && panel.classList.contains('show')) {
      // 패널이 열려있고, 클릭된 부분이 패널 내부가 아니거나 햄버거 버튼이 아닌 경우,
      // 다음 외부 클릭을 감지하도록 리스너를 다시 등록할 수 있습니다.
      // 현재는 once: true로 한번 실행 후 자동 제거되므로, 패널이 다시 열릴 때 새로 등록됩니다.
  }
}

// 로그인 패널 닫기 (기존 함수 구조 유지)
function closeLoginPanel() {
  const panel = document.getElementById('login-panel');
  if (panel) {
      panel.classList.remove('show');
  }
  document.body.classList.remove('no-scroll');
}

// 로그인 패널 내용 설정 (기존 함수 구조 유지, 내부 HTML 생성 시 Django 고려)
function setupLoginPanelContent() {
  const inner = document.getElementById('login-panel-inner');
  if (!inner) {
      console.error('Error: login-panel-inner element not found!');
      return;
  }

  // Django 템플릿에서 전달된 로그인 상태 사용
  // 예: <body data-is-authenticated="{{ request.user.is_authenticated|yesno:'true,false' }}"
  //          data-user-nickname="{{ request.user.nickname|escapejs|default:'' }}"
  //          data-login-url="{% url 'account:login' %}"
  //          data-logout-url="{% url 'account:logout' %}"
  //          data-rocket-img="{% static 'images/rocket.png' %}">
  const bodyData = document.body.dataset;
  const isAuthenticated = bodyData.isAuthenticated === 'true';
  const userNickname = bodyData.userNickname || '사용자';
  const loginUrl = bodyData.loginUrl || '#'; // 기본값 또는 오류 처리
  const logoutUrl = bodyData.logoutUrl || '#';
  const rocketImgSrc = bodyData.rocketImg || './images/rocket.png'; // 기본값

  if (isAuthenticated) {
      const recentPostHtml = (typeof recentPosts !== 'undefined' ? recentPosts : []).map(post => `
          <li class="d-flex justify-content-between align-items-center recent-view-item" onclick="location.href='${post.link}'">
              <span class="text-truncate title" style="max-width: 80%;">${post.title}</span>
              <span class="badge bg-light text-dark">${post.tag}</span>
          </li>
      `).join('');

      // Django URL을 사용하도록 onclick 핸들러 수정
      inner.innerHTML = `
          <div class="login-menu-wrapper">
              <div class="login-menu-header p-3 border-bottom">
                  <h5 class="mb-0">안녕하세요, ${userNickname}님!</h5>
              </div>
              <div class="login-menu-body">
                  <div class="login-menu-grid">
                      <div class="menu-item" onclick="location.href='alert.html'"> {/* TODO: Django URL로 변경 */}
                          <i class="bi bi-bell"></i>
                          <span>알림</span>
                      </div>
                      <div class="menu-item profile-icon" onclick="location.href='profile.html'"> {/* TODO: Django URL로 변경 */}
                          <i class="bi bi-person-circle" style="color: #9376e0;"></i>
                          <span>프로필</span>
                      </div>
                      <div class="menu-item" onclick="location.href='settings.html'"> {/* TODO: Django URL로 변경 */}
                          <i class="bi bi-gear"></i>
                          <span>설정</span>
                      </div>
                  </div>
                  <div class="mt-4 px-3">
                      <div class="fw-bold mb-2">최근 본 게시글</div>
                      <ul class="list-unstyled small recent-view-list">
                          ${recentPostHtml}
                      </ul>
                      <div class="fw-bold mt-4 mb-2">참여 스페이스</div>
                      <div class="text-muted small">관심 스페이스가 없습니다.</div>
                      <div class="text-primary mt-2" style="cursor:pointer;">더 보기</div>
                  </div>
              </div>
              <div class="login-menu-footer p-3 border-top">
                  <button class="btn btn-outline-secondary w-100" onclick="location.href='${logoutUrl}'">로그아웃</button>
              </div>
          </div>
      `;
  } else {
      // 비로그인 사용자를 위한 패널 내용
      inner.innerHTML = `
          <div class="text-center mt-5 p-3">
              <img src="${rocketImgSrc}" alt="로켓" style="width: 60px;" />
              <p class="mt-3 fw-bold">더 많은 기능을 위해<br />로그인하세요.</p>
              <button class="btn btn-primary mt-3 px-4 w-100" onclick="location.href='${loginUrl}'">로그인</button>
          </div>
      `;
  }
}

// moveToLogin 함수는 이제 setupLoginPanelContent 내부에서 직접 URL을 사용하므로 별도로 필요 없을 수 있습니다.
// 만약 다른 곳에서 사용된다면 유지합니다.
function moveToLogin() { // 이 함수는 이제 loginUrl을 직접 사용하므로, 전역 변수나 data 속성에서 가져와야 함
  const loginUrl = document.body.dataset.loginUrl || 'login.html'; // 기본값
  sessionStorage.setItem('prevPage', window.location.pathname);
  window.location.href = loginUrl;
}

function performLogout() { // 이 함수는 이제 logoutUrl을 직접 사용하므로, 전역 변수나 data 속성에서 가져와야 함
  const logoutUrl = document.body.dataset.logoutUrl || '#'; // 기본값
  // sessionStorage.removeItem('user'); // Django 세션은 서버에서 관리됨. 클라이언트에서 직접 제거 불가.
  closeLoginPanel();
  window.location.href = logoutUrl;
}


// --- 나머지 기존 함수들은 여기에 그대로 유지 ---
// 예: fetchFooter, initializeScrollToTopButton, initializeBanners, etc.

function fetchFooter() {
  // fetch('./layout/footer.html') // Django에서는 include 사용
  //  .then(response => response.text())
  //  .then(data => {
  //    const footerContainer = document.getElementById('footer-container');
  //    if (footerContainer) { // footerContainer가 항상 존재한다고 가정하지 않도록 주의
  //      footerContainer.innerHTML = data;
  //      // ... (기존 footerItems 로직) ...
  //    }
  //  });
  // Django 템플릿에서 footer가 이미 포함되어 있다면, 아래 로직만 실행
  const footerContainer = document.getElementById('footer-container');
  if (footerContainer) {
      const footerItems = footerContainer.querySelectorAll('.footer-item');
      if (footerItems.length > 0) {
          const currentPath = window.location.pathname.split('/').pop() || 'main'; // 기본값 설정 또는 index.html 등
          footerItems.forEach(item => {
              const href = item.getAttribute('href');
              // href가 null이 아니고, currentPath와 일치하는지 확인
              if (href && (href === currentPath || (currentPath === 'main' && (href === '/' || href === 'main.html')))) { // 기본 경로 처리 추가
                  footerItems.forEach(i => i.classList.remove('active'));
                  item.classList.add('active');
              }
          });

          footerItems.forEach(item => {
              item.addEventListener('click', (e) => {
                  // 페이지 이동은 a 태그의 기본 동작에 맡기거나,
                  // SPA 방식이라면 여기서 e.preventDefault() 후 라우팅 처리
                  footerItems.forEach(i => i.classList.remove('active'));
                  item.classList.add('active');
              });
          });
      }
  }
}


function initializeScrollToTopButton() {
  const scrollTopBtn = document.getElementById('scrollTopBtn');
  if (!scrollTopBtn) return;

  window.addEventListener('scroll', () => {
      if (window.scrollY > 80) {
          scrollTopBtn.style.display = 'flex'; // d-flex로 되어있을 수 있으므로 block 대신 flex
      } else {
          scrollTopBtn.style.display = 'none';
      }
  });

  scrollTopBtn.addEventListener('click', () => {
      window.scrollTo({
          top: 0,
          behavior: 'smooth'
      });
  });
}

const banners = [ // 이 데이터는 외부에서 오거나 Django 템플릿에서 JS 변수로 전달될 수 있음
  { link: "https://example.com/banner1", img: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=800&q=80", alt: "배너1" },
  { link: "https://example.com/banner2", img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80", alt: "배너2" },
  { link: "https://example.com/banner3", img: "https://images.unsplash.com/photo-1593642634367-d91a135587b5?auto=format&fit=crop&w=800&q=80", alt: "배너3" }
];

function initializeBanners() {
  const carouselInner = document.getElementById('carousel-inner');
  const bannerCountDiv = document.getElementById('carousel-count');
  const carouselElement = document.getElementById('mainBannerCarousel');

  if (!carouselInner || !carouselElement || !bannerCountDiv) return;

  banners.forEach((banner, index) => {
      const itemDiv = document.createElement('div');
      itemDiv.className = `carousel-item ${index === 0 ? 'active' : ''}`;
      itemDiv.innerHTML = `
          <a href="${banner.link}" target="_blank">
              <img src="${banner.img}" class="d-block w-100" alt="${banner.alt}" style="max-height: 200px; object-fit: cover; border-radius: 0.5rem;">
          </a>
      `; // 스타일 약간 추가
      carouselInner.appendChild(itemDiv);
  });

  if (banners.length > 0) {
      bannerCountDiv.textContent = `1 / ${banners.length}`; // 초기 카운트 설정
      const carousel = new bootstrap.Carousel(carouselElement);
      carouselElement.addEventListener('slide.bs.carousel', (e) => {
          const current = e.to + 1;
          const total = banners.length;
          bannerCountDiv.textContent = `${current} / ${total}`;
      });
  } else {
       bannerCountDiv.textContent = `0 / 0`;
  }
}

const realtimeItems = [ // 이 데이터는 외부에서 오거나 Django 템플릿에서 JS 변수로 전달될 수 있음
  { rank: 1, title: "밀크", link: "milk.html" }, { rank: 2, title: "cobak-token", link: "cobak-token.html" },
  { rank: 3, title: "bitcoin", link: "bitcoin.html" }, { rank: 4, title: "ethereum", link: "ethereum.html" },
  { rank: 5, title: "파일 암호화폐", link: "filecoin.html" }, { rank: 6, title: "리플", link: "ripple.html" },
  { rank: 7, title: "도지코인", link: "dogecoin.html" }, { rank: 8, title: "pump", link: "pump.html" },
  { rank: 9, title: "퀀텀", link: "quantum.html" }, { rank: 10, title: "메타플래닛", link: "metaplanet.html" }
];

function initializeRealtimeSearch() {
  const realtimeToggle = document.getElementById('realtimeToggle');
  const realtimeDropdown = document.getElementById('realtimeDropdown');
  const realtimeList = document.getElementById('realtimeList');
  const realtimeCloseButton = document.getElementById('realtimeClose');

  if (!realtimeToggle || !realtimeDropdown || !realtimeList || !realtimeCloseButton) return;

  realtimeItems.forEach(item => {
      const li = document.createElement('li');
      li.className = "mb-2";
      // Django URL 사용 예시: const url = `/search/${item.title}/`;
      li.innerHTML = `<a href="${item.link}" class="text-primary text-decoration-none">${item.rank}. ${item.title}</a>`;
      realtimeList.appendChild(li);
  });

  realtimeToggle.addEventListener('click', toggleRealtime); // window 객체에 등록 대신 직접 연결
  realtimeCloseButton.addEventListener('click', () => {
      realtimeDropdown.classList.add('d-none');
      realtimeToggle.classList.remove('d-none');
  });
}

function toggleRealtime() { // 이 함수는 이제 initializeRealtimeSearch 내부에서 호출되거나 직접 연결됨
  const realtimeDropdown = document.getElementById('realtimeDropdown');
  const realtimeToggle = document.getElementById('realtimeToggle');
  if (!realtimeDropdown || !realtimeToggle) return;

  if (realtimeDropdown.classList.contains('d-none')) {
      realtimeDropdown.classList.remove('d-none');
      realtimeToggle.classList.add('d-none');
  } else {
      realtimeDropdown.classList.add('d-none');
      realtimeToggle.classList.remove('d-none');
  }
}

const originalsData = [  /* ... 기존 데이터 ... */ ];
function renderOriginalsList() { /* ... 기존 함수 ... */ }

const headlinesData = { /* ... 기존 데이터 ... */ };
function renderHeadlines(type) { /* ... 기존 함수 ... */ }
function switchTab(active) { /* ... 기존 함수 ... */ }
function initializeHeadlines() {
  const btnLatest = document.getElementById("btn-latest");
  const btnPopular = document.getElementById("btn-popular");
  const btnRising = document.getElementById("btn-rising");
  const btnFalling = document.getElementById("btn-falling");
  const btnMarket = document.getElementById('btn-view-market');
  const btnCuration = document.getElementById('btn-view-curation');

  if(btnLatest) renderHeadlines("latest"); // 초기 로드

  if(btnLatest) btnLatest.addEventListener("click", () => { renderHeadlines("latest"); switchTab("latest"); });
  if(btnPopular) btnPopular.addEventListener("click", () => { renderHeadlines("popular"); switchTab("popular"); });
  if(btnRising) btnRising.addEventListener("click", () => { renderHeadlines("rising"); switchTab("rising"); });
  if(btnFalling) btnFalling.addEventListener("click", () => { renderHeadlines("falling"); switchTab("falling"); });

  if(btnMarket && btnCuration) {
      const tabBasic = document.getElementById('tab-basic');
      const tabCuration = document.getElementById('tab-curation');

      btnMarket.addEventListener('click', () => {
          if(btnMarket.classList.contains('active')) return;
          btnMarket.classList.add('active'); btnCuration.classList.remove('active');
          btnMarket.classList.remove('bg-light', 'text-muted'); btnMarket.classList.add('bg-white', 'text-dark');
          btnCuration.classList.remove('bg-white', 'text-dark'); btnCuration.classList.add('bg-light', 'text-muted');
          if(tabBasic) tabBasic.classList.remove('d-none');
          if(tabCuration) tabCuration.classList.add('d-none');
          renderHeadlines('latest'); switchTab('latest');
      });
      btnCuration.addEventListener('click', () => {
          if(btnCuration.classList.contains('active')) return;
          btnCuration.classList.add('active'); btnMarket.classList.remove('active');
          btnCuration.classList.remove('bg-light', 'text-muted'); btnCuration.classList.add('bg-white', 'text-dark');
          btnMarket.classList.remove('bg-white', 'text-dark'); btnMarket.classList.add('bg-light', 'text-muted');
          if(tabCuration) tabCuration.classList.remove('d-none');
          if(tabBasic) tabBasic.classList.add('d-none');
          renderHeadlines('rising'); switchTab('rising');
      });
       // 초기 상태 설정 (예: '시장 관점' 활성화)
      if(btnMarket) btnMarket.classList.add('active'); else if(btnCuration) btnCuration.classList.add('active');

  }
}


const filterData = { /* ... 기존 데이터 ... */ };
function renderFilteredList(type) { /* ... 기존 함수 ... */ }
function toggleMainTab(type) { /* ... 기존 함수 ... */ }
// toggleMainTab 호출은 DOMContentLoaded에서 이미 처리됨

const coinInfoData = [ /* ... 기존 데이터 ... */ ];
function renderCoinInfoList() { /* ... 기존 함수 ... */ }

const noticeListData = [ /* ... 기존 데이터 ... */ ];
function renderNoticeList() { /* ... 기존 함수 ... */ }

const exchangeData = [ /* ... 기존 데이터 ... */ ];
const sampleExchangeLogo = "https://placehold.co/100x100/cccccc/000000?text=EX"; // 수정된 플레이스홀더
function renderExchangeList() {
  const container = document.getElementById('exchange-list');
  if (!container) { console.error('exchange-list 요소를 찾을 수 없습니다.'); return; }
  container.innerHTML = '';
  [...exchangeData, ...exchangeData].forEach(exchange => { // 무한 스크롤 효과를 위해 데이터 두번 반복
      const a = document.createElement('a');
      a.href = exchange.link; a.target = '_blank'; a.className = 'exchange-item';
      a.innerHTML = `<img src="${sampleExchangeLogo}" alt="${exchange.name}"><span>${exchange.name}</span>`;
      container.appendChild(a);
  });
}
let exchangeScrollInterval;
function initializeExchangeList() {
  const exchangeListEl = document.querySelector('.exchange-list');
  if (!exchangeListEl) return;
  renderExchangeList(); // 먼저 렌더링

  if (exchangeScrollInterval) clearInterval(exchangeScrollInterval); // 기존 인터벌 클리어

  // 스크롤 가능할 때만 인터벌 설정
  // clientWidth가 0이거나 scrollWidth가 clientWidth보다 작거나 같으면 스크롤 불필요
  if (exchangeListEl.clientWidth > 0 && exchangeListEl.scrollWidth > exchangeListEl.clientWidth) {
      let scrollAmount = 0;
      const scrollStep = 73;
      const scrollDelay = 2000;
      exchangeScrollInterval = setInterval(() => {
          scrollAmount += scrollStep;
          if (scrollAmount >= exchangeListEl.scrollWidth / 2) { // 절반 지점에서 초기화 (데이터 두번 반복했으므로)
              exchangeListEl.scrollTo({ left: 0, behavior: 'auto' }); // 부드럽지 않게 바로 이동
              scrollAmount = scrollStep; // 다음 스크롤을 위해 초기화
          }
          exchangeListEl.scrollTo({ left: scrollAmount, behavior: 'smooth' });
      }, scrollDelay);
  }
}


const coinData = [ /* ... 기존 데이터 ... */ ];
let currentPage = 1; const itemsPerPage = 10; let filteredCoins = [...coinData];
function renderCoinList() { /* ... 기존 함수 (내부 이미지 플레이스홀더 수정) ... */
  const list = document.getElementById('coin-list');
  if(!list) return;
  list.innerHTML = '';
  const start = (currentPage - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const coinsToShow = filteredCoins.slice(start, end);
  coinsToShow.forEach(coin => {
      const li = document.createElement('li');
      li.className = 'd-flex align-items-center px-2 py-2';
      li.style.cursor = 'pointer';
      li.onclick = () => window.location.href = coin.link; // Django URL 사용 고려
      li.innerHTML = `
          <div class="d-flex align-items-center flex-grow-1" style="min-width: 0;">
              <img src="https://placehold.co/24x24/cccccc/000000?text=C" alt="coin" style="width: 24px; height: 24px; object-fit: cover; border-radius: 50%; margin-right: 8px;">
              <span class="text-truncate" style="font-size: 14px;">${coin.name}</span>
          </div>
          <div style="width: 100px; text-align: right; font-size: 14px; ${coin.change >= 0 ? 'color:red;' : 'color:blue;'}">
              ${coin.price.toLocaleString()}
          </div>
          <div style="width: 80px; text-align: right; font-size: 14px; ${coin.change >= 0 ? 'color:red;' : 'color:blue;'}">
              ${coin.change > 0 ? '+' : ''}${coin.change}%
          </div>
      `;
      list.appendChild(li);
  });
  updatePagination();
}
function updatePagination() { /* ... 기존 함수 ... */ }
// 검색 및 페이지 이동 이벤트 리스너는 DOMContentLoaded 밖에서도 괜찮으나, 요소가 로드된 후여야 함.
// DOMContentLoaded 내에서 호출되는 renderCoinList()가 최초 실행을 담당.
// 버튼 이벤트 리스너는 DOMContentLoaded 내에서 등록하는 것이 안전.
document.addEventListener('DOMContentLoaded', () => {
  const coinSearchInput = document.getElementById('coinSearch');
  const prevPageBtn = document.getElementById('prevPage');
  const nextPageBtn = document.getElementById('nextPage');

  if(coinSearchInput) {
      coinSearchInput.addEventListener('input', (e) => {
          const keyword = e.target.value.trim().toLowerCase();
          filteredCoins = coinData.filter(coin => coin.name.toLowerCase().includes(keyword));
          currentPage = 1;
          renderCoinList();
      });
  }
  if(prevPageBtn) {
      prevPageBtn.addEventListener('click', () => {
          if (currentPage > 1) { currentPage--; renderCoinList(); }
      });
  }
  if(nextPageBtn) {
      nextPageBtn.addEventListener('click', () => {
          const totalPages = Math.ceil(filteredCoins.length / itemsPerPage);
          if (currentPage < totalPages) { currentPage++; renderCoinList(); }
      });
  }
  if(document.getElementById('coin-list')) renderCoinList(); // 최초 로딩
});


const recentPosts = [ /* ... 기존 데이터 ... */ ]; // 이 데이터는 Django 템플릿에서 전달받는 것이 좋음

function initializeWriteButton() {
  // Django 템플릿에서 로그인 상태를 data 속성으로 전달받는다고 가정
  // 예: <body data-is-authenticated="{{ request.user.is_authenticated|yesno:'true,false' }}"
  //          data-write-url="{% url 'your_write_page_url_name' %}">
  const bodyData = document.body.dataset;
  const isLoggedIn = bodyData.isAuthenticated === 'true';
  const writeUrl = bodyData.writeUrl || 'write.html'; // 기본값

  if (!isLoggedIn) return;

  const writeBtn = document.createElement('button');
  writeBtn.id = 'goToWriteBtn';
  writeBtn.className = 'btn btn-primary rounded-circle d-flex align-items-center justify-content-center';
  writeBtn.style.cssText = `
      position: fixed; bottom: 150px; right: 15px; z-index: 1000;
      width: 45px; height: 45px; font-size: 22px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: flex;`; // 로그인 시 항상 보이도록 display: flex
  writeBtn.innerHTML = '<i class="bi bi-pencil" style="font-size: 17px;"></i>';
  document.body.appendChild(writeBtn);

  writeBtn.addEventListener('click', () => {
      window.location.href = writeUrl;
  });
}
