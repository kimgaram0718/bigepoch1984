// header.html 가져오기
fetch('./layout/header.html')
  .then(response => response.text())
  .then(data => {
    const headerContainer = document.getElementById('header-container');
    if (headerContainer) {
      headerContainer.innerHTML = data;
      setHeaderContent();
    }
  });

  function setHeaderContent() {
    const headerContent = document.getElementById('header-content');
    if (!headerContent) return;
  
    if (window.location.pathname.includes('community.html')) {
      // 커뮤니티 화면
      headerContent.innerHTML = `
    <div class="d-flex align-items-center justify-content-between w-100">
      <div class="bg-light rounded-pill p-1 d-flex align-items-center" style="gap: 8px;">
        <button id="btn-community" class="btn btn-sm fw-bold text-dark bg-white rounded-pill px-3 py-1">커뮤니티</button>
        <button id="btn-news" class="btn btn-sm fw-bold text-muted bg-transparent rounded-pill px-3 py-1">뉴스</button>
      </div>
      <div class="d-flex gap-2 align-items-center">
        <i class="bi bi-search" style="font-size: 20px;"></i>
        <i class="bi bi-list" style="font-size: 24px;"></i>
      </div>
    </div>
  `;
  
      // 토글 버튼 클릭 이벤트
      const btnCommunity = document.getElementById('btn-community');
      const btnNews = document.getElementById('btn-news');
  
      btnCommunity.addEventListener('click', () => {
        btnCommunity.classList.remove('text-muted');
        btnCommunity.classList.add('text-dark');
        btnNews.classList.remove('text-dark');
        btnNews.classList.add('text-muted');
      });
  
      btnNews.addEventListener('click', () => {
        btnCommunity.classList.remove('text-dark');
        btnCommunity.classList.add('text-muted');
        btnNews.classList.remove('text-muted');
        btnNews.classList.add('text-dark');
      });
  
    } else {
      // 홈 화면 (또는 다른 기본 화면)
      headerContent.innerHTML = `
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png" alt="로고" style="width:32px;height:32px; border-radius:50%;">
        <div class="d-flex align-items-center gap-3">
          <i class="bi bi-search" style="font-size: 24px; font-weight: bold; color: #1b1e26;"></i>
          <button type="button" class="btn p-0 m-0" style="background: none; border: none; font-size: 16px; font-weight: 700; color: #1b1e26;">로그인</button>
          <div class="dropdown">
            <button class="btn lang-btn dropdown-toggle" type="button" id="langMenu" data-bs-toggle="dropdown" aria-expanded="false">KR</button>
            <ul class="dropdown-menu dropdown-lang dropdown-menu-end" aria-labelledby="langMenu">
              <li><a class="dropdown-item" href="#">KR</a></li>
              <li><a class="dropdown-item" href="#">EN</a></li>
            </ul>
          </div>
        </div>
      `;
    }
  }

// footer.html 불러오기
fetch('./layout/footer.html')
  .then(response => response.text())
  .then(data => {
    const footerContainer = document.getElementById('footer-container');
    if (footerContainer) {
      footerContainer.innerHTML = data;

      const footerItems = footerContainer.querySelectorAll('.footer-item');

      // 🔥 여기 추가
      const currentPath = window.location.pathname.split('/').pop(); // ex) community.html
      footerItems.forEach(item => {
        const href = item.getAttribute('href');
        if (href === currentPath) {
          footerItems.forEach(i => i.classList.remove('active')); // 다 지우고
          item.classList.add('active'); // 현재 경로에 맞는 것만 active
        }
      });

      // 원래 있던 클릭 이벤트 (필요하면 유지)
      footerItems.forEach(item => {
        item.addEventListener('click', (e) => {
          footerItems.forEach(i => i.classList.remove('active'));
          item.classList.add('active');
        });
      });
    }
  });

// 페이지가 로딩되자마자 초기 필터값 표시
window.addEventListener('DOMContentLoaded', () => {
  toggleMainTab('popular');
});

// 스크롤 최상단 이동 버튼 기능
const scrollTopBtn = document.getElementById('scrollTopBtn');

// 스크롤 내릴 때 버튼 보이기
window.addEventListener('scroll', () => {
  if (window.scrollY > 80) {
    scrollTopBtn.style.display = 'block';
  } else {
    scrollTopBtn.style.display = 'none';
  }
});

// 버튼 클릭하면 맨 위로 부드럽게 이동
scrollTopBtn.addEventListener('click', () => {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
});

const banners = [
    {
      link: "https://example.com/banner1",
      img: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=800&q=80",
      alt: "배너1"
    },
    {
      link: "https://example.com/banner2",
      img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80",
      alt: "배너2"
    },
    {
      link: "https://example.com/banner3",
      img: "https://images.unsplash.com/photo-1593642634367-d91a135587b5?auto=format&fit=crop&w=800&q=80",
      alt: "배너3"
    }
  ];
  
  // carousel-inner 채우기
  const carouselInner = document.getElementById('carousel-inner');
  
  banners.forEach((banner, index) => {
    const itemDiv = document.createElement('div');
    itemDiv.className = `carousel-item ${index === 0 ? 'active' : ''}`;
    itemDiv.innerHTML = `
      <a href="${banner.link}" target="_blank">
        <img src="${banner.img}" class="d-block w-100" alt="${banner.alt}">
      </a>
    `;
    carouselInner.appendChild(itemDiv);
  });
  
  // 숫자 카운트 갱신
  const bannerCountDiv = document.getElementById('carousel-count');
  const carouselElement = document.getElementById('mainBannerCarousel');
  const carousel = new bootstrap.Carousel(carouselElement);
  
  carouselElement.addEventListener('slide.bs.carousel', (e) => {
    const current = e.to + 1;
    const total = banners.length;
    bannerCountDiv.textContent = `${current} / ${total}`;
  });


  /* 실시간 인기검색 로직 */
  // 실시간/인기 검색어 데이터 가정
  const realtimeToggle = document.getElementById('realtimeToggle');
  const realtimeDropdown = document.getElementById('realtimeDropdown');
  const realtimeArrow = document.getElementById('realtimeArrow');
  const realtimeText = document.getElementById('realtimeText');
  const realtimeList = document.getElementById('realtimeList');
  
  const realtimeItems = [
    { rank: 1, title: "밀크", link: "milk.html" },
    { rank: 2, title: "cobak-token", link: "cobak-token.html" },
    { rank: 3, title: "bitcoin", link: "bitcoin.html" },
    { rank: 4, title: "ethereum", link: "ethereum.html" },
    { rank: 5, title: "파일 암호화폐", link: "filecoin.html" },
    { rank: 6, title: "리플", link: "ripple.html" },
    { rank: 7, title: "도지코인", link: "dogecoin.html" },
    { rank: 8, title: "pump", link: "pump.html" },
    { rank: 9, title: "퀀텀", link: "quantum.html" },
    { rank: 10, title: "메타플래닛", link: "metaplanet.html" }
  ];
  
  // 리스트 채우기
  realtimeItems.forEach(item => {
    const li = document.createElement('li');
    li.className = "mb-2";
    li.innerHTML = `<a href="${item.link}" class="text-primary text-decoration-none">${item.rank}. ${item.title}</a>`;
    realtimeList.appendChild(li);
  });
  
  // 토글 동작
  function toggleRealtime() {
    const dropdown = document.getElementById('realtimeDropdown');
    const toggleBox = document.getElementById('realtimeToggle');
  
    if (dropdown.classList.contains('d-none')) {
      dropdown.classList.remove('d-none');
      toggleBox.classList.add('d-none');
    } else {
      dropdown.classList.add('d-none');
      toggleBox.classList.remove('d-none');
    }
  }
  
  // **함수를 window 객체에 등록해줘야 해**
  window.toggleRealtime = toggleRealtime;
  
  // 그리고 닫기 버튼 이벤트 연결
  document.getElementById('realtimeClose').addEventListener('click', () => {
    document.getElementById('realtimeDropdown').classList.add('d-none');
    document.getElementById('realtimeToggle').classList.remove('d-none');
  });
   /* 실시간 인기검색 로직 끝 */


   const originalsData = [
    {
      title: "RomanHodl",
      description: "한국거래소 빗썸, IPO 앞두고 리스크 완화 위해 '빗썸A' 출시",
      img: "https://images.unsplash.com/photo-1556740749-887f6717d7e4?crop=entropy&cs=tinysrgb&fit=crop&w=400&h=300"
      //  사람 손+노트북 보이는 오픈 이미지 (절대 안깨짐)
    },
    {
      title: "CryptoCaster",
      description: "젊고 부유하며 암호화폐에 투자하는 사람들: 한국 엘리트",
      img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80"
      // 금화, 코인 이미지 (절대 안깨짐)
    },
    {
      title: "알트코인",
      description: "현재 이더리움 가격 흐름을 빠르게 체크 해 봅시다",
      img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80"
      // 금화, 코인 이미지 (절대 안깨짐)
    },
    {
      title: "테스트입니다 고쳐요",
      description: "2025.04.25 나스닥 이슈 및 지수 분석",
      img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80"
      // 금화, 코인 이미지 (절대 안깨짐)
    },
    {
      title: "테스트요",
      description: "이더리움은 정말 끝인가...?",
      img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?auto=format&fit=crop&w=800&q=80"
      // 금화, 코인 이미지 (절대 안깨짐)
    }
  ];
  
  // 카드 리스트 그리기
  function renderOriginalsList() {
    const container = document.getElementById('originals-list');
    container.innerHTML = '';
  
    originalsData.forEach((item) => {
      const card = document.createElement('div');
      card.className = "flex-shrink-0";
      card.style = "width: 150px;";
  
      card.innerHTML = `
        <img src="${item.img}" alt="${item.title}" class="rounded-3 mb-2" style="width: 100%; height: 100px; object-fit: cover;">
        <div class="text-muted" style="font-size: 12px;">${item.title}</div>
        <div class="fw-bold" style="font-size: 14px;">${item.description}</div>
      `;
  
      container.appendChild(card);
    });
  }
  
  // 최초 로딩
  renderOriginalsList();

 /* NOW Headlines 섹션 시작*/
  // 뉴스 데이터
  const headlinesData = {
    latest: [
      { title: "시진핑, 기후 영상 정상회의에서 연설 - 신화통신.", link: "news1.html" },
      { title: "베트남: 미국과 무역 협상을 시작했습니다. - 국영 언론.", link: "news2.html" },
      { title: "구글·아마존은 넘었다…비트코인, 세계 자산 5위 등극!", link: "news3.html" },
      { title: "벤치마크, 코인베이스 목표 주가 $252 책정", link: "news4.html" },
      { title: "모스크바시 비즈니스 센터에서 폭발음 3회 발생", link: "news5.html" }
    ],
    popular: [
      { title: "XRP, 60% 급등 가능성 열리나…비트코인 강세", link: "news6.html" },
      { title: "트럼프 '파월 해고안해'…달러·비트코인 오르고 금값 하락", link: "news7.html" },
      { title: "비트코인, 미·중 무역 협상 진전 신호에 급등", link: "news8.html" },
      { title: "XRP, 디지털 금 대체 가능성 부상", link: "news9.html" },
      { title: "24시간 암호화폐 무기한 선물 강제청산", link: "news10.html" }
    ],
    rising: [
      { title: "비트코인 6주 만에 9만 달러 돌파", link: "rising1.html" },
      { title: "XRP·SOL까지 강세장 진입", link: "rising2.html" },
      { title: "MVRV 지표 유지 시 최대 80% 상승 가능", link: "rising3.html" },
      { title: "비트코인, 무역 협상 진전 소식에 급등", link: "rising4.html" },
      { title: "4월 23일 암호화폐 아침 뉴스", link: "rising5.html" }
    ],
    falling: [
      { title: "게이트아이오 트래픽 급증으로 선물 거래 중단", link: "fall1.html" },
      { title: "현물 금, 2% 하락", link: "fall2.html" },
      { title: "칠리즈, 미국 재진출 준비 중", link: "fall3.html" },
      { title: "ETH 숏 포지션 증가", link: "fall4.html" },
      { title: "MS, 한국 시장에서 신작 판매 제한", link: "fall5.html" }
    ]
  };
  
  function renderHeadlines(type) {
    const list = document.getElementById("headline-list");
    list.innerHTML = "";
    headlinesData[type].forEach((item, index) => {
      const li = document.createElement("li");
      li.className = "mb-1";
      li.innerHTML = `
        <a href="${item.link}" class="text-decoration-none text-dark">
          <span class="fw-bold text-primary me-2">${index + 1}.</span> ${item.title}
        </a>
      `;
      list.appendChild(li);
    });
  }
  
  // 상태 초기화
  renderHeadlines("latest");
  
  // 버튼 이벤트 연결
  document.getElementById("btn-latest").addEventListener("click", () => {
    renderHeadlines("latest");
    switchTab("latest");
  });
  
  document.getElementById("btn-popular").addEventListener("click", () => {
    renderHeadlines("popular");
    switchTab("popular");
  });
  
  document.getElementById("btn-rising").addEventListener("click", () => {
    renderHeadlines("rising");
    switchTab("rising");
  });
  
  document.getElementById("btn-falling").addEventListener("click", () => {
    renderHeadlines("falling");
    switchTab("falling");
  });
  
  function switchTab(active) {
    // 기본 탭 버튼
    ["btn-latest", "btn-popular", "btn-rising", "btn-falling"].forEach((id) => {
      const btn = document.getElementById(id);
      const isActive = id === `btn-${active}`;
      btn.classList.toggle("bg-white", isActive);
      btn.classList.toggle("bg-light", !isActive);
      btn.classList.toggle("text-muted", !isActive);
    });
  }

// 탭 전환 버튼 요소
const btnMarket = document.getElementById('btn-view-market');
const btnCuration = document.getElementById('btn-view-curation');
const tabBasic = document.getElementById('tab-basic');
const tabCuration = document.getElementById('tab-curation');

btnMarket.addEventListener('click', () => {
  // 버튼 스타일 변경
  btnMarket.classList.remove('bg-light', 'text-muted');
  btnMarket.classList.add('bg-white', 'text-dark');

  btnCuration.classList.remove('bg-white', 'text-dark');
  btnCuration.classList.add('bg-light', 'text-muted');

  // 탭 전환
  tabBasic.classList.remove('d-none');
  tabCuration.classList.add('d-none');

  // 데이터 렌더링
  renderHeadlines('latest');
  switchTab('latest');
});

btnCuration.addEventListener('click', () => {
  btnCuration.classList.remove('bg-light', 'text-muted');
  btnCuration.classList.add('bg-white', 'text-dark');

  btnMarket.classList.remove('bg-white', 'text-dark');
  btnMarket.classList.add('bg-light', 'text-muted');

  tabCuration.classList.remove('d-none');
  tabBasic.classList.add('d-none');

  renderHeadlines('rising');
  switchTab('rising');
});
 /* NOW Headlines 섹션 시작*/


  /* NOW Headlines 섹션 밑 인기 큐레이션 */
const filterData = {
  "1d": [
    { title: "초고액 자산가에게 ‘매수’ 힌트 준 베센트... 오늘은 ‘...", link: "#" },
    { title: "현재 이더리움 가격 흐름을 빠르게 체크 해 봅니다...", link: "#" },
    { title: "현재 이더리움 가격 흐름을 빠르게 체크 해 봅니다.2...", link: "#" },
    { title: "루나 다들 신청하셨나요 업비트에 물리신 분들 필독!...", link: "#" },
    { title: "트럼프 한 마디로 시장 정상화 되는 중! (9)", link: "#" }
  ],
  "7d": [
    { title: "트럼프 “중국에 더 이상 관세 올리고 싶지 않다” 발언...", link: "#" },
    { title: "마이클 세일러 치매 인증! (24)", link: "#" },
    { title: "클레이 500개 준다는 업비트 (25)", link: "#" },
    { title: "게리 겐슬러, 암호화폐 시장에 대한 강한 우려... 정말...", link: "#" },
    { title: "후기가 기대되는군요. (18)", link: "#" }
  ],
  "1m": [
    { title: "\"비트코인, 5년 내 S&P500 기업 25%의 투자 대상...", link: "#" },
    { title: "트럼프 “중국에 더 이상 관세 올리고 싶지 않다” 발언...", link: "#" },
    { title: "마이클 세일러 치매 인증! (24)", link: "#" },
    { title: "SEC 철회에 불붙은 솔라나! 45% 폭등 예고 (14)", link: "#" },
    { title: "클레이 500개 준다는 업비트 (25)", link: "#" }
  ],
  "heart": [
    { title: "2025년 4월 22일 기준 XRP에 대한 최신 소식은 ...", link: "#" },
    { title: "트럼프 한 마디로 시장 정상화 되는 중! (9)", link: "#" },
    { title: "리플 보안 이슈 해결?! 다행이네요. (5)", link: "#" },
    { title: "[공지] CBK 스테이킹 서비스 종료 안내 (6)", link: "#" },
    { title: "백악관 “중국과 협정 잘 하고 있다” (5)", link: "#" }
  ],
  "shock": [
    { title: "딥복이 상장했네요 ㅎㅎ (4)", link: "#" },
    { title: "백악관 “중국과 협정 잘 하고 있다” (5)", link: "#" },
    { title: "비트코인 조정없이 계속 올라갈까요? 잘 모르겠는데...", link: "#" },
    { title: "와 진짜 비트코인이 (2)", link: "#" },
    { title: "비트코인, 진짜 백만 달러 갈 수 있을까? 시장이 주목...", link: "#" }
  ],
  "sad": [
    { title: "[공지] CBK 스테이킹 서비스 종료 안내 (6)", link: "#" },
    { title: "비트코인 올해 20만달러 도달? (3)", link: "#" },
    { title: "초고액 자산가에게 ‘매수’ 힌트 준 베센트... 오늘은 ‘...", link: "#" },
    { title: "4월 23일 비트코인: 크게 틀렸습니다, 의사결정 과정...", link: "#" },
    { title: "넌 하나의 경이(驚異)야 (1)", link: "#" }
  ]
};

function renderFilteredList(type) {
  const list = document.getElementById("filtered-list");
  list.innerHTML = "";
  filterData[type].forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "mb-2";
    li.innerHTML = `
      <a href="${item.link}" class="text-decoration-none text-dark">
        <span class="fw-bold text-primary me-2">${index + 1}.</span> ${item.title}
      </a>
    `;
    list.appendChild(li);
  });

  // 버튼 색상 초기화
  const btns = ['btn-1d', 'btn-7d', 'btn-1m'];
  btns.forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.classList.remove('bg-white', 'text-dark');
    btn.classList.add('bg-light', 'text-muted');
  });

  // 클릭된 버튼만 스타일 적용
  const activeBtnId = 'btn-' + type;
  const activeBtn = document.getElementById(activeBtnId);
  if (activeBtn) {
    activeBtn.classList.remove('bg-light', 'text-muted');
    activeBtn.classList.add('bg-white', 'text-dark');
  }
}

function toggleMainTab(type) {
  const isCuration = (type === 'curation');
  document.getElementById('btn-main-popular').classList.toggle('text-dark', !isCuration);
  document.getElementById('btn-main-popular').classList.toggle('text-muted', isCuration);
  document.getElementById('btn-main-curation').classList.toggle('text-dark', isCuration);
  document.getElementById('btn-main-curation').classList.toggle('text-muted', !isCuration);

  document.getElementById('filter-tabs-1to3').classList.toggle('d-none', isCuration);
  document.getElementById('filter-tabs-4to6').classList.toggle('d-none', !isCuration);

  if (isCuration) {
    renderFilteredList('heart');
  } else {
    renderFilteredList('1d');
  }
}

document.getElementById('btn-main-popular').addEventListener('click', () => {
  toggleMainTab('popular');
});
document.getElementById('btn-main-curation').addEventListener('click', () => {
  toggleMainTab('curation');
});

// 페이지가 로딩되자마자 초기 필터값 표시
window.addEventListener('DOMContentLoaded', () => {
  toggleMainTab('popular');
});
  /* NOW Headlines 섹션 밑 인기 큐레이션 */

  /*Investing Insight 영역*/
  const coinInfoData = [
    { title: "비트코인과 수익", commentCount: 2, views: 63, time: "1시간 전", img: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?crop=entropy&cs=tinysrgb&fit=crop&w=100&h=100", link: "coin1.html" },
    { title: "비트코인과 수익", commentCount: 0, views: 47, time: "1시간 전", img: "", link: "coin2.html" },
    { title: "비트코인이 양지에서 더 빛나는 이유?", commentCount: 4, views: 92, time: "4시간 전", img: "https://images.unsplash.com/photo-1556740749-887f6717d7e4?crop=entropy&cs=tinysrgb&fit=crop&w=100&h=100", link: "coin3.html" },
    { title: "[기대] 트론(TRX) 저스틴 선, 트럼프표", commentCount: 5, views: 117, time: "4시간 전", img: "https://images.unsplash.com/photo-1521737604893-d14cc237f11d?crop=entropy&cs=tinysrgb&fit=crop&w=100&h=100", link: "coin4.html" },
    { title: "트럼프코인 대단하군요", commentCount: 13, views: 128, time: "5시간 전", img: "https://images.unsplash.com/photo-1593642634367-d91a135587b5?crop=entropy&cs=tinysrgb&fit=crop&w=100&h=100", link: "coin5.html" }
  ];
  
  function renderCoinInfoList() {
    const list = document.getElementById('coin-info-list');
    list.innerHTML = '';
  
    coinInfoData.forEach(item => {
      const li = document.createElement('li');
      li.className = "mb-4 py-2";
  
      li.innerHTML = `
        <a href="${item.link || '#'}" class="d-flex justify-content-between align-items-center w-100 text-decoration-none text-dark">
          <div class="flex-grow-1" style="min-width:0;">
            <div class="fw-bold text-truncate" style="max-width: calc(100% - 60px); font-size: 14px;">
              ${item.title} ${item.commentCount !== undefined ? `<span class="text-primary">(${item.commentCount})</span>` : ''}
            </div>
            <div class="text-muted small mt-1">조회 ${item.views} · ${item.time}</div>
          </div>
          ${item.img ? `<img src="${item.img}" alt="썸네일" class="ms-2 rounded-3 flex-shrink-0" style="width: 40px; height: 40px; object-fit: cover;">` : ''}
        </a>
      `;
  
      list.appendChild(li);
    });
  }
  
  // 페이지 로드될 때 호출
  renderCoinInfoList();
    /*Investing Insight 영역 끝*/

// 공지 데이터
const noticeListData = [
  { title: "[공지] CBK 스테이킹 서비스 종료 안내", link: "./notice1.html" },
  { title: "[EVENT] 🏆 CBK 에어드랍 이벤트! 코비가 되어 꿀캐기", link: "./notice2.html" },
  { title: "[공지] (4/14 수정) 코박블랙 유지 자격 오류 발생 및 복구", link: "./notice3.html" },
  { title: "[EVENT] CBK 트위터 팔로워를 위한 CBK 에어드랍!", link: "./notice4.html" },
  { title: "[공지] 회원가입, 약관 동의, 이메일 인증 절차 도입을 안내합니다", link: "./notice5.html" }
];

function renderNoticeList() {
  const list = document.getElementById('notice-list');
  list.innerHTML = '';

  noticeListData.forEach(item => {
    const li = document.createElement('li');
    li.className = 'mb-2';
    li.innerHTML = `
      <a href="${item.link}" class="text-decoration-none text-dark d-block text-truncate" style="font-size: 14px;">
        ${item.title}
      </a>
    `;
    list.appendChild(li);
  });
}

renderNoticeList();

// 거래소 리스트 채우는 데이터
const exchangeData = [
  { name: "Upbit", link: "https://upbit.com" },
  { name: "Bithumb", link: "https://www.bithumb.com" },
  { name: "HTX", link: "https://www.htx.com" },
  { name: "Coinstore", link: "https://www.coinstore.com" },
  { name: "XT.COM", link: "https://www.xt.com" },
  { name: "Bybit", link: "https://www.bybit.com" },
  { name: "Binance", link: "https://www.binance.com" },
  { name: "Bitget", link: "https://www.bitget.com" }
];

// 더미이미지 링크 하나 공통으로
const sampleExchangeLogo = "https://dummyimage.com/100x100/cccccc/000000.png&text=EX";

// 거래소 렌더링 함수
function renderExchangeList() {
  const container = document.getElementById('exchange-list');
  if (!container) {
    console.error('exchange-list 요소를 찾을 수 없습니다.');
    return;
  }
  container.innerHTML = '';

  exchangeData.forEach(exchange => {
    const a = document.createElement('a');
    a.href = exchange.link;
    a.target = '_blank';
    a.className = 'exchange-item';

    a.innerHTML = `
      <img src="${sampleExchangeLogo}" alt="${exchange.name}">
      <span>${exchange.name}</span>
    `;
    container.appendChild(a);
  });

  // 무한처럼 보이게 복제
  exchangeData.forEach(exchange => {
    const a = document.createElement('a');
    a.href = exchange.link;
    a.target = '_blank';
    a.className = 'exchange-item';

    a.innerHTML = `
      <img src="${sampleExchangeLogo}" alt="${exchange.name}">
      <span>${exchange.name}</span>
    `;
    container.appendChild(a);
  });
}

// 거래소 자동 스크롤
const exchangeList = document.querySelector('.exchange-list');

let scrollAmount = 0;
let scrollStep = 73; // 한 번에 이동할 거리 (px)
let scrollDelay = 2000; // 1초 간격 (ms)

function autoScrollExchange() {
  if (exchangeList.scrollWidth - exchangeList.clientWidth === 0) return; // 스크롤 필요 없으면 return

  scrollAmount += scrollStep;
  
  if (scrollAmount >= exchangeList.scrollWidth - exchangeList.clientWidth) {
    scrollAmount = 0; // 끝까지 가면 다시 처음으로
  }
  
  exchangeList.scrollTo({
    left: scrollAmount,
    behavior: 'smooth'
  });
}

setInterval(autoScrollExchange, scrollDelay);

// 페이지 로딩될 때 거래소 리스트도 같이 로딩
window.addEventListener('DOMContentLoaded', () => {
  renderExchangeList();
});
/* 거래소 리스트 로직 끝 */

/* 코인 시세 조회 로직 시작작 */
const coinData = [
  { name: "코박토큰", price: 816.4, change: 0.52, link: "coin1.html" },
  { name: "비트코인", price: 135314694, change: -0.22, link: "coin2.html" },
  { name: "이더리움", price: 2602300, change: 0.19, link: "coin3.html" },
  { name: "테더", price: 1438, change: -0.01, link: "coin4.html" },
  { name: "리플", price: 3136, change: -1.32, link: "coin5.html" },
  { name: "바이낸스 코인", price: 864676, change: -0.54, link: "coin6.html" },
  { name: "솔라나", price: 213641, change: -0.5, link: "coin7.html" },
  { name: "유에스디 코인", price: 1438, change: 0, link: "coin8.html" },
  { name: "도지코인", price: 260.2, change: -1.34, link: "coin9.html" },
  { name: "에이다", price: 1006, change: -2.64, link: "coin10.html" },
  { name: "트론", price: 359.4, change: -0.32, link: "coin11.html" },
  { name: "리도 스테이크 이더", price: 2598560, change: 0.12, link: "coin12.html" },
  { name: "랩피드 비트코인", price: 135511751, change: -0.04, link: "coin13.html" },
  { name: "수이", price: 5164, change: 2.54, link: "coin14.html" },
  { name: "체인링크", price: 20943, change: -2.67, link: "coin15.html" },
  { name: "아발란체", price: 32061, change: 0.18, link: "coin16.html" },
  { name: "스텔라", price: 410.6, change: -2.97, link: "coin17.html" },
  { name: "레오 토큰", price: 12960, change: -0.81, link: "coin18.html" },
  { name: "톤코인", price: 4703, change: 0.6, link: "coin19.html" }
];

let currentPage = 1;
const itemsPerPage = 10;
let filteredCoins = [...coinData];

// 코인 리스트 렌더링 함수
function renderCoinList() {
  const list = document.getElementById('coin-list');
  list.innerHTML = '';

  const start = (currentPage - 1) * itemsPerPage;
  const end = start + itemsPerPage;
  const coinsToShow = filteredCoins.slice(start, end);

  coinsToShow.forEach(coin => {
    const li = document.createElement('li');
    li.className = 'd-flex align-items-center px-2 py-2';  
    li.style.cursor = 'pointer';
    li.onclick = () => window.location.href = coin.link;

    li.innerHTML = `
      <div class="d-flex align-items-center flex-grow-1" style="min-width: 0;">
        <img src="https://dummyimage.com/24x24/cccccc/000000.png&text=⧉" alt="coin" style="width: 24px; height: 24px; object-fit: cover; border-radius: 50%; margin-right: 8px;">
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

// 페이지네이션 버튼 활성/비활성 처리
function updatePagination() {
  const totalPages = Math.ceil(filteredCoins.length / itemsPerPage);
  document.getElementById('prevPage').disabled = (currentPage === 1);
  document.getElementById('nextPage').disabled = (currentPage === totalPages);
}

// 검색 필터링 기능
document.getElementById('coinSearch').addEventListener('input', (e) => {
  const keyword = e.target.value.trim().toLowerCase();
  filteredCoins = coinData.filter(coin => coin.name.toLowerCase().includes(keyword));
  currentPage = 1;
  renderCoinList();
});

// 페이지 이동
document.getElementById('prevPage').addEventListener('click', () => {
  if (currentPage > 1) {
    currentPage--;
    renderCoinList();
  }
});
document.getElementById('nextPage').addEventListener('click', () => {
  const totalPages = Math.ceil(filteredCoins.length / itemsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderCoinList();
  }
});

// 최초 로딩
renderCoinList();