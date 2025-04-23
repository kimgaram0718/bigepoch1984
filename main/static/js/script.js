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