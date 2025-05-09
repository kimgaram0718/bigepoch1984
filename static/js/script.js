document.addEventListener('DOMContentLoaded', () => {
    initializeHeaderFeatures();
    fetchFooter();
    if (typeof toggleMainTab === 'function' && document.getElementById('btn-main-popular')) {
        toggleMainTab('popular');
    }
    initializeScrollToTopButton();
    initializeBanners();
    initializeRealtimeSearch();
    if (typeof renderOriginalsList === 'function' && document.getElementById('originals-list')) {
        renderOriginalsList();
    }
    initializeHeadlines();
    if (typeof renderCoinInfoList === 'function' && document.getElementById('coin-info-list')) {
        renderCoinInfoList();
    }
    if (typeof renderNoticeList === 'function' && document.getElementById('notice-list')) {
        renderNoticeList();
    }
    initializeExchangeList();
    if (typeof renderCoinList === 'function' && document.getElementById('coin-list')) {
        renderCoinList();
    }
    initializeWriteButton();
});

function initializeHeaderFeatures() {
    const alertIcon = document.getElementById('alertIcon');
    const profileIcon = document.getElementById('profileIcon');
    const mainPanelButton = document.getElementById('mainPanelIcon');

    const isLoggedIn = document.body.dataset.isAuthenticated === 'true';

    if (isLoggedIn) {
        if (alertIcon) {
            alertIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                togglePopup('alert-popup');
            });
        }

        if (profileIcon) {
            profileIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                setupAndToggleProfilePopup();
            });
        }
    }

    if (mainPanelButton) {
        mainPanelButton.addEventListener('click', (event) => {
            event.stopPropagation();
            openLoginPanel();
        });
    } else {
        console.error('Error: mainPanelIcon element not found in the HTML.');
    }

    document.addEventListener('click', (event) => {
        const activePopups = document.querySelectorAll('.popup-box:not(.d-none)');
        activePopups.forEach(popup => {
            if (!popup.contains(event.target)) {
                let clickedOnToggleButton = false;
                if (popup.id === 'alert-popup' && alertIcon && alertIcon.contains(event.target)) clickedOnToggleButton = true;
                if (popup.id === 'profile-popup' && profileIcon && profileIcon.contains(event.target)) clickedOnToggleButton = true;

                if (!clickedOnToggleButton) {
                    popup.classList.add('d-none');
                }
            }
        });
    });
}

function setupAndToggleProfilePopup() {
    let profilePopup = document.getElementById('profile-popup');
    if (!profilePopup) {
        const nickname = document.body.dataset.userNickname;
        const popupHtml = `
            <div id="profile-popup" class="popup-box d-none" style="position: absolute; top: 60px; right: 20px; width: 200px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); z-index: 9999; font-size: 14px;">
                <div class="px-3 py-2 d-flex align-items-center">
                    <i class="bi bi-person-circle me-2" style="font-size: 20px; color: #9376e0;"></i>
                    <span class="fw-normal text-dark" style="font-weight: 500;">${nickname}</span>
                </div>
                <ul class="list-unstyled m-0">
                    <li class="px-3 py-2 hover-bg text-muted" onclick="location.href='{% url 'mypage:mypage' %}'">마이페이지 이동</li>
                    <li class="px-3 py-2 hover-bg text-muted" onclick="location.href='{% url 'space:space' %}'">스페이스 이동</li>
                    <li class="px-3 py-2 hover-bg text-danger" onclick="performLogout()">로그아웃</li>
                </ul>
            </div>`;
        document.body.insertAdjacentHTML('beforeend', popupHtml);
        profilePopup = document.getElementById('profile-popup');
    }

    if (profilePopup) {
        profilePopup.classList.toggle('d-none');
    }
}

function performLogout() {
    window.location.href = document.body.dataset.logoutUrl;
}

function togglePopup(id) {
    const popup = document.getElementById(id);
    if (popup) {
        popup.classList.toggle('d-none');
    }
}

function openLoginPanel() {
    let panel = document.getElementById('login-panel');
    if (!panel) {
        const bodyData = document.body.dataset;
        const isAuthenticated = bodyData.isAuthenticated === 'true';
        const logoutUrl = bodyData.logoutUrl || '#';
        const loginUrl = bodyData.loginUrl || '#';

        const panelHtml = `
            <div id="login-panel" class="login-panel d-none" style="position: fixed; top: 0; right: 0; width: 250px; height: 100%; background: white; box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1); z-index: 10000; transform: translateX(100%); transition: transform 0.3s ease-in-out;">
                <div id="login-panel-inner" class="p-3">
                    <!-- 닫기 버튼 추가 -->
                    <div class="d-flex justify-content-end mb-3">
                        <i class="bi bi-x-lg" id="closePanelIcon" style="font-size: 20px; cursor: pointer; color: #1b1e26;"></i>
                    </div>
                    ${isAuthenticated ? `
                        <div class="login-menu-wrapper">
                            <div class="login-menu-header p-3 border-bottom">
                                <h5 class="mb-0">안녕하세요, ${bodyData.userNickname}님!</h5>
                            </div>
                            <div class="login-menu-body">
                                <div class="login-menu-grid">
                                    <div class="menu-item" onclick="location.href='{% url 'alert:alert' %}'">
                                        <i class="bi bi-bell"></i>
                                        <span style="font-size: 12px;">알림</span>
                                    </div>
                                    <div class="menu-item profile-icon" onclick="location.href='{% url 'mypage:mypage' %}'">
                                        <i class="bi bi-person-circle" style="color: #9376e0;"></i>
                                        <span style="font-size: 12px;">프로필</span>
                                    </div>
                                </div>
                                <div class="mt-4 px-3">
                                    <button class="btn btn-outline-secondary w-100" onclick="window.location.href='${logoutUrl}'">로그아웃</button>
                                </div>
                            </div>
                        </div>
                    ` : `
                        <div class="text-center mt-5 p-3">
                            <p class="mt-3 fw-bold">더 많은 기능을 위해<br />로그인하세요.</p>
                            <button class="btn btn-primary mt-3 px-4 w-100" onclick="location.href='${loginUrl}'">로그인</button>
                        </div>
                    `}
                </div>
            </div>`;
        document.body.insertAdjacentHTML('beforeend', panelHtml);
        panel = document.getElementById('login-panel');

        // 닫기 버튼 이벤트 리스너 추가
        const closePanelIcon = document.getElementById('closePanelIcon');
        if (closePanelIcon) {
            closePanelIcon.addEventListener('click', (event) => {
                event.stopPropagation();
                closeLoginPanel();
            });
        }
    }

    if (panel) {
        if (panel.classList.contains('d-none')) {
            panel.classList.remove('d-none');
            setTimeout(() => panel.style.transform = 'translateX(0)', 0); // 슬라이드 인
        }
        document.body.classList.add('no-scroll');
        setTimeout(() => {
            document.addEventListener('click', handleOutsideClickForPanel, { once: true });
        }, 0);
    }
}

function handleOutsideClickForPanel(e) {
    const panel = document.getElementById('login-panel');
    const mainPanelButton = document.getElementById('mainPanelIcon');
    if (panel && !panel.contains(e.target) && mainPanelButton && !mainPanelButton.contains(e.target)) {
        closeLoginPanel();
    }
}

function closeLoginPanel() {
    const panel = document.getElementById('login-panel');
    if (panel) {
        panel.style.transform = 'translateX(100%)';
        setTimeout(() => panel.classList.add('d-none'), 300); // 애니메이션 시간과 동기화
        document.body.classList.remove('no-scroll');
    }
}

function setupLoginPanelContent() {
    // 이 함수는 더 이상 별도로 호출되지 않으며, openLoginPanel 내에서 처리됩니다.
}

function fetchFooter() {
    const footerContainer = document.getElementById('footer-container');
    if (footerContainer) {
        const footerItems = footerContainer.querySelectorAll('.footer-item');
        if (footerItems.length > 0) {
            const currentPath = window.location.pathname.split('/').pop() || 'main';
            footerItems.forEach(item => {
                const href = item.getAttribute('href');
                if (href && (href === currentPath || (currentPath === 'main' && (href === '/' || href === 'main.html')))) {
                    footerItems.forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                }
            });

            footerItems.forEach(item => {
                item.addEventListener('click', (e) => {
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
            scrollTopBtn.style.display = 'flex';
        } else {
            scrollTopBtn.style.display = 'none';
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}

const banners = [
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
        `;
        carouselInner.appendChild(itemDiv);
    });

    if (banners.length > 0) {
        bannerCountDiv.textContent = `1 / ${banners.length}`;
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

const realtimeItems = [
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
        li.innerHTML = `<a href="${item.link}" class="text-primary text-decoration-none">${item.rank}. ${item.title}</a>`;
        realtimeList.appendChild(li);
    });

    realtimeToggle.addEventListener('click', toggleRealtime);
    realtimeCloseButton.addEventListener('click', () => {
        realtimeDropdown.classList.add('d-none');
        realtimeToggle.classList.remove('d-none');
    });
}

function toggleRealtime() {
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

const originalsData = [ /* ... 기존 데이터 ... */ ];
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

    if (btnLatest) renderHeadlines("latest");

    if (btnLatest) btnLatest.addEventListener("click", () => { renderHeadlines("latest"); switchTab("latest"); });
    if (btnPopular) btnPopular.addEventListener("click", () => { renderHeadlines("popular"); switchTab("popular"); });
    if (btnRising) btnRising.addEventListener("click", () => { renderHeadlines("rising"); switchTab("rising"); });
    if (btnFalling) btnFalling.addEventListener("click", () => { renderHeadlines("falling"); switchTab("falling"); });

    if (btnMarket && btnCuration) {
        const tabBasic = document.getElementById('tab-basic');
        const tabCuration = document.getElementById('tab-curation');

        btnMarket.addEventListener('click', () => {
            if (btnMarket.classList.contains('active')) return;
            btnMarket.classList.add('active'); btnCuration.classList.remove('active');
            btnMarket.classList.remove('bg-light', 'text-muted'); btnMarket.classList.add('bg-white', 'text-dark');
            btnCuration.classList.remove('bg-white', 'text-dark'); btnCuration.classList.add('bg-light', 'text-muted');
            if (tabBasic) tabBasic.classList.remove('d-none');
            if (tabCuration) tabCuration.classList.add('d-none');
            renderHeadlines('latest'); switchTab('latest');
        });
        btnCuration.addEventListener('click', () => {
            if (btnCuration.classList.contains('active')) return;
            btnCuration.classList.add('active'); btnMarket.classList.remove('active');
            btnCuration.classList.remove('bg-light', 'text-muted'); btnCuration.classList.add('bg-white', 'text-dark');
            btnMarket.classList.remove('bg-white', 'text-dark'); btnMarket.classList.add('bg-light', 'text-muted');
            if (tabCuration) tabCuration.classList.remove('d-none');
            if (tabBasic) tabBasic.classList.add('d-none');
            renderHeadlines('rising'); switchTab('rising');
        });
        if (btnMarket) btnMarket.classList.add('active'); else if (btnCuration) btnCuration.classList.add('active');
    }
}

const filterData = { /* ... 기존 데이터 ... */ };
function renderFilteredList(type) { /* ... 기존 함수 ... */ }
function toggleMainTab(type) { /* ... 기존 함수 ... */ }

const coinInfoData = [ /* ... 기존 데이터 ... */ ];
function renderCoinInfoList() { /* ... 기존 함수 ... */ }

const noticeListData = [ /* ... 기존 데이터 ... */ ];
function renderNoticeList() { /* ... 기존 함수 ... */ }

const exchangeData = [ /* ... 기존 데이터 ... */ ];
const sampleExchangeLogo = "https://placehold.co/100x100/cccccc/000000?text=EX";
function renderExchangeList() {
    const container = document.getElementById('exchange-list');
    if (!container) { console.error('exchange-list 요소를 찾을 수 없습니다.'); return; }
    container.innerHTML = '';
    [...exchangeData, ...exchangeData].forEach(exchange => {
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
    renderExchangeList();

    if (exchangeScrollInterval) clearInterval(exchangeScrollInterval);

    if (exchangeListEl.clientWidth > 0 && exchangeListEl.scrollWidth > exchangeListEl.clientWidth) {
        let scrollAmount = 0;
        const scrollStep = 73;
        const scrollDelay = 2000;
        exchangeScrollInterval = setInterval(() => {
            scrollAmount += scrollStep;
            if (scrollAmount >= exchangeListEl.scrollWidth / 2) {
                exchangeListEl.scrollTo({ left: 0, behavior: 'auto' });
                scrollAmount = scrollStep;
            }
            exchangeListEl.scrollTo({ left: scrollAmount, behavior: 'smooth' });
        }, scrollDelay);
    }
}

const coinData = [ /* ... 기존 데이터 ... */ ];
let currentPage = 1; const itemsPerPage = 10; let filteredCoins = [...coinData];
function renderCoinList() {
    const list = document.getElementById('coin-list');
    if (!list) return;
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
document.addEventListener('DOMContentLoaded', () => {
    const coinSearchInput = document.getElementById('coinSearch');
    const prevPageBtn = document.getElementById('prevPage');
    const nextPageBtn = document.getElementById('nextPage');

    if (coinSearchInput) {
        coinSearchInput.addEventListener('input', (e) => {
            const keyword = e.target.value.trim().toLowerCase();
            filteredCoins = coinData.filter(coin => coin.name.toLowerCase().includes(keyword));
            currentPage = 1;
            renderCoinList();
        });
    }
    if (prevPageBtn) {
        prevPageBtn.addEventListener('click', () => {
            if (currentPage > 1) { currentPage--; renderCoinList(); }
        });
    }
    if (nextPageBtn) {
        nextPageBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(filteredCoins.length / itemsPerPage);
            if (currentPage < totalPages) { currentPage++; renderCoinList(); }
        });
    }
    if (document.getElementById('coin-list')) renderCoinList();
});

const recentPosts = [ /* ... 기존 데이터 ... */ ];

function initializeWriteButton() {
    const bodyData = document.body.dataset;
    const isLoggedIn = bodyData.isAuthenticated === 'true';
    const writeUrl = bodyData.writeUrl || 'write.html';

    if (!isLoggedIn) return;

    const writeBtn = document.createElement('button');
    writeBtn.id = 'goToWriteBtn';
    writeBtn.className = 'btn btn-primary rounded-circle d-flex align-items-center justify-content-center';
    writeBtn.style.cssText = `
        position: fixed; bottom: 150px; right: 15px; z-index: 1000;
        width: 45px; height: 45px; font-size: 22px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: flex;`;
    writeBtn.innerHTML = '<i class="bi bi-pencil" style="font-size: 17px;"></i>';
    document.body.appendChild(writeBtn);

    writeBtn.addEventListener('click', () => {
        window.location.href = writeUrl;
    });
}

async function fetchNaverNews() {
    try {
        const response = await fetch('/api/news/');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching news:', data.error);
            return;
        }
        
        const newsList = document.getElementById('headline-list');
        newsList.innerHTML = ''; // 기존 내용 초기화
        
        data.news.forEach(news => {
            const newsItem = document.createElement('li');
            newsItem.className = 'news-item mb-3';
            
            const pubDate = new Date(news.pubDate);
            const formattedDate = `${pubDate.getFullYear()}-${String(pubDate.getMonth() + 1).padStart(2, '0')}-${String(pubDate.getDate()).padStart(2, '0')} ${String(pubDate.getHours()).padStart(2, '0')}:${String(pubDate.getMinutes()).padStart(2, '0')}`;
            
            newsItem.innerHTML = `
                <a href="${news.link}" target="_blank" class="text-decoration-none text-dark">
                    <div class="d-flex flex-column">
                        <h6 class="mb-1" style="font-size: 14px; line-height: 1.4;">${news.title}</h6>
                        <p class="text-muted mb-1" style="font-size: 12px;">${news.description}</p>
                        <small class="text-muted" style="font-size: 11px;">${formattedDate}</small>
                    </div>
                </a>
            `;
            
            newsList.appendChild(newsItem);
        });
    } catch (error) {
        console.error('Error:', error);
    }
}

document.addEventListener('DOMContentLoaded', function() {
    fetchNaverNews();
    setInterval(fetchNaverNews, 5 * 60 * 1000);
});