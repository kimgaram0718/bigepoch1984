function closeUpdateBox() {
  const container = document.getElementById('update-container');
  if (container) container.style.display = 'none';
}

function openFilterPopup() {
  const popup = document.getElementById('filter-popup');
  popup.style.display = 'block';
  setTimeout(() => popup.classList.add('show'), 10);
}

function closeFilterPopup() {
  const popup = document.getElementById('filter-popup');
  popup.classList.remove('show');
  setTimeout(() => popup.style.display = 'none', 300);
}

function selectPeriod(period) {
  document.querySelectorAll('#period-options .filter-option').forEach(option => {
    option.classList.toggle('active', option.textContent === period);
  });
}

function selectSort(sort) {
  document.querySelectorAll('#sort-options .filter-option').forEach(option => {
    option.classList.toggle('active', option.textContent === sort);
  });
}

function confirmFilter() {
  const period = document.querySelector('#period-options .filter-option.active')?.textContent || '한달';
  const sort = document.querySelector('#sort-options .filter-option.active')?.textContent || '최신순';
  const url = new URL(window.location);
  url.searchParams.set('period', period);
  url.searchParams.set('sort', sort);
  window.location.href = url.toString();
}

function openSharePopup() {
  document.getElementById('share-popup').style.display = 'block';
}

function closeSharePopup() {
  document.getElementById('share-popup').style.display = 'none';
}

function goShare(platform) {
  const url = window.location.href;
  let shareUrl = '';
  switch (platform) {
    case 'kakao':
      shareUrl = `https://story.kakao.com/s/share?url=${encodeURIComponent(url)}`;
      break;
    case 'telegram':
      shareUrl = `https://t.me/share/url?url=${encodeURIComponent(url)}`;
      break;
    case 'facebook':
      shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`;
      break;
    case 'twitter':
      shareUrl = `https://twitter.com/intent/tweet?url=${encodeURIComponent(url)}`;
      break;
  }
  window.open(shareUrl, '_blank');
}

// 푸터 항목 활성화
document.addEventListener('DOMContentLoaded', () => {
  const footerItems = document.querySelectorAll('.footer-item');
  const currentPath = window.location.pathname;

  footerItems.forEach(item => {
    const href = item.getAttribute('href');
    if (currentPath === href) {
      footerItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');
    }
  });
});