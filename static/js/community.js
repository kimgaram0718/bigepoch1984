document.addEventListener('DOMContentLoaded', () => {
  const scrollTopBtn = document.getElementById('scrollTopBtn');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 200) {
      scrollTopBtn.style.display = 'block';
    } else {
      scrollTopBtn.style.display = 'none';
    }
  });
  scrollTopBtn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
});

function closeUpdateBox() {
  document.getElementById('update-container').style.display = 'none';
}

function openFilterPopup() {
  document.getElementById('filter-popup').style.display = 'flex';
}

function closeFilterPopup() {
  document.getElementById('filter-popup').style.display = 'none';
}

function selectPeriod(period) {
  document.querySelectorAll('#period-options .filter-option').forEach(option => {
    option.classList.remove('active');
    if (option.textContent === period) {
      option.classList.add('active');
    }
  });
}

function selectSort(sort) {
  document.querySelectorAll('#sort-options .filter-option').forEach(option => {
    option.classList.remove('active');
    if (option.textContent === sort) {
      option.classList.add('active');
    }
  });
}

function confirmFilter() {
  const period = document.querySelector('#period-options .filter-option.active').textContent;
  const sort = document.querySelector('#sort-options .filter-option.active').textContent;
  window.location.href = `/community/?period=${encodeURIComponent(period)}&sort=${encodeURIComponent(sort)}`;
}

function openSharePopup() {
  document.getElementById('share-popup').style.display = 'block';
}

function closeSharePopup() {
  document.getElementById('share-popup').style.display = 'none';
}

function goShare(platform) {
  const url = window.location.href;
  let shareUrl;
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
    default:
      return;
  }
  window.open(shareUrl, '_blank');
}

function toggleLike(button) {
  const postId = button.getAttribute('data-post-id');
  fetch(`/community/${postId}/like/`, {
    method: 'POST',
    headers: {
      'X-CSRFToken': getCsrfToken(),
      'Content-Type': 'application/json',
    },
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
      } else {
        button.querySelector('span').textContent = data.likes_count;
        if (data.is_liked) {
          button.classList.add('liked');
        } else {
          button.classList.remove('liked');
        }
      }
    })
    .catch(error => console.error('Error:', error));
}

function getCsrfToken() {
  return document.querySelector('meta[name="csrf-token"]')?.content ||
         document.querySelector('[name=csrfmiddlewaretoken]')?.value;
}