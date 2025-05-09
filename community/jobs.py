# community/jobs.py

from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django.core.management import call_command
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone as django_timezone
from datetime import datetime, timedelta
import logging
import requests 
from dateutil import parser as date_parser 

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=settings.TIME_ZONE)
scheduler.add_jobstore(DjangoJobStore(), "default")

User = get_user_model()

# --- DART 공시 관련 작업 ---
def fetch_dart_job():
    """
    (스케줄러에 의해 주기적으로 실행됨)
    1. DART API에서 최신 공시를 가져와 DartDisclosure 테이블을 업데이트합니다.
    2. DartDisclosure 테이블의 내용을 FreeBoard로 포스팅합니다.
    """
    try:
        logger.info("스케줄러 실행: DART 공시 업데이트 작업 시작 (fetch_disclosure 명령어 호출)...")
        call_command('fetch_disclosure', '--days', '1') # DartDisclosure 테이블 업데이트
        logger.info("스케줄러 실행: DART 공시 업데이트 작업 완료 (fetch_disclosure 명령어 호출 완료).")
        
        logger.info("스케줄러 실행: DartDisclosure -> FreeBoard 포스팅 작업 시작...")
        auto_post_disclosures_to_freeboard() # FreeBoard 업데이트
        logger.info("스케줄러 실행: DartDisclosure -> FreeBoard 포스팅 작업 완료.")
    except Exception as e:
        logger.error(f"스케줄러 실행: DART 공시 처리 작업 중 오류: {e}")

def auto_post_disclosures_to_freeboard(initial_run=False): # 초기 실행 여부 플래그 추가
    """
    DartDisclosure 테이블의 내용을 FreeBoard 게시글로 자동 등록/업데이트합니다.
    initial_run=True일 경우, 최근 1일치 전체를 대상으로 합니다.
    initial_run=False일 경우 (스케줄러에 의한 실행), 최근 1일치 중 새로운 것만 처리합니다.
    """
    from .models import DartDisclosure, FreeBoard 

    log_prefix = "초기 실행 - 자동 포스팅:" if initial_run else "주기 실행 - 자동 포스팅:"
    logger.info(f"{log_prefix} DartDisclosure 내용을 FreeBoard로 자동 포스팅/업데이트 시작...")
    
    author = None
    try:
        author = User.objects.filter(is_superuser=True).first()
        if not author:
            author = User.objects.first() 
        if not author:
            logger.error(f"{log_prefix} FreeBoard에 글을 작성할 사용자를 찾을 수 없습니다. 자동 포스팅을 중단합니다.")
            return
        logger.info(f"{log_prefix} 작성자로 사용할 사용자: {getattr(author, author.USERNAME_FIELD, author.pk)}")
    except Exception as e:
        logger.error(f"{log_prefix} 작성자 검색 중 오류: {e}")
        return

    updated_count = 0
    newly_posted_count = 0
    
    # 처리 대상 DartDisclosure 데이터 (최근 1일치)
    one_day_ago = django_timezone.now().date() - timedelta(days=1)
    disclosures_to_process = DartDisclosure.objects.filter(rcept_dt__gte=one_day_ago).order_by('-rcept_dt', '-rcept_no')
    logger.info(f"{log_prefix} FreeBoard로 옮기거나 업데이트할 대상 DartDisclosure 수 (최근 1일): {disclosures_to_process.count()}")

    for dart_item in disclosures_to_process:
        try:
            freeboard_post, created = FreeBoard.objects.get_or_create(
                dart_rcept_no=dart_item.rcept_no, 
                defaults={
                    'user': author,
                    'title': dart_item.report_nm, 
                    'content': f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:1000]}...\n\n공시 원문 보기: {dart_item.report_link}",
                    'category': 'API공시',
                    'reg_dt': django_timezone.make_aware(datetime.combine(dart_item.rcept_dt, datetime.min.time())), 
                    'is_deleted': False
                }
            )

            if created:
                newly_posted_count += 1
                logger.info(f"{log_prefix} 새 FreeBoard 게시글 등록 완료 - RceptNo: {dart_item.rcept_no}, Title: {dart_item.report_nm}")
            else:
                made_change = False
                if freeboard_post.category != 'API공시':
                    freeboard_post.category = 'API공시'
                    made_change = True
                if freeboard_post.is_deleted: 
                    freeboard_post.is_deleted = False
                    made_change = True
                
                # 내용이나 제목이 변경될 수 있으므로, 업데이트도 고려 (선택 사항)
                # current_title = dart_item.report_nm
                # current_content = f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:1000]}...\n\n공시 원문 보기: {dart_item.report_link}"
                # if freeboard_post.title != current_title or freeboard_post.content != current_content:
                #     freeboard_post.title = current_title
                #     freeboard_post.content = current_content
                #     made_change = True

                if made_change:
                    freeboard_post.save()
                    updated_count += 1
                    logger.info(f"{log_prefix} 기존 FreeBoard 게시글 업데이트 완료 - RceptNo: {dart_item.rcept_no}")

        except Exception as e:
            logger.error(f"{log_prefix} FreeBoard 처리 중 오류 발생 (rcept_no: {dart_item.rcept_no}): {e}")

    logger.info(f"{log_prefix} FreeBoard 자동 포스팅/업데이트 완료. 신규: {newly_posted_count}건, 업데이트: {updated_count}건")

# --- 네이버 뉴스 관련 작업 ---
def fetch_and_save_naver_news_job(initial_run=False): # 초기 실행 여부 플래그 추가
    from .models import NewsArticle 

    log_prefix = "초기 실행 - 네이버 뉴스:" if initial_run else "주기 실행 - 네이버 뉴스:"
    client_id = getattr(settings, "NAVER_NEWS_API_CLIENT_ID", None)
    client_secret = getattr(settings, "NAVER_NEWS_API_CLIENT_SECRET", None)

    if not client_id or not client_secret:
        logger.error(f"{log_prefix} API ID 또는 Secret이 settings.py에 설정되지 않았습니다.")
        return

    query = "증시 OR 주식 OR 코스피 OR 코스닥 OR 경제 OR 금융 OR 투자 OR 기업"
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    # 초기 실행 시 더 많은 뉴스를 가져오도록 display 값 조정 가능 (예: 50)
    params = {"query": query, "display": 50 if initial_run else 30, "sort": "date", "start": 1} 
    
    logger.info(f"{log_prefix} API 호출 시작 (검색어: {query}, display: {params['display']})")
    saved_count = 0
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        news_data = response.json()
        
        if news_data.get('items'):
            for item in news_data['items']:
                original_link = item.get('originallink')
                naver_link = item.get('link')
                title = item.get('title', '').replace('<b>', '').replace('</b>', '').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                description = item.get('description', '').replace('<b>', '').replace('</b>', '').replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                
                pub_date_to_save = None
                try:
                    parsed_dt = date_parser.parse(item.get('pubDate')) 
                    if settings.USE_TZ:
                        if django_timezone.is_naive(parsed_dt):
                            pub_date_to_save = django_timezone.make_aware(parsed_dt, timezone=django_timezone.get_default_timezone())
                        else: 
                            pub_date_to_save = parsed_dt.astimezone(django_timezone.get_default_timezone())
                    else:
                        if django_timezone.is_aware(parsed_dt):
                            local_dt = parsed_dt.astimezone(django_timezone.get_default_timezone())
                            pub_date_to_save = local_dt.replace(tzinfo=None)
                        else:
                            pub_date_to_save = parsed_dt
                except (ValueError, TypeError, AttributeError) as e: 
                    logger.warning(f"{log_prefix} 뉴스 발행일 파싱/변환 오류 (제목: {title}, pubDate: {item.get('pubDate')}): {e}. 현재 시간(naive)으로 대체합니다.")
                    pub_date_to_save = datetime.now() 

                if original_link: 
                    if not NewsArticle.objects.filter(original_link=original_link).exists():
                        NewsArticle.objects.create(
                            title=title, original_link=original_link, naver_link=naver_link,
                            description=description, pub_date=pub_date_to_save 
                        )
                        saved_count += 1
            logger.info(f"{log_prefix} {len(news_data['items'])}건 중 {saved_count}건 신규 저장 완료.")
        else:
            logger.info(f"{log_prefix} API로부터 가져온 뉴스 항목이 없습니다. 응답: {news_data}")
    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix} API 요청 실패: {e}")
    except Exception as e:
        logger.error(f"{log_prefix} 처리 중 오류 발생: {e}")

def start_scheduler():
    if settings.DEBUG:
        try:
            if scheduler.get_job('fetch_dart_disclosure_job'): scheduler.remove_job('fetch_dart_disclosure_job')
            if scheduler.get_job('fetch_naver_news_job'): scheduler.remove_job('fetch_naver_news_job')
            logger.info("스케줄러: 기존 작업들 제거 시도 완료.")
        except Exception: pass

    try:
        scheduler.add_job(
            fetch_dart_job, 
            trigger='interval', hours=2, id='fetch_dart_disclosure_job',
            max_instances=1, replace_existing=True, misfire_grace_time=60*10
        )
        logger.info("스케줄러: 'fetch_dart_disclosure_job' 작업 추가 (2시간 간격).")

        scheduler.add_job(
            fetch_and_save_naver_news_job, # 주기 실행 시에는 initial_run=False (기본값)
            trigger='interval', minutes=30, id='fetch_naver_news_job',
            max_instances=1, replace_existing=True, misfire_grace_time=60*5
        )
        logger.info("스케줄러: 'fetch_naver_news_job' 작업 추가 (30분 간격).")

    except Exception as e:
        logger.error(f"스케줄러: 작업 추가 중 오류: {e}")

    register_events(scheduler)
    
    if not scheduler.running:
        try:
            logger.info("스케줄러: 스케줄러를 시작합니다...")
            scheduler.start()
        except KeyboardInterrupt: logger.info("스케줄러: 중지됨."); scheduler.shutdown()
        except Exception as e: logger.error(f"스케줄러: 시작 중 오류: {e}")
    else:
        logger.info("스케줄러: 이미 실행 중입니다.")

