# community/jobs.py

from apscheduler.schedulers.background import BackgroundScheduler
from django_apscheduler.jobstores import DjangoJobStore, register_events
from django.core.management import call_command
from django.conf import settings
from django.contrib.auth import get_user_model
from django.utils import timezone as django_timezone
from datetime import datetime, timedelta # datetime 임포트 추가
import logging

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=settings.TIME_ZONE)
scheduler.add_jobstore(DjangoJobStore(), "default")

User = get_user_model()

def fetch_dart_job():
    """
    DART 공시를 가져오는 관리자 명령어를 실행하고, 그 결과를 FreeBoard로 포스팅합니다.
    """
    try:
        logger.info("자동 스케줄러: DART 공시 업데이트 작업 시작...")
        call_command('fetch_disclosure', '--days', '1') # 최근 1일치 DartDisclosure 테이블 업데이트
        logger.info("자동 스케줄러: DART 공시 업데이트 작업 완료.")
        
        # DartDisclosure 테이블 업데이트 후, FreeBoard로 자동 포스팅 함수 호출
        auto_post_disclosures_to_freeboard() 
    except Exception as e:
        logger.error(f"자동 스케줄러: DART 공시 업데이트 작업 중 오류: {e}")

def auto_post_disclosures_to_freeboard():
    """
    DartDisclosure 테이블의 내용을 FreeBoard 게시글로 자동 등록/업데이트합니다.
    최근 1일치 DartDisclosure 데이터 중 FreeBoard에 없거나, 카테고리가 다르거나, 삭제된 경우 처리합니다.
    """
    from .models import DartDisclosure, FreeBoard # 함수 내에서 임포트

    logger.info("자동 포스팅: DartDisclosure 내용을 FreeBoard로 자동 포스팅/업데이트 시작...")
    
    author = None
    try:
        author = User.objects.filter(is_superuser=True).first()
        if not author:
            author = User.objects.first() 
        if not author:
            logger.error("자동 포스팅: FreeBoard에 글을 작성할 사용자를 찾을 수 없습니다. 자동 포스팅을 중단합니다.")
            return
        logger.info(f"자동 포스팅: 작성자로 사용할 사용자: {author.username if hasattr(author, 'username') else author.pk}")
    except Exception as e:
        logger.error(f"자동 포스팅: 작성자 검색 중 오류: {e}")
        return

    updated_count = 0
    newly_posted_count = 0
    
    # 처리 대상 DartDisclosure 데이터 (예: 최근 1일치)
    one_day_ago = django_timezone.now().date() - timedelta(days=1)
    disclosures_to_process = DartDisclosure.objects.filter(rcept_dt__gte=one_day_ago).order_by('-rcept_dt', '-rcept_no')
    logger.info(f"자동 포스팅: FreeBoard로 옮기거나 업데이트할 대상 DartDisclosure 수 (최근 1일): {disclosures_to_process.count()}")

    for dart_item in disclosures_to_process:
        try:
            freeboard_post, created = FreeBoard.objects.get_or_create(
                dart_rcept_no=dart_item.rcept_no, # 이 접수번호로 기존 게시글 검색 또는 생성
                defaults={
                    'user': author,
                    'title': dart_item.report_nm,
                    'content': f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:2000]}...\n\n공시 원문 보기: {dart_item.report_link}",
                    'category': 'API공시',
                    'reg_dt': django_timezone.make_aware(datetime.combine(dart_item.rcept_dt, datetime.min.time())),
                    'is_deleted': False
                }
            )

            if created:
                newly_posted_count += 1
                logger.info(f"자동 포스팅: 새 FreeBoard 게시글 등록 완료 - RceptNo: {dart_item.rcept_no}, Title: {dart_item.report_nm}")
            else:
                # 이미 존재하는 게시글이라면, 카테고리와 삭제 여부 확인 및 업데이트
                made_change = False
                if freeboard_post.category != 'API공시':
                    freeboard_post.category = 'API공시'
                    made_change = True
                if freeboard_post.is_deleted: # 혹시 삭제 처리되었었다면 복구
                    freeboard_post.is_deleted = False
                    made_change = True
                
                # 내용이나 제목이 변경될 수 있으므로, 업데이트도 고려 (선택 사항)
                # freeboard_post.title = dart_item.report_nm
                # freeboard_post.content = f"회사명: {dart_item.corp_name}\n접수일자: {dart_item.rcept_dt.strftime('%Y-%m-%d')}\n\n{dart_item.document_content[:2000]}...\n\n공시 원문 보기: {dart_item.report_link}"
                # made_change = True

                if made_change:
                    freeboard_post.save()
                    updated_count += 1
                    logger.info(f"자동 포스팅: 기존 FreeBoard 게시글 업데이트 완료 - RceptNo: {dart_item.rcept_no}")
                # else:
                    # logger.debug(f"자동 포스팅: 변경사항 없어 건너뜀 - RceptNo: {dart_item.rcept_no}")

        except Exception as e:
            logger.error(f"자동 포스팅: FreeBoard 처리 중 오류 발생 (rcept_no: {dart_item.rcept_no}): {e}")

    logger.info(f"자동 포스팅: FreeBoard 자동 포스팅/업데이트 완료. 신규: {newly_posted_count}건, 업데이트: {updated_count}건")


def start_scheduler():
    if settings.DEBUG:
        try:
            if scheduler.get_job('fetch_dart_disclosure_job'):
                scheduler.remove_job('fetch_dart_disclosure_job')
                logger.info("스케줄러: 기존 'fetch_dart_disclosure_job' 작업을 제거했습니다.")
        except Exception: pass

    try:
        scheduler.add_job(
            fetch_dart_job, 
            trigger='interval',
            hours=2,
            id='fetch_dart_disclosure_job',
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=60*10
        )
        logger.info("스케줄러: 'fetch_dart_disclosure_job' 작업이 스케줄에 추가되었습니다 (2시간 간격).")
    except Exception as e:
        logger.error(f"스케줄러: 'fetch_dart_disclosure_job' 작업 추가 중 오류: {e}")

    register_events(scheduler)
    
    if not scheduler.running:
        try:
            logger.info("스케줄러: 스케줄러를 시작합니다...")
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("스케줄러: 스케줄러가 중지되었습니다.")
            scheduler.shutdown()
        except Exception as e:
            logger.error(f"스케줄러: 스케줄러 시작 중 오류: {e}")
    else:
        logger.info("스케줄러: 스케줄러가 이미 실행 중입니다.")
