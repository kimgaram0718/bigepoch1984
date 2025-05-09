# community/apps.py

from django.apps import AppConfig
import os
from django.conf import settings

class CommunityConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'community'

    def ready(self):
        run_main_direct = os.environ.get('RUN_MAIN', None) == 'true'
        
        from . import jobs

        # 스케줄러 시작 (주기적인 DART API 데이터 가져오기 및 FreeBoard 포스팅용)
        # jobs.py의 fetch_dart_job 함수 내에서 auto_post_disclosures_to_freeboard가 호출됩니다.
        if settings.DEBUG and run_main_direct: # 개발 서버의 메인 프로세스에서만 실행
            if not jobs.scheduler.running:
                jobs.start_scheduler()
                print("커뮤니티 앱 준비: DART 공시 스케줄러 시작됨 (2시간 간격).")
            
            # runserver 시작 시 1회만 실행하여 기존 DartDisclosure 내용을 FreeBoard로 옮김
            # (이미 스케줄러의 fetch_dart_job 내에서 처리되므로, 여기서 중복 호출할 필요는 없음)
            # 다만, 서버 시작 시점에 즉시 동기화를 원한다면 아래 로직을 유지할 수 있습니다.
            # 또는, 초기 데이터 로딩은 별도의 관리자 명령어로 분리하는 것이 더 깔끔할 수 있습니다.
            print("커뮤니티 앱 준비: 서버 시작 시 DartDisclosure -> FreeBoard 동기화 시도...")
            try:
                jobs.auto_post_disclosures_to_freeboard()
            except Exception as e:
                print(f"커뮤니티 앱 준비: FreeBoard 자동 포스팅 중 오류 발생: {e}")
        # else:
            # current_scheduler_running = 'N/A'
            # try: current_scheduler_running = jobs.scheduler.running
            # except Exception: pass
            # print(f"커뮤니티 앱 준비 - 스케줄러 시작 조건 미충족 (DEBUG: {settings.DEBUG}, RUN_MAIN: {run_main_direct}, SCHEDULER_RUNNING: {current_scheduler_running})")

