import requests
from datetime import datetime, timedelta
from django.conf import settings
from django.core.management.base import BaseCommand

# ↓ User 모델 가져오기
from django.contrib.auth import get_user_model
User = get_user_model()

from community.models import Disclosure, FreeBoard

class Command(BaseCommand):
    help = "DART 공시 데이터를 가져와 Disclosure 및 FreeBoard에 등록합니다."

    def handle(self, *args, **options):
        admin_user = User.objects.filter(is_superuser=True).first()
        if not admin_user:
            self.stderr.write("슈퍼유저가 없습니다. FreeBoard 글쓰기에 실패합니다.")
            return
        
        api_url = "https://opendart.fss.or.kr/api/list.json"

        # 90일 전부터 오늘까지
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=90)

        params = {
            'crtfc_key': settings.DART_API_KEY,
            'corp_code': '005930',                  # ← 여기에 조회할 회사 고유코드 (예: 삼성전자)
            'bgn_de':    start_date.strftime("%Y%m%d"),
            'end_de':    end_date.strftime("%Y%m%d"),
            'page_count': 100,
        }

        response = requests.get(api_url, params=params)
        data     = response.json()

        if data.get('status') != '000':  
            # DART API 응답 status '000'은 성공을 의미:contentReference[oaicite:4]{index=4} 
            self.stderr.write(f"API 호출 실패: {data.get('message')}")
            return

        disclosures = data.get('list', [])  # 공시 목록 데이터 리스트:contentReference[oaicite:5]{index=5}

        # 3. 가져온 공시 데이터를 모델과 게시판에 저장
        for item in disclosures:
            # 공시 종류, 제목, 날짜 등 추출
            report_name = item.get('report_nm')       # 예: "사업보고서 (2022년도)" 등:contentReference[oaicite:6]{index=6}
            rcept_date = item.get('rcept_dt')         # 접수일자 (YYYYMMDD 문자열):contentReference[oaicite:7]{index=7}
            corp_name = item.get('corp_name')         # 회사명
            # 공시 종류 추출: 보고서명에서 괄호 전에 나오는 공시 유형 부분을 사용
            # 예: "사업보고서", "주요사항보고서" 등을 추출
            if report_name is None:
                continue
            if '(' in report_name:
                disclosure_type = report_name.split('(')[0].strip()
            else:
                disclosure_type = report_name  # 괄호가 없으면 전체를 유형으로 간주

            # 날짜 문자열 -> 날짜 객체 변환 (예: "20230508" -> date(2023,05,08))
            try:
                date_obj = None
                if rcept_date:
                    date_obj = datetime.datetime.strptime(rcept_date, "%Y%m%d").date()
            except Exception as e:
                date_obj = None

            # 공시 본문 내용 구성
            # 실제 공시의 상세 본문은 DART 사이트의 문서를 파싱하거나 별도 API로 받아야 합니다.
            # 본 예제에서는 DART 공시뷰어 URL을 포함하여 본문을 구성합니다.
            content_text = f"{corp_name} 공시 \"{report_name}\"의 상세 내용은 DART 사이트를 참고하세요.\n"
            content_text += f"공시 원문 보기: https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no')}"

            # Disclosure 저장
            disclosure_obj = Disclosure.objects.create(
                disclosure_type=disclosure_type,
                date=date_obj,
                title=report_name,
                content=content_text
            )

            # 커뮤니티 게시판에도 글 작성 (user 필수)
            FreeBoard.objects.create(
                user    = admin_user,      # ← 여기에 user 인스턴스를 넣습니다
                title   = report_name,
                content = content_text
                # 필요하다면 다른 필드도 지정 (예: category, tags 등)
            )

        self.stdout.write(self.style.SUCCESS(f"{len(disclosures)}건의 공시를 저장하고 게시판에 등록했습니다."))
