from django.core.management.base import BaseCommand
from predict.models import StockPrice
from pykrx import stock
import datetime
import math
import time

class Command(BaseCommand):
    help = '코스피/코스닥 전체 종목의 최근 30일치 일별 시세 데이터를 DB에 저장합니다.'

    def handle(self, *args, **options):
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=29)
        start = start_date.strftime('%Y%m%d')
        end = end_date.strftime('%Y%m%d')
        markets = [
            ("KOSPI", stock.get_market_ticker_list(date=end, market="KOSPI")),
            ("KOSDAQ", stock.get_market_ticker_list(date=end, market="KOSDAQ"))
        ]
        total_codes = sum(len(codes) for _, codes in markets)
        total_days = 30
        total_tasks = total_codes * total_days
        done = 0
        start_time = time.time()
        for market, codes in markets:
            self.stdout.write(f"{market} 전체 종목 최근 30일치 데이터 수집 시작...")
            for code in codes:
                name = stock.get_market_ticker_name(code)
                df = stock.get_market_ohlcv_by_date(start, end, code)
                for date, row in df.iterrows():
                    # 등락률 결측치 처리
                    change_rate = row["등락률"]
                    if change_rate is None or (isinstance(change_rate, float) and math.isnan(change_rate)):
                        change_rate = 0.0
                    StockPrice.objects.update_or_create(
                        code=code,
                        date=date.date(),
                        defaults={
			    "market" : market , 
			    "name": name ,  
                            "close": float(row["종가"]),
                            "high": float(row["고가"]),
                            "low": float(row["저가"]),
                            "open": float(row["시가"]),
                            "volume": int(row["거래량"]),
                            "change_rate": float(change_rate),
                        }
                    )
                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - start_time
                        percent = done / total_tasks * 100
                        est_total = elapsed / done * total_tasks
                        est_left = est_total - elapsed
                        self.stdout.write(f"진행률: {percent:.2f}% ({done}/{total_tasks}), 경과: {elapsed/60:.1f}분, 남음: {est_left/60:.1f}분")
            self.stdout.write(f"{market} 저장 완료")
        self.stdout.write("모든 데이터 저장 완료!") 
