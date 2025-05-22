# predict_info/management/commands/update_daily_data.py
import pandas as pd
import FinanceDataReader as fdr
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta, datetime, date # 'date'를 명시적으로 임포트
from predict_info.models import MarketIndex, StockPrice 
from pandas.tseries.offsets import BDay 
import holidays 
import time
import traceback

class Command(BaseCommand):
    help = ('Fetches and stores stock data. Ensures a minimum history (e.g., 60 trading days) '
            'and updates with the previous trading day\'s data, including fundamentals.')

    # DB에 최소한으로 유지하고자 하는 거래일 수
    MIN_RECORDS_FOR_SUFFICIENT_HISTORY = 60 
    # 데이터가 부족할 경우 한 번에 가져올 과거 거래일 수 (MIN_RECORDS_FOR_SUFFICIENT_HISTORY 보다 크게)
    DAYS_TO_FETCH_FOR_BACKFILL = 80 

    def get_previous_trading_day(self, reference_date, days_offset=1):
        if not isinstance(reference_date, date):
            try: reference_date = pd.to_datetime(reference_date).date()
            except: raise TypeError("reference_date must be a datetime.date or convertible to it.")
        
        # 휴일 정보는 매번 다시 계산 (연도가 바뀔 수 있으므로)
        kr_holidays = holidays.KR(years=list(set([reference_date.year - (days_offset // 200 + 2), reference_date.year, reference_date.year + 1])))
        
        current_check_date = reference_date
        trading_days_found = 0
        
        while trading_days_found < days_offset:
            current_check_date -= timedelta(days=1)
            if current_check_date.weekday() < 5 and current_check_date not in kr_holidays:
                trading_days_found += 1
        return current_check_date

    def get_n_trading_days_before(self, reference_date, n_days):
        """reference_date로부터 n 거래일 이전의 날짜를 반환합니다."""
        return self.get_previous_trading_day(reference_date, n_days)


    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Starting daily data update with backfill logic (min history: {self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY} days)..."))

        today_dt = timezone.now()
        # 데이터 가져올 최종 목표 날짜 (어제 거래일)
        target_date = self.get_previous_trading_day(today_dt, 1)
        
        self.stdout.write(f"Target date for data processing: {target_date}")

        # 1. 시장 지수 업데이트 (target_date 및 그 이전 거래일 데이터 필요)
        day_before_target_date_for_index = self.get_previous_trading_day(target_date, 1)
        indices_to_fetch = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11'}
        for market_readable_name, fdr_ticker in indices_to_fetch.items():
            if MarketIndex.objects.filter(market_name=market_readable_name, date=target_date).exists():
                self.stdout.write(self.style.SUCCESS(f"{market_readable_name} index data for {target_date} already exists. Skipping."))
                continue
            self.stdout.write(f"Fetching data for {market_readable_name} index ({fdr_ticker})...")
            try:
                df_index = fdr.DataReader(fdr_ticker, start=day_before_target_date_for_index, end=target_date)
                if df_index.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} for dates {day_before_target_date_for_index} to {target_date}."))
                    continue
                df_index.index = pd.to_datetime(df_index.index).date
                data_target_date_rows = df_index[df_index.index == target_date]
                data_day_before_rows = df_index[df_index.index == day_before_target_date_for_index]
                if data_target_date_rows.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} on target date {target_date}."))
                    continue
                current_data = data_target_date_rows.iloc[0]
                prev_close_val = data_day_before_rows.iloc[0]['Close'] if not data_day_before_rows.empty else None
                MarketIndex.objects.create(
                    market_name=market_readable_name, date=target_date, close_price=current_data['Close'],
                    previous_day_close_price=prev_close_val, volume=current_data.get('Volume'),
                    trade_value=current_data.get('Amount'),
                )
                self.stdout.write(self.style.SUCCESS(f"Created {market_readable_name} index data for {target_date}."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error fetching {market_readable_name} index: {e}"))
            time.sleep(0.5)

        # 2. 전체 상장 종목의 최신 펀더멘털 정보 일괄 가져오기
        self.stdout.write("Fetching latest fundamental data for all KRX stocks (Marcap, PER, PBR)...")
        df_all_fundamentals = pd.DataFrame()
        try:
            df_all_fundamentals = fdr.StockListing('KRX-DESC') 
            if 'Code' in df_all_fundamentals.columns: df_all_fundamentals.rename(columns={'Code': 'Symbol'}, inplace=True)
            if 'Symbol' in df_all_fundamentals.columns:
                df_all_fundamentals.set_index('Symbol', inplace=True)
                self.stdout.write(self.style.SUCCESS(f"Fetched fundamental data for {len(df_all_fundamentals)} stocks."))
            else:
                self.stderr.write(self.style.ERROR("Could not find 'Symbol' column in fundamental data. Fundamentals will be missing."))
                df_all_fundamentals = pd.DataFrame() # 빈 DataFrame으로 설정
        except Exception as e_fund:
            self.stderr.write(self.style.ERROR(f"Error fetching KRX fundamental data: {e_fund}"))
            df_all_fundamentals = pd.DataFrame() 

        # 3. 개별 종목 데이터 처리 (가격, 투자자, 펀더멘털)
        markets_to_list = ['KOSPI', 'KOSDAQ']
        for market in markets_to_list:
            self.stdout.write(f"\nProcessing {market} stocks up to {target_date}...")
            try:
                stocks_df_listing = fdr.StockListing(market)
                if 'Code' in stocks_df_listing.columns: stocks_df_listing.rename(columns={'Code': 'Symbol'}, inplace=True)
                if 'Symbol' not in stocks_df_listing.columns or 'Name' not in stocks_df_listing.columns:
                    self.stderr.write(self.style.ERROR(f"Stock listing for {market} missing 'Symbol' or 'Name'. Skipping market."))
                    continue
                
                stocks_df_listing = stocks_df_listing[stocks_df_listing['Symbol'].notna() & stocks_df_listing['Name'].notna()]
                total_stocks_in_market = len(stocks_df_listing)
                self.stdout.write(f"Found {total_stocks_in_market} stocks in {market}. Starting data update/backfill...")

                processed_this_run_count, skipped_this_run_count, error_this_run_count = 0, 0, 0

                for idx, row_listing in stocks_df_listing.iterrows():
                    stock_code = row_listing['Symbol']
                    stock_name = row_listing['Name']
                    
                    # 현재 진행 상황 로그 (예: 20개 종목마다)
                    if (idx + 1) % 20 == 0:
                        self.stdout.write(f"  [{market}] Checked {(idx + 1)}/{total_stocks_in_market} stocks...")

                    target_date_exists_in_db = StockPrice.objects.filter(stock_code=stock_code, date=target_date).exists()
                    num_records_in_db = StockPrice.objects.filter(stock_code=stock_code).count()

                    needs_fetch = False
                    fetch_start_date = None
                    fetch_end_date = target_date # 항상 target_date까지 가져오는 것을 목표로 함

                    if not target_date_exists_in_db or num_records_in_db < self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY:
                        needs_fetch = True
                        # 데이터가 아예 없거나, 부족하거나, 최신이 아니면 충분한 기간을 가져옴
                        fetch_start_date = self.get_n_trading_days_before(target_date, self.DAYS_TO_FETCH_FOR_BACKFILL)
                        self.stdout.write(f"    Fetching/Backfilling for {stock_name}({stock_code}): {fetch_start_date} to {fetch_end_date} (DB records: {num_records_in_db})")
                    else:
                        self.stdout.write(f"    {stock_name}({stock_code}) is up-to-date with sufficient history ({num_records_in_db} records). Skipping fetch.")
                        skipped_this_run_count +=1
                        continue
                    
                    if needs_fetch and fetch_start_date:
                        try:
                            df_new_data = fdr.DataReader(stock_code, start=fetch_start_date, end=fetch_end_date, data_source='naver')
                            if df_new_data.empty:
                                self.stdout.write(self.style.WARNING(f"      No data from FDR for {stock_name} in range {fetch_start_date}-{fetch_end_date}."))
                                error_this_run_count +=1
                                continue
                            
                            df_new_data.index = pd.to_datetime(df_new_data.index).date
                            df_new_data.sort_index(inplace=True) # 날짜순 정렬 보장

                            saved_count_for_stock = 0
                            for current_iter_date, row_data in df_new_data.iterrows():
                                if StockPrice.objects.filter(stock_code=stock_code, date=current_iter_date).exists():
                                    continue # 이미 DB에 있는 날짜는 건너<0xEB><0><0xA9>

                                # 이전 거래일 종가 계산
                                prev_close_val = None
                                current_df_idx = df_new_data.index.get_loc(current_iter_date)
                                if current_df_idx > 0: # 현재 가져온 데이터프레임 내에 이전 날짜가 있는 경우
                                    prev_row_date_in_df = df_new_data.index[current_df_idx - 1]
                                    prev_close_val = df_new_data.loc[prev_row_date_in_df, 'Close']
                                else: # 현재 가져온 데이터프레임의 첫 번째 날짜인 경우, DB에서 찾아야 함
                                    prev_trading_day_of_first_fetched = self.get_previous_trading_day(current_iter_date, 1)
                                    prev_day_record_in_db = StockPrice.objects.filter(stock_code=stock_code, date=prev_trading_day_of_first_fetched).first()
                                    if prev_day_record_in_db:
                                        prev_close_val = prev_day_record_in_db.close_price
                                
                                # 펀더멘털 정보 (스크립트 실행 시점의 최신 값 사용)
                                market_cap_val, per_val, pbr_val = None, None, None
                                if not df_all_fundamentals.empty and stock_code in df_all_fundamentals.index:
                                    fund_data_row = df_all_fundamentals.loc[stock_code]
                                    market_cap_val = fund_data_row.get('Marcap')
                                    per_val = fund_data_row.get('PER')
                                    pbr_val = fund_data_row.get('PBR')

                                StockPrice.objects.create(
                                    stock_code=stock_code, date=current_iter_date, stock_name=stock_name, market_name=market,
                                    open_price=row_data.get('Open'), high_price=row_data.get('High'),
                                    low_price=row_data.get('Low'), close_price=row_data['Close'],
                                    previous_day_close_price=prev_close_val, volume=row_data.get('Volume'),
                                    trade_value=row_data.get('거래대금'),
                                    indi_volume=row_data.get('개인') if pd.notna(row_data.get('개인')) else None,
                                    foreign_volume=row_data.get('외국인') if pd.notna(row_data.get('외국인')) else None,
                                    organ_volume=row_data.get('기관') if pd.notna(row_data.get('기관')) else None,
                                    market_cap=market_cap_val if pd.notna(market_cap_val) else None,
                                    per=per_val if pd.notna(per_val) else None,
                                    pbr=pbr_val if pd.notna(pbr_val) else None,
                                )
                                saved_count_for_stock += 1
                            
                            if saved_count_for_stock > 0:
                                self.stdout.write(self.style.SUCCESS(f"      Saved {saved_count_for_stock} new daily records for {stock_name}."))
                                processed_this_run_count +=1
                            elif not target_date_exists_in_db : # 가져왔는데 저장할게 없고, 타겟날짜도 없었으면 문제
                                self.stdout.write(self.style.WARNING(f"      Fetched data for {stock_name} but no new records were saved (possibly all existed)."))
                                error_this_run_count +=1


                        except Exception as e_stock_fetch:
                            self.stderr.write(self.style.ERROR(f"    Error processing/fetching data for {stock_name} ({stock_code}): {e_stock_fetch}"))
                            # traceback.print_exc() # 디버깅 시 상세 오류
                            error_this_run_count +=1
                        time.sleep(0.1) # FDR 호출 간격
                
                self.stdout.write(self.style.SUCCESS(f"Finished processing {market}. Total stocks: {total_stocks_in_market}, Updated/Backfilled: {processed_this_run_count}, Skipped (up-to-date): {skipped_this_run_count}, Errors: {error_this_run_count}"))
            except Exception as e_market_listing:
                self.stderr.write(self.style.ERROR(f"Error fetching stock list for {market}: {e_market_listing}"))
        
        self.stdout.write(self.style.SUCCESS(f'\nDaily data update/backfill process finished at {timezone.now()}.'))

