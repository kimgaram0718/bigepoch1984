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
    help = 'Fetches and stores the previous trading day\'s market, stock data (including investor data and fundamentals like Marcap, PER, PBR) from FinanceDataReader.'

    def get_previous_trading_day(self, date_reference, days_offset=1):
        if isinstance(date_reference, datetime):
            current_check_date = date_reference.date()
        elif isinstance(date_reference, date):
            current_check_date = date_reference
        else:
            raise TypeError("date_reference must be a datetime.datetime or datetime.date object.")
        kr_holidays = holidays.KR(years=list(set([current_check_date.year -1, current_check_date.year, current_check_date.year + 1])))
        trading_days_found = 0
        temp_date = current_check_date
        while trading_days_found < days_offset:
            temp_date -= timedelta(days=1)
            if temp_date.weekday() < 5 and temp_date not in kr_holidays:
                trading_days_found += 1
        return temp_date

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting daily market, stock, and fundamental data update...'))

        today_dt = timezone.now()
        target_date = self.get_previous_trading_day(today_dt, 1)
        day_before_target_date = self.get_previous_trading_day(target_date, 1)

        self.stdout.write(f"Target date for data fetching: {target_date}")
        self.stdout.write(f"Day before target date (for change calculation): {day_before_target_date}")

        # 1. 시장 지수 업데이트 (KOSPI, KOSDAQ) - 기존 로직 유지
        indices_to_fetch = {'KOSPI': 'KS11', 'KOSDAQ': 'KQ11'}
        for market_readable_name, fdr_ticker in indices_to_fetch.items():
            if MarketIndex.objects.filter(market_name=market_readable_name, date=target_date).exists():
                self.stdout.write(self.style.SUCCESS(f"{market_readable_name} index data for {target_date} already exists. Skipping."))
                continue
            self.stdout.write(f"Fetching data for {market_readable_name} index ({fdr_ticker})...")
            try:
                df_index = fdr.DataReader(fdr_ticker, start=day_before_target_date, end=target_date)
                if df_index.empty:
                    self.stdout.write(self.style.WARNING(f"No data returned for {market_readable_name} for dates {day_before_target_date} to {target_date}."))
                    continue
                df_index.index = pd.to_datetime(df_index.index).date
                data_target_date_rows = df_index[df_index.index == target_date]
                data_day_before_rows = df_index[df_index.index == day_before_target_date]
                if data_target_date_rows.empty:
                    self.stdout.write(self.style.WARNING(f"No data for {market_readable_name} on target date {target_date} in the fetched range."))
                    continue
                current_data = data_target_date_rows.iloc[0]
                prev_close_val = data_day_before_rows.iloc[0]['Close'] if not data_day_before_rows.empty else None
                MarketIndex.objects.create(
                    market_name=market_readable_name, date=target_date, close_price=current_data['Close'],
                    previous_day_close_price=prev_close_val, volume=current_data.get('Volume'),
                    trade_value=current_data.get('Amount'),
                )
                self.stdout.write(self.style.SUCCESS(f"Successfully created {market_readable_name} index data for {target_date}."))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error fetching {market_readable_name} index: {e}"))
                self.stderr.write(traceback.format_exc())
            time.sleep(0.5) # API 호출 간격

        # 2. 전체 상장 종목의 최신 펀더멘털 정보 가져오기 (StockListing 활용)
        self.stdout.write("Fetching latest fundamental data for all KRX stocks (Marcap, PER, PBR)...")
        df_all_fundamentals = pd.DataFrame()
        try:
            # KRX-DESC는 시가총액 내림차순, 다른 정보도 포함
            # KRX, KOSPI, KOSDAQ, KONEX 등 원하는 시장 지정 가능
            df_all_fundamentals = fdr.StockListing('KRX-DESC') # 또는 'KRX' 사용 가능
            if 'Code' in df_all_fundamentals.columns: # 컬럼명 통일
                 df_all_fundamentals.rename(columns={'Code': 'Symbol'}, inplace=True)
            df_all_fundamentals.set_index('Symbol', inplace=True) # 종목코드를 인덱스로
            self.stdout.write(self.style.SUCCESS(f"Successfully fetched fundamental data for {len(df_all_fundamentals)} stocks."))
        except Exception as e_fund:
            self.stderr.write(self.style.ERROR(f"Error fetching KRX fundamental data: {e_fund}"))
            self.stderr.write(traceback.format_exc())
            # 펀더멘털 데이터 가져오기 실패 시, 이후 로직에서 해당 값들은 None으로 처리됨

        # 3. 개별 종목 가격, 투자자 정보, 펀더멘털 정보 업데이트
        markets_to_list = ['KOSPI', 'KOSDAQ']
        for market in markets_to_list:
            self.stdout.write(f"Processing {market} stocks for {target_date}...")
            try:
                stocks_df_listing = fdr.StockListing(market)
                if 'Code' in stocks_df_listing.columns: stocks_df_listing.rename(columns={'Code': 'Symbol'}, inplace=True)
                if 'Symbol' not in stocks_df_listing.columns or 'Name' not in stocks_df_listing.columns:
                    self.stderr.write(self.style.ERROR(f"Stock listing for {market} is missing 'Symbol' or 'Name' column."))
                    continue
                
                stocks_df_listing = stocks_df_listing[stocks_df_listing['Symbol'].notna() & stocks_df_listing['Name'].notna()]
                total_stocks = len(stocks_df_listing)
                self.stdout.write(f"Found {total_stocks} stocks in {market}. Processing prices, investor data, and fundamentals...")

                processed_count, skipped_count, error_count = 0, 0, 0
                for index, row_listing in stocks_df_listing.iterrows():
                    stock_code = row_listing['Symbol']
                    stock_name = row_listing['Name']
                    
                    if StockPrice.objects.filter(stock_code=stock_code, date=target_date).exists():
                        skipped_count += 1
                        if (skipped_count + processed_count + error_count) % 200 == 0:
                             self.stdout.write(f"[{market}] Progress: {(skipped_count + processed_count + error_count)}/{total_stocks} (S:{skipped_count}, P:{processed_count}, E:{error_count})")
                        continue
                    
                    current_progress = skipped_count + processed_count + error_count
                    if current_progress > 0 and current_progress % (max(1, total_stocks // 20)) == 0:
                        percent = int((current_progress / total_stocks) * 100)
                        self.stdout.write(f"[{market}] Progress: {percent}% ({current_progress}/{total_stocks})")

                    try:
                        df_stock_price_investor = fdr.DataReader(stock_code, start=day_before_target_date, end=target_date, data_source='naver')
                        if df_stock_price_investor.empty: continue
                        
                        df_stock_price_investor.index = pd.to_datetime(df_stock_price_investor.index).date
                        data_target_date_stock_rows = df_stock_price_investor[df_stock_price_investor.index == target_date]
                        data_day_before_stock_rows = df_stock_price_investor[df_stock_price_investor.index == day_before_target_date]

                        if data_target_date_stock_rows.empty: continue
                        
                        current_stock_data = data_target_date_stock_rows.iloc[0]
                        prev_close_val_stock = data_day_before_stock_rows.iloc[0]['Close'] if not data_day_before_stock_rows.empty else None

                        indi_vol = current_stock_data.get('개인'); foreign_vol = current_stock_data.get('외국인'); organ_vol = current_stock_data.get('기관')
                        
                        # 펀더멘털 정보 가져오기 (df_all_fundamentals 에서)
                        market_cap_val, per_val, pbr_val = None, None, None
                        if not df_all_fundamentals.empty and stock_code in df_all_fundamentals.index:
                            fund_data_row = df_all_fundamentals.loc[stock_code]
                            market_cap_val = fund_data_row.get('Marcap')
                            per_val = fund_data_row.get('PER')
                            pbr_val = fund_data_row.get('PBR')
                        
                        StockPrice.objects.create(
                            stock_code=stock_code, date=target_date, stock_name=stock_name, market_name=market,
                            open_price=current_stock_data.get('Open'), high_price=current_stock_data.get('High'),
                            low_price=current_stock_data.get('Low'), close_price=current_stock_data['Close'],
                            previous_day_close_price=prev_close_val_stock, volume=current_stock_data.get('Volume'),
                            trade_value=current_stock_data.get('거래대금'),
                            indi_volume=indi_vol if pd.notna(indi_vol) else None,
                            foreign_volume=foreign_vol if pd.notna(foreign_vol) else None,
                            organ_volume=organ_vol if pd.notna(organ_vol) else None,
                            market_cap=market_cap_val if pd.notna(market_cap_val) else None,
                            per=per_val if pd.notna(per_val) else None,
                            pbr=pbr_val if pd.notna(pbr_val) else None,
                        )
                        processed_count += 1
                    except Exception as e_stock:
                        self.stderr.write(self.style.ERROR(f"  Error processing {stock_name} ({stock_code}): {e_stock}"))
                        error_count += 1
                    
                    if (processed_count + skipped_count + error_count) % 50 == 0:
                         self.stdout.write(f"[{market}] Checked {processed_count + skipped_count + error_count}/{total_stocks}. (New: {processed_count}, Skip: {skipped_count}, Err: {error_count})")
                    time.sleep(0.1) # API 호출 간격
                
                self.stdout.write(self.style.SUCCESS(f"Finished {market}. Total: {total_stocks}, New: {processed_count}, Skip: {skipped_count}, Err: {error_count}"))
            except Exception as e_market_list:
                self.stderr.write(self.style.ERROR(f"Error fetching stock list for {market}: {e_market_list}"))
                self.stderr.write(traceback.format_exc())
        
        self.stdout.write(self.style.SUCCESS('Daily data update (including fundamentals) finished.'))
