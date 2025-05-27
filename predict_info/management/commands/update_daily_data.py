# predict_info/management/commands/update_daily_data.py
import pandas as pd
import FinanceDataReader as fdr
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import timedelta, datetime, date as date_type
from predict_info.models import StockPrice, MarketIndex 
from predict_info.utils import ( 
    calculate_all_features,
    get_market_macro_data, 
    get_kr_holidays,
    add_fundamental_indicator_features,
    PANDAS_TA_AVAILABLE
)
import time
import traceback
import numpy as np
from django.db import models as django_db_models
from django.db import transaction
from django.core.exceptions import FieldDoesNotExist

try:
    from pykrx import stock as pykrx_stock
    PYKRX_AVAILABLE = True
    print("[INFO][update_daily_data.py] pykrx 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PYKRX_AVAILABLE = False
    print("[WARNING][update_daily_data.py] 'pykrx' 라이브러리를 찾을 수 없습니다. 펀더멘털 및 일부 투자자 데이터가 누락될 수 있습니다.")


class Command(BaseCommand):
    help = ('Fetches and stores daily stock data (OHLCV, Investor Trends, Fundamentals, TA features) '
            'and Market Index data into the database.')

    MIN_RECORDS_FOR_SUFFICIENT_HISTORY = 120 
    DAYS_TO_FETCH_FOR_BACKFILL = 300       
    DAYS_TO_FETCH_OHLCV_BASE = DAYS_TO_FETCH_FOR_BACKFILL + 100


    def get_previous_trading_day(self, reference_date, kr_holidays_list, days_offset=1):
        if not isinstance(reference_date, date_type):
            try: reference_date = pd.to_datetime(reference_date).date()
            except: raise TypeError("reference_date must be a datetime.date or convertible to it.")

        current_check_date = reference_date
        trading_days_found = 0

        if days_offset == 0: 
            while current_check_date.weekday() >= 5 or current_check_date in kr_holidays_list:
                current_check_date -= timedelta(days=1)
            return current_check_date

        while trading_days_found < days_offset:
            current_check_date -= timedelta(days=1)
            if current_check_date.weekday() < 5 and current_check_date not in kr_holidays_list:
                trading_days_found += 1
        return current_check_date

    def fetch_pykrx_fundamentals(self, stock_code, start_date_str, end_date_str):
        if not PYKRX_AVAILABLE:
            fund_cols = ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'CashDPS', 'ROE']
            return pd.DataFrame(columns=fund_cols)

        df_fund_combined = pd.DataFrame()
        try:
            df_cap_raw = pykrx_stock.get_market_cap_by_date(start_date_str, end_date_str, stock_code)
            if not df_cap_raw.empty:
                df_cap = df_cap_raw.reset_index()
                df_cap.rename(columns={'날짜': 'Date', '시가총액': 'MarketCap', '상장주식수': 'ListedShares'}, inplace=True)
                if 'Date' in df_cap.columns:
                    df_cap['Date'] = pd.to_datetime(df_cap['Date']).dt.date
                    df_fund_combined = df_cap[['Date', 'MarketCap', 'ListedShares']]

            df_fundamental_raw = pykrx_stock.get_market_fundamental_by_date(start_date_str, end_date_str, stock_code)
            if not df_fundamental_raw.empty:
                df_pbr_per_eps = df_fundamental_raw.reset_index().rename(columns={'날짜': 'Date', 'DIV': 'CashDPS'})
                if 'Date' in df_pbr_per_eps.columns:
                    df_pbr_per_eps['Date'] = pd.to_datetime(df_pbr_per_eps['Date']).dt.date
                
                    fund_cols_to_merge = ['Date']
                    pykrx_to_model_fund_map = {'PBR':'PBR', 'PER':'PER', 'EPS':'EPS', 'BPS':'BPS', 'CashDPS':'CashDPS', 'DPS':'DPS'}
                    for pykrx_col, model_col in pykrx_to_model_fund_map.items():
                        if pykrx_col in df_pbr_per_eps.columns:
                            fund_cols_to_merge.append(pykrx_col) 
                    
                    if len(fund_cols_to_merge) > 1:
                        if not df_fund_combined.empty:
                            df_fund_combined = pd.merge(df_fund_combined, df_pbr_per_eps[fund_cols_to_merge], on='Date', how='outer')
                        else:
                            df_fund_combined = df_pbr_per_eps[fund_cols_to_merge]
            
            if not df_fund_combined.empty:
                final_fund_rename_map = {'PBR':'PBR', 'PER':'PER', 'EPS':'EPS', 'BPS':'BPS', 'CashDPS':'CashDPS', 'DPS':'DPS', 'MarketCap':'MarketCap'}
                df_fund_combined.rename(columns={k:v for k,v in final_fund_rename_map.items() if k in df_fund_combined.columns}, inplace=True)

                df_fund_combined.set_index('Date', inplace=True)
                
                for col in ['MarketCap', 'ListedShares', 'PBR', 'PER', 'EPS', 'BPS', 'CashDPS', 'DPS']:
                    if col in df_fund_combined.columns:
                        df_fund_combined[col] = pd.to_numeric(df_fund_combined[col], errors='coerce')
                        df_fund_combined[col] = df_fund_combined[col].ffill().bfill()
            
                if 'EPS' in df_fund_combined.columns and 'BPS' in df_fund_combined.columns:
                    df_fund_combined['ROE'] = np.where(
                        (df_fund_combined['BPS'].notna()) & (df_fund_combined['BPS'] != 0),
                        (df_fund_combined['EPS'] / df_fund_combined['BPS']) * 100,
                        np.nan 
                    )
                    df_fund_combined['ROE'] = df_fund_combined['ROE'].ffill().bfill()
                else:
                    df_fund_combined['ROE'] = np.nan
            return df_fund_combined.reset_index() 
        except Exception as e:
            self.stderr.write(self.style.WARNING(f"      pykrx fundamental fetch error for {stock_code} ({start_date_str}~{end_date_str}): {e}\n{traceback.format_exc()}"))
            fund_cols = ['Date', 'MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'CashDPS', 'ROE'] 
            return pd.DataFrame(columns=fund_cols)

    def save_market_index_data(self, market_name_param, df_market_data_full, kr_holidays_list_for_index):
        if df_market_data_full is None or df_market_data_full.empty:
            self.stdout.write(self.style.NOTICE(f"    No data to save for MarketIndex: {market_name_param}"))
            return 0

        saved_count = 0
        
        if not all(isinstance(i, date_type) for i in df_market_data_full.index if pd.notna(i)):
            try:
                df_market_data_full.index = pd.to_datetime(df_market_data_full.index, errors='coerce').date
            except Exception as e_idx:
                self.stderr.write(self.style.ERROR(f"MarketIndex data for {market_name_param}: Date index conversion failed ({e_idx}). Skipping save."))
                return 0
        
        df_to_save = df_market_data_full.copy()
        
        close_col_name_in_df = '종가' 
        volume_col_name_in_df = '거래량'
        value_col_name_in_df = '거래대금'   
        
        if close_col_name_in_df not in df_to_save.columns:
            self.stderr.write(self.style.ERROR(f"MarketIndex data for {market_name_param}: Missing '{close_col_name_in_df}' column. pykrx may have changed column names. Available: {df_to_save.columns.tolist()}"))
            return 0

        with transaction.atomic():
            for date_idx, row_data in df_to_save.iterrows():
                if not isinstance(date_idx, date_type): continue

                current_close_price = row_data.get(close_col_name_in_df)
                if pd.isna(current_close_price): continue 

                defaults = {
                    'close_price': float(current_close_price),
                    'volume': float(row_data.get(volume_col_name_in_df)) if volume_col_name_in_df in row_data and pd.notna(row_data.get(volume_col_name_in_df)) else None,
                    'trade_value': float(row_data.get(value_col_name_in_df)) if value_col_name_in_df in row_data and pd.notna(row_data.get(value_col_name_in_df)) else None,
                }
                
                prev_day_for_lookup = self.get_previous_trading_day(date_idx, kr_holidays_list_for_index, 1)
                prev_market_record = MarketIndex.objects.filter(market_name=market_name_param, date=prev_day_for_lookup).first()
                
                if prev_market_record and prev_market_record.close_price is not None:
                    defaults['previous_day_close_price'] = prev_market_record.close_price
                elif prev_day_for_lookup in df_to_save.index and pd.notna(df_to_save.loc[prev_day_for_lookup, close_col_name_in_df]):
                     defaults['previous_day_close_price'] = float(df_to_save.loc[prev_day_for_lookup, close_col_name_in_df])
                else:
                    defaults['previous_day_close_price'] = None
                
                try:
                    obj, created = MarketIndex.objects.update_or_create(
                        market_name=market_name_param,
                        date=date_idx,
                        defaults=defaults
                    )
                    if created:
                        saved_count += 1
                except Exception as e_db_save:
                    self.stderr.write(self.style.ERROR(f"  MarketIndex DB Save Error for {market_name_param} on {date_idx}: {e_db_save}"))
        
        if saved_count > 0:
            self.stdout.write(self.style.SUCCESS(f"    Successfully saved/updated {saved_count} records for MarketIndex: {market_name_param}."))
        return saved_count


    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"Starting daily data update for DB (min history: {self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY} days)..."))

        today_dt = timezone.now().date()
        current_year = today_dt.year
        all_years_for_holidays = list(range(current_year - 11, current_year + 3)) 
        kr_holidays_list_global = get_kr_holidays(all_years_for_holidays)
        
        target_date_for_data = self.get_previous_trading_day(today_dt, kr_holidays_list_global, 1)
        self.stdout.write(f"Target date for data processing: {target_date_for_data}")

        overall_fetch_start_date_str_pykrx = (target_date_for_data - timedelta(days=10*365)).strftime("%Y%m%d")
        target_date_str_pykrx = target_date_for_data.strftime("%Y%m%d")

        if PYKRX_AVAILABLE:
            self.stdout.write(f"Pre-fetching KOSPI market data from {overall_fetch_start_date_str_pykrx} to {target_date_str_pykrx} for MarketIndex model using pykrx...")
            try:
                df_kospi_index_raw = pykrx_stock.get_index_ohlcv_by_date(overall_fetch_start_date_str_pykrx, target_date_str_pykrx, "1001") 
                if df_kospi_index_raw is not None and not df_kospi_index_raw.empty:
                    self.save_market_index_data('KOSPI', df_kospi_index_raw, kr_holidays_list_global)
            except Exception as e_pykrx_idx:
                self.stderr.write(self.style.ERROR(f"Error fetching KOSPI index data from pykrx: {e_pykrx_idx}"))
        else:
            self.stdout.write(self.style.WARNING("pykrx not available, skipping KOSPI index data fetch for MarketIndex."))

        if PYKRX_AVAILABLE:
            self.stdout.write(f"Pre-fetching KOSDAQ market data from {overall_fetch_start_date_str_pykrx} to {target_date_str_pykrx} for MarketIndex model using pykrx...")
            try:
                df_kosdaq_index_raw = pykrx_stock.get_index_ohlcv_by_date(overall_fetch_start_date_str_pykrx, target_date_str_pykrx, "2001") 
                if df_kosdaq_index_raw is not None and not df_kosdaq_index_raw.empty:
                    self.save_market_index_data('KOSDAQ', df_kosdaq_index_raw, kr_holidays_list_global)
            except Exception as e_pykrx_idx:
                 self.stderr.write(self.style.ERROR(f"Error fetching KOSDAQ index data from pykrx: {e_pykrx_idx}"))
        else:
            self.stdout.write(self.style.WARNING("pykrx not available, skipping KOSDAQ index data fetch for MarketIndex."))

        overall_fetch_start_date_str_util = (target_date_for_data - timedelta(days=10*365)).strftime("%Y%m%d")
        target_date_str_util = target_date_for_data.strftime("%Y%m%d")

        self.stdout.write(f"Pre-fetching KOSPI market & macro data (for StockPrice) from {overall_fetch_start_date_str_util} to {target_date_str_util}...")
        df_kospi_macro_full_for_stockprice = get_market_macro_data(overall_fetch_start_date_str_util, target_date_str_util, market_name='KOSPI', other_market_name_for_index='KOSDAQ')
        
        self.stdout.write(f"Pre-fetching KOSDAQ market & macro data (for StockPrice) from {overall_fetch_start_date_str_util} to {target_date_str_util}...")
        df_kosdaq_macro_full_for_stockprice = get_market_macro_data(overall_fetch_start_date_str_util, target_date_str_util, market_name='KOSDAQ', other_market_name_for_index='KOSPI')


        krx_listing_df = fdr.StockListing('KRX')
        krx_listing_df = krx_listing_df[krx_listing_df['Market'].isin(['KOSPI', 'KOSDAQ'])]
        
        total_stocks_to_process = len(krx_listing_df)
        self.stdout.write(f"Found {total_stocks_to_process} stocks in KOSPI/KOSDAQ. Starting data update/backfill for StockPrice DB...")
        processed_count, skipped_count, error_count = 0, 0, 0

        for idx, row_listing in krx_listing_df.iterrows():
            stock_code = row_listing['Code']
            stock_name = row_listing['Name']
            market_name_str = row_listing['Market'].upper() # 'KOSPI' 또는 'KOSDAQ' 문자열

            if (idx + 1) % 20 == 0: 
                self.stdout.write(f"   [{market_name_str}] Checked {(idx + 1)}/{total_stocks_to_process} stocks... (OK: {processed_count}, Skip: {skipped_count}, Err: {error_count})")

            last_stock_date_in_db_obj = StockPrice.objects.filter(stock_code=stock_code).order_by('-date').first()
            num_records_in_db = StockPrice.objects.filter(stock_code=stock_code).count()

            fetch_start_date_stock_specific_dt = None
            if last_stock_date_in_db_obj: 
                if last_stock_date_in_db_obj.date >= target_date_for_data and num_records_in_db >= self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY:
                    skipped_count += 1
                    continue
                
                fetch_start_date_stock_specific_dt = last_stock_date_in_db_obj.date + timedelta(days=1)
                
                if num_records_in_db < self.MIN_RECORDS_FOR_SUFFICIENT_HISTORY or \
                   (target_date_for_data - fetch_start_date_stock_specific_dt).days < self.DAYS_TO_FETCH_FOR_BACKFILL:
                    fetch_start_date_stock_specific_dt = self.get_previous_trading_day(target_date_for_data, kr_holidays_list_global, self.DAYS_TO_FETCH_OHLCV_BASE)
            else: 
                fetch_start_date_stock_specific_dt = self.get_previous_trading_day(target_date_for_data, kr_holidays_list_global, self.DAYS_TO_FETCH_OHLCV_BASE)

            if fetch_start_date_stock_specific_dt > target_date_for_data:
                skipped_count += 1
                continue

            fetch_start_date_stock_specific_str_fdr = fetch_start_date_stock_specific_dt.strftime("%Y-%m-%d") 
            target_date_stock_specific_str_fdr = target_date_for_data.strftime("%Y-%m-%d")
            
            fetch_start_date_stock_specific_str_pykrx = fetch_start_date_stock_specific_dt.strftime("%Y%m%d")
            target_date_stock_specific_str_pykrx = target_date_for_data.strftime("%Y%m%d")


            try:
                df_stock_ohlcv_fdr = fdr.DataReader(stock_code, fetch_start_date_stock_specific_str_fdr, target_date_stock_specific_str_fdr)
                if df_stock_ohlcv_fdr.empty:
                    self.stdout.write(self.style.NOTICE(f"    No data from FDR for {stock_name} ({stock_code}) for period {fetch_start_date_stock_specific_str_fdr}-{target_date_stock_specific_str_fdr}. Skipping."))
                    skipped_count +=1
                    continue

                df_stock_ohlcv_fdr.index = pd.to_datetime(df_stock_ohlcv_fdr.index).date
                df_stock_ohlcv_fdr.sort_index(inplace=True)

                stock_df_ohlcv_arg = df_stock_ohlcv_fdr[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if 'Change' in df_stock_ohlcv_fdr.columns: 
                    stock_df_ohlcv_arg['Change'] = pd.to_numeric(df_stock_ohlcv_fdr['Change'], errors='coerce')
                else: 
                    stock_df_ohlcv_arg['Change'] = stock_df_ohlcv_fdr['Close'].pct_change(fill_method=None) 

                investor_df_arg = pd.DataFrame(index=df_stock_ohlcv_fdr.index.copy())
                
                possible_investor_cols_fdr = {
                    'Indi': ['개인', 'Individual', 'Retail'], 
                    'Foreign': ['외국인', 'Foreigner', 'Forn'], 
                    'Organ': ['기관', '기관계', 'Institution', 'Inst'] 
                }
                
                fdr_investor_data_found_at_least_one_type = False
                for model_col_prefix, fdr_name_options in possible_investor_cols_fdr.items():
                    found_this_type = False
                    for fdr_name in fdr_name_options:
                        if fdr_name in df_stock_ohlcv_fdr.columns:
                            investor_df_arg[model_col_prefix] = pd.to_numeric(df_stock_ohlcv_fdr[fdr_name], errors='coerce')
                            fdr_investor_data_found_at_least_one_type = True 
                            found_this_type = True
                            break 
                    if not found_this_type: 
                         investor_df_arg[model_col_prefix] = np.nan
                
                should_try_pykrx_investor = PYKRX_AVAILABLE and \
                                           (not fdr_investor_data_found_at_least_one_type or \
                                            (investor_df_arg['Indi'].isnull().all() if 'Indi' in investor_df_arg else True) or \
                                            (investor_df_arg['Foreign'].isnull().all() if 'Foreign' in investor_df_arg else True) or \
                                            (investor_df_arg['Organ'].isnull().all() if 'Organ' in investor_df_arg else True) )


                if should_try_pykrx_investor:
                    self.stdout.write(self.style.NOTICE(f"    FDR for {stock_name} ({stock_code}) did not provide sufficient investor data. Trying pykrx for trading value..."))
                    try:
                        df_investor_pykrx_raw = pykrx_stock.get_market_trading_value_by_date(
                            fromdate=fetch_start_date_stock_specific_str_pykrx, 
                            todate=target_date_stock_specific_str_pykrx, 
                            ticker=stock_code,
                            on='순매수' 
                        )
                        if not df_investor_pykrx_raw.empty:
                            df_investor_pykrx_temp = df_investor_pykrx_raw.reset_index()
                            if '날짜' in df_investor_pykrx_temp.columns:
                                df_investor_pykrx_temp.rename(columns={'날짜':'Date'}, inplace=True)
                            elif df_investor_pykrx_temp.columns[0] != 'Date' and pd.api.types.is_datetime64_any_dtype(df_investor_pykrx_temp.iloc[:, 0]):
                                df_investor_pykrx_temp.rename(columns={df_investor_pykrx_temp.columns[0]:'Date'}, inplace=True) 
                            
                            df_investor_pykrx_temp['Date'] = pd.to_datetime(df_investor_pykrx_temp['Date']).dt.date
                            
                            # Prepare investor_df_arg_for_merge to ensure 'Date' column exists and is of correct type
                            if investor_df_arg.index.name == 'Date' and isinstance(investor_df_arg.index, pd.DatetimeIndex):
                                investor_df_arg_for_merge = investor_df_arg.reset_index()
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            elif 'Date' in investor_df_arg.columns and pd.api.types.is_datetime64_any_dtype(investor_df_arg['Date']):
                                investor_df_arg_for_merge = investor_df_arg.copy()
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            else: # Fallback: create from index if it's date-like, or ensure it has a Date column
                                investor_df_arg_for_merge = investor_df_arg.copy()
                                if isinstance(investor_df_arg.index, pd.DatetimeIndex):
                                     investor_df_arg_for_merge.index = pd.to_datetime(investor_df_arg.index).date
                                     investor_df_arg_for_merge.reset_index(inplace=True) # 'Date' becomes a column
                                     if 'index' in investor_df_arg_for_merge.columns and 'Date' not in investor_df_arg_for_merge.columns:
                                         investor_df_arg_for_merge.rename(columns={'index':'Date'}, inplace=True)
                                elif 'Date' not in investor_df_arg_for_merge.columns:
                                     # If investor_df_arg was from pd.DataFrame(index=df_stock_ohlcv_fdr.index.copy())
                                     # its index is already date objects.
                                     investor_df_arg_for_merge['Date'] = investor_df_arg.index 
                            
                            # Ensure 'Date' column in investor_df_arg_for_merge is of date objects for merging
                            if 'Date' in investor_df_arg_for_merge.columns:
                                investor_df_arg_for_merge['Date'] = pd.to_datetime(investor_df_arg_for_merge['Date']).dt.date
                            else: # This should ideally not be reached if preparation is correct
                                self.stderr.write(self.style.WARNING(f"      'Date' column missing in investor_df_arg_for_merge for {stock_name} before pykrx merge. Attempting to use main OHLCV index."))
                                investor_df_arg_for_merge['Date'] = df_stock_ohlcv_fdr.index


                            target_model_investor_cols = {
                                'Indi': ['개인'],
                                'Foreign': ['외국인합계', '외국인'], # 우선순위: 외국인합계 > 외국인
                                'Organ': ['기관합계', '기관']      # 우선순위: 기관합계 > 기관
                            }

                            for model_c, pykrx_source_options in target_model_investor_cols.items():
                                pykrx_actual_source_col = None
                                for source_opt in pykrx_source_options:
                                    if source_opt in df_investor_pykrx_temp.columns:
                                        pykrx_actual_source_col = source_opt
                                        break 

                                if pykrx_actual_source_col:
                                    df_pykrx_specific_investor = df_investor_pykrx_temp[['Date', pykrx_actual_source_col]].rename(
                                        columns={pykrx_actual_source_col: model_c}
                                    )

                                    if model_c not in investor_df_arg_for_merge.columns:
                                        investor_df_arg_for_merge[model_c] = np.nan

                                    investor_df_arg_for_merge = pd.merge(
                                        investor_df_arg_for_merge,
                                        df_pykrx_specific_investor,
                                        on='Date',
                                        how='left',
                                        suffixes=('_fdr', '_pykrx') 
                                    )
                                    
                                    pykrx_suffixed_col = f"{model_c}_pykrx"
                                    fdr_suffixed_col = f"{model_c}_fdr"
                                    original_col_to_update = fdr_suffixed_col if fdr_suffixed_col in investor_df_arg_for_merge.columns else model_c
                                    
                                    if pykrx_suffixed_col in investor_df_arg_for_merge.columns:
                                        investor_df_arg_for_merge[model_c] = investor_df_arg_for_merge[pykrx_suffixed_col].combine_first(
                                            investor_df_arg_for_merge[original_col_to_update]
                                        )
                                        
                                        cols_to_drop_after_merge = [pykrx_suffixed_col]
                                        if fdr_suffixed_col in investor_df_arg_for_merge.columns and fdr_suffixed_col != model_c:
                                            cols_to_drop_after_merge.append(fdr_suffixed_col)
                                        investor_df_arg_for_merge.drop(columns=cols_to_drop_after_merge, errors='ignore', inplace=True)
                                    elif fdr_suffixed_col in investor_df_arg_for_merge.columns and fdr_suffixed_col != model_c :
                                         investor_df_arg_for_merge.rename(columns={fdr_suffixed_col: model_c}, inplace=True, errors='ignore')
                            
                            if 'Date' in investor_df_arg_for_merge.columns:
                                investor_df_arg = investor_df_arg_for_merge.set_index('Date')
                            else:
                                self.stderr.write(self.style.ERROR(f"      Investor data merge for {stock_name} resulted in missing 'Date' column after pykrx attempt."))
                                # If merge fails to produce 'Date', fall back to original investor_df_arg to prevent crash, though pykrx data will be lost.
                                # This path indicates a more fundamental issue with 'Date' column handling.

                            # Ensure standard investor columns exist in the final DataFrame, even if all NaN
                            for model_col_final in ['Indi', 'Foreign', 'Organ']:
                                if model_col_final not in investor_df_arg.columns:
                                    investor_df_arg[model_col_final] = np.nan

                            self.stdout.write(self.style.SUCCESS(f"    Successfully attempted to update investor data for {stock_name} using pykrx (value based)."))
                        else:
                            self.stdout.write(self.style.NOTICE(f"    pykrx (value based) returned no investor data for {stock_name} ({stock_code})."))
                    except AttributeError as e_attr:
                         self.stderr.write(self.style.ERROR(f"      AttributeError for pykrx investor data for {stock_name}: {e_attr}. pykrx function 'get_market_trading_value_by_date' might not be available in your pykrx version or was misspelled."))
                    except Exception as e_pykrx_inv:
                        self.stderr.write(self.style.ERROR(f"      Error fetching/processing investor data from pykrx for {stock_name}: {e_pykrx_inv}\n{traceback.format_exc()}"))

                
                actual_stock_data_start_date_str = df_stock_ohlcv_fdr.index.min().strftime("%Y%m%d")
                actual_stock_data_end_date_str = df_stock_ohlcv_fdr.index.max().strftime("%Y%m%d")
                
                fundamental_df_from_pykrx = self.fetch_pykrx_fundamentals(stock_code, actual_stock_data_start_date_str, actual_stock_data_end_date_str)
                fundamental_df_arg = pd.DataFrame() 
                if not fundamental_df_from_pykrx.empty and 'Date' in fundamental_df_from_pykrx.columns:
                    fundamental_df_arg = fundamental_df_from_pykrx.set_index('Date') 
                    if isinstance(fundamental_df_arg.index, pd.DatetimeIndex):
                         fundamental_df_arg.index = fundamental_df_arg.index.date

                market_macro_data_df_arg = pd.DataFrame(index=df_stock_ohlcv_fdr.index) 
                current_stock_market_macro_df_full = df_kospi_macro_full_for_stockprice if market_name_str == 'KOSPI' else df_kosdaq_macro_full_for_stockprice
                
                if not current_stock_market_macro_df_full.empty:
                    if not isinstance(current_stock_market_macro_df_full.index, pd.DatetimeIndex) and \
                       not all(isinstance(i, date_type) for i in current_stock_market_macro_df_full.index if pd.notna(i)):
                        current_stock_market_macro_df_full.index = pd.to_datetime(current_stock_market_macro_df_full.index, errors='coerce').date

                    market_macro_data_df_arg = market_macro_data_df_arg.join(current_stock_market_macro_df_full, how='left')
                
                df_features_from_calc = calculate_all_features(
                    stock_df_ohlcv=stock_df_ohlcv_arg,
                    market_macro_data_df=market_macro_data_df_arg,
                    investor_df=investor_df_arg,
                    fundamental_df=fundamental_df_arg, 
                    pandas_ta_available=PANDAS_TA_AVAILABLE
                )
                
                if df_features_from_calc is None or df_features_from_calc.empty:
                    self.stdout.write(self.style.WARNING(f"    No features calculated for {stock_name} ({stock_code}). Skipping DB save for this stock."))
                    skipped_count +=1
                    continue

                df_final_for_db = add_fundamental_indicator_features(df_features_from_calc.copy())
                df_final_for_db.replace([np.inf, -np.inf], np.nan, inplace=True)

                model_to_df_col_map = {
                    'open_price': 'Open', 'high_price': 'High', 'low_price': 'Low', 'close_price': 'Close',
                    'volume': 'Volume', 'change': 'Change',
                    
                    'indi_volume': 'Indi', 'foreign_volume': 'Foreign', 'organ_volume': 'Organ',
                    
                    'market_cap': 'MarketCap', 'per': 'PER', 'pbr': 'PBR',
                    'eps': 'EPS', 'bps': 'BPS', 'dps': 'DPS', 
                    'roe': 'ROE',

                    'MA5': 'MA_5', 'MA10': 'MA_10', 'MA20': 'MA_20', 'MA60': 'MA_60', 'MA120': 'MA_120',
                    'EMA5': 'EMA_5', 'EMA10': 'EMA_10', 'EMA20': 'EMA_20', 'EMA60': 'EMA_60', 'EMA120': 'EMA_120',
                    
                    'BB_Upper': 'BBU_20_2.0', 
                    'BB_Middle': 'BBM_20_2.0',
                    'BB_Lower': 'BBL_20_2.0',
                    'BB_Width': 'BBB_20_2.0', 
                    'BB_PercentB': 'BBP_20_2.0',

                    'MACD': 'MACD_12_26_9',
                    'MACD_Signal': 'MACDs_12_26_9',
                    'MACD_Hist': 'MACDh_12_26_9',

                    'RSI6': 'RSI_6',
                    'RSI14': 'RSI_14',
                    'RSI28': 'RSI_28',

                    'STOCH_K': 'STOCHk_14_3_3',       
                    'STOCH_D': 'STOCHd_14_3_3',       
                    'STOCH_SLOW_K': 'STOCHk_fast_14_3_1', 
                    'STOCH_SLOW_D': 'STOCHd_fast_14_3_1', 

                    'ATR14': 'ATR_14',
                    'ADX14': 'ADX_14',
                    'DMP14': 'DMP_14', 
                    'DMN14': 'DMN_14', 
                    'CCI14': 'CCI_14_0.015',
                    'MFI14': 'MFI_14',
                    'OBV': 'OBV',
                    'WilliamsR14': 'WILLR_14',
                    'Momentum': 'MOM_10',
                    'ROC': 'ROC_10',
                    'TRIX': 'TRIX_14_9',
                    'VR': 'VR_20',
                    'PSY': 'PSL_12',

                    'Market_Index_Close': f"{market_name_str}_Close",
                    'Market_Index_Change': f"{market_name_str}_Change",
                    
                    'USD_KRW_Close': 'USD_KRW_Close',
                    'USD_KRW_Change': 'USD_KRW_Change',

                    'log_close_price': 'Log_Close',
                    'log_volume': 'Log_Volume',
                    
                    'per_is_high': 'PER_is_high', 
                    'per_is_low': 'PER_is_low',
                    'per_is_zero': 'PER_is_zero',
                    'per_is_nan': 'PER_is_nan',
                    'pbr_is_high': 'PBR_is_high',
                    'pbr_is_low': 'PBR_is_low',
                    'pbr_is_zero': 'PBR_is_zero',
                    'pbr_is_nan': 'PBR_is_nan',
                    'market_cap_is_nan': 'MarketCap_is_nan',
                }

                saved_count_for_stock_this_fetch = 0
                with transaction.atomic():
                    if not isinstance(df_final_for_db.index, pd.DatetimeIndex) and \
                       not all(isinstance(i, date_type) for i in df_final_for_db.index if pd.notna(i)):
                        try:
                            df_final_for_db.index = pd.to_datetime(df_final_for_db.index, errors='coerce').date
                        except: 
                             self.stderr.write(self.style.ERROR(f"      Date index conversion failed for {stock_name} ({stock_code}). Skipping DB save."))
                             error_count +=1
                             continue 


                    for date_iter_dt, row_final_data in df_final_for_db.iterrows():
                        if not isinstance(date_iter_dt, date_type): 
                            continue
                        
                        current_iter_date_for_db = date_iter_dt
                        defaults_for_db = {'stock_name': stock_name, 'market_name': market_name_str}

                        for model_field_name, df_col_name in model_to_df_col_map.items():
                            try:
                                model_field = StockPrice._meta.get_field(model_field_name)
                            except FieldDoesNotExist:
                                continue

                            if df_col_name in row_final_data:
                                val_from_df = row_final_data[df_col_name]
                                if pd.isna(val_from_df) or (isinstance(val_from_df, float) and (np.isinf(val_from_df) or np.isnan(val_from_df))): 
                                    defaults_for_db[model_field_name] = None
                                else:
                                    if isinstance(model_field, (django_db_models.FloatField, django_db_models.DecimalField)):
                                        defaults_for_db[model_field_name] = float(val_from_df)
                                    elif isinstance(model_field, (django_db_models.IntegerField, django_db_models.BigIntegerField)):
                                        if model_field_name in ['indi_volume', 'foreign_volume', 'organ_volume', 'market_cap', 'volume']: 
                                            try:
                                                defaults_for_db[model_field_name] = int(float(val_from_df)) 
                                            except (ValueError, TypeError):
                                                defaults_for_db[model_field_name] = None 
                                        else:
                                            defaults_for_db[model_field_name] = int(round(float(val_from_df)))
                                    elif isinstance(model_field, django_db_models.BooleanField):
                                        defaults_for_db[model_field_name] = bool(val_from_df)
                                    else: 
                                        defaults_for_db[model_field_name] = val_from_df
                            else:
                                defaults_for_db[model_field_name] = None
                        
                        prev_day_for_db_lookup = self.get_previous_trading_day(current_iter_date_for_db, kr_holidays_list_global, 1)
                        prev_day_stock_record_db = StockPrice.objects.filter(stock_code=stock_code, date=prev_day_for_db_lookup).first()
                        
                        if prev_day_stock_record_db and prev_day_stock_record_db.close_price is not None:
                            defaults_for_db['previous_day_close_price'] = prev_day_stock_record_db.close_price
                        else: 
                            if prev_day_for_db_lookup in df_final_for_db.index:
                                prev_close_in_df = df_final_for_db.loc[prev_day_for_db_lookup, 'Close'] 
                                if pd.notna(prev_close_in_df):
                                    defaults_for_db['previous_day_close_price'] = float(prev_close_in_df)
                                else:
                                    defaults_for_db['previous_day_close_price'] = None
                            else:
                                defaults_for_db['previous_day_close_price'] = None
                        
                        required_ohlcv_fields = ['open_price', 'high_price', 'low_price', 'close_price']
                        can_save = True
                        for req_fld in required_ohlcv_fields:
                            if defaults_for_db.get(req_fld) is None: 
                                can_save = False
                                break
                        if not can_save:
                            continue

                        try:
                            obj, created = StockPrice.objects.update_or_create(
                                stock_code=stock_code, date=current_iter_date_for_db,
                                defaults=defaults_for_db
                            )
                            if created:
                                saved_count_for_stock_this_fetch += 1
                        except Exception as e_db_save:
                            self.stderr.write(self.style.ERROR(f"      DB Save Error for {stock_name} ({stock_code}) on {current_iter_date_for_db}: {e_db_save}"))
                            problematic_data_log = {k: (type(v), v) for k, v in defaults_for_db.items()}
                            self.stderr.write(self.style.ERROR(f"        Problematic Defaults (type, value): {problematic_data_log}"))

                if saved_count_for_stock_this_fetch > 0:
                    self.stdout.write(self.style.SUCCESS(f"    Successfully saved/updated {saved_count_for_stock_this_fetch} records for {stock_name} ({stock_code}) up to {target_date_for_data}."))
                    processed_count +=1
                elif not df_final_for_db.empty : 
                    skipped_count += 1 
                else: 
                    pass

            except Exception as e_stock_fetch:
                self.stderr.write(self.style.ERROR(f"      Error processing/fetching data for {stock_name} ({stock_code}): {e_stock_fetch}\n{traceback.format_exc()}"))
                error_count +=1
            time.sleep(0.1) 

        self.stdout.write(self.style.SUCCESS(f"\nFinished processing all markets. Total Stocks: {total_stocks_to_process}, Successfully Processed/NewData: {processed_count}, Skipped (No new data/Already up-to-date/No FDR data/No features): {skipped_count}, Errors in processing: {error_count}"))
        self.stdout.write(self.style.SUCCESS(f'Daily DB data update/backfill process finished at {timezone.now()}.'))
