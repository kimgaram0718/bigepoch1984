# predict_Info/startup_tasks.py
import pandas as pd
import FinanceDataReader as fdr
from django.utils import timezone
from datetime import timedelta, datetime, date as date_type
import holidays
import time
import os
import glob
import numpy as np
import traceback 

from .utils import calculate_all_features, get_market_macro_data, PANDAS_TA_AVAILABLE
from django.conf import settings

# 컬럼명 정의 (FDR StockListing 기준 또는 DB 저장 기준과 일치시킬 수 있음)
FUNDAMENTAL_COLS_FOR_CSV = ['Marcap', 'PER', 'PBR'] # FDR StockListing 컬럼명 사용
INVESTOR_COLS_FOR_CSV = ['Indi', 'Foreign', 'Organ'] 

BASE_OHLCV_COLS_FOR_CSV = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
EXISTING_TA_COLS_FOR_CSV = [
    'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
]
NEW_TA_COLS_FOR_CSV = ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14']
MACRO_DATA_COLS_FOR_CSV = ['USD_KRW_Close', 'USD_KRW_Change']

APPROX_5YR_TRADING_DAYS = 5 * 252 

def get_trading_day_before(date_reference, days_offset=1):
    if isinstance(date_reference, datetime): current_check_date = date_reference.date()
    elif isinstance(date_reference, date_type): current_check_date = date_reference
    else:
        try: current_check_date = pd.to_datetime(date_reference).date()
        except Exception as e:
            print(f"[Error] get_trading_day_before: Date conversion failed ({date_reference}): {e}. Using today.")
            current_check_date = timezone.now().date()
    
    kr_holidays = holidays.KR(years=list(set([current_check_date.year -1, current_check_date.year, current_check_date.year + 1])))
    trading_days_found = 0
    temp_date = current_check_date
    while trading_days_found < days_offset:
        temp_date -= timedelta(days=1)
        if temp_date.weekday() < 5 and temp_date not in kr_holidays:
            trading_days_found += 1
    return temp_date

def update_stock_csv_with_all_features(stock_code, stock_name, market_name_upper, base_csv_folder, market_id_suffix_for_filename, df_market_fundamentals):
    if not PANDAS_TA_AVAILABLE:
        print(f"    [CRITICAL] {stock_name}({stock_code}) pandas_ta library missing. CSV update skipped.")
        return False
        
    print(f"  - Updating CSV for {stock_name}({stock_code}) (Market: {market_name_upper})...")

    general_pattern = os.path.join(base_csv_folder, f"{stock_code}{market_id_suffix_for_filename}*.csv")
    all_potential_files = glob.glob(general_pattern)
    candidate_files = [f_path for f_path in all_potential_files if "_features_manualTA.csv" in os.path.basename(f_path)]
    
    df_existing_features = pd.DataFrame(); last_csv_date = None; original_csv_path_to_remove = None

    if candidate_files:
        try:
            candidate_files.sort(key=os.path.getmtime, reverse=True)
            original_csv_path_to_remove = candidate_files[0]
            if os.path.exists(original_csv_path_to_remove) and os.path.getsize(original_csv_path_to_remove) > 0:
                df_existing_features = pd.read_csv(original_csv_path_to_remove)
                if 'Date' in df_existing_features.columns:
                    df_existing_features['Date'] = pd.to_datetime(df_existing_features['Date']).dt.date
                    last_csv_date = df_existing_features['Date'].max()
                else: df_existing_features = pd.DataFrame() # No Date column
            else: df_existing_features = pd.DataFrame() # Empty file
        except Exception as e_read:
            print(f"    [Warning] Error reading existing CSV ({original_csv_path_to_remove}): {e_read}. Fetching new data.")
            df_existing_features = pd.DataFrame()

    yesterday_trading_date = get_trading_day_before(timezone.now(), 1)
    if last_csv_date and last_csv_date >= yesterday_trading_date:
        print(f"    {stock_name}({stock_code}) CSV is up-to-date ({last_csv_date}). Skipping.")
        return False

    start_fetch_date_fdr = (last_csv_date + timedelta(days=1)) if last_csv_date else (yesterday_trading_date - timedelta(days=APPROX_5YR_TRADING_DAYS + 60))
    print(f"    Fetching FDR data for {stock_name}({stock_code}): {start_fetch_date_fdr} to {yesterday_trading_date} (source: naver)")
    
    try:
        df_new_ohlcv_investor_raw = fdr.DataReader(stock_code, start=start_fetch_date_fdr, end=yesterday_trading_date, data_source='naver')
        if df_new_ohlcv_investor_raw.empty and not last_csv_date:
            print(f"    [Warning] No new data from FDR for {stock_name}({stock_code}). Skipping.")
            return False 
        if df_new_ohlcv_investor_raw.empty and last_csv_date:
            print(f"    No additional new data from FDR for {stock_name}({stock_code}). Existing data maintained.")
            return False

        df_new_ohlcv_investor_raw.index = pd.to_datetime(df_new_ohlcv_investor_raw.index).date
        df_new_ohlcv_investor_raw.rename_axis('Date', inplace=True)
        investor_col_map = {'개인': 'Indi', '외국인': 'Foreign', '기관': 'Organ'}
        df_new_ohlcv_investor_raw.rename(columns=investor_col_map, inplace=True)
        
        cols_from_fdr = BASE_OHLCV_COLS_FOR_CSV + INVESTOR_COLS_FOR_CSV
        for col in cols_from_fdr:
            if col not in df_new_ohlcv_investor_raw.columns: df_new_ohlcv_investor_raw[col] = np.nan
        
        df_new_data_processed = df_new_ohlcv_investor_raw.reset_index()[['Date'] + cols_from_fdr]

        if not df_existing_features.empty and 'Date' in df_existing_features.columns:
            base_cols_in_existing = ['Date'] + [col for col in cols_from_fdr if col in df_existing_features.columns]
            df_base_for_concat = df_existing_features[base_cols_in_existing]
            df_combined_base = pd.concat([df_base_for_concat, df_new_data_processed], ignore_index=True)
        else:
            df_combined_base = df_new_data_processed.copy()

        df_combined_base = df_combined_base.drop_duplicates(subset=['Date'], keep='last').sort_values('Date').reset_index(drop=True)
        if df_combined_base.empty or 'Date' not in df_combined_base.columns:
             print(f"    {stock_name}({stock_code}) Data empty after merge or Date column missing. Skipping.")
             return False
        df_combined_base.set_index('Date', inplace=True)

        min_date_for_other_data = df_combined_base.index.min(); max_date_for_other_data = df_combined_base.index.max()
        market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
        df_market_idx, df_macro_fx = get_market_macro_data(min_date_for_other_data, max_date_for_other_data, market_fdr_code=market_fdr_code_param)

        df_to_calculate_ta = df_combined_base.copy()
        market_cols_to_add = ['KOSPI_Close', 'KOSPI_Change'] if market_name_upper == "KOSPI" else ['KOSDAQ_Close', 'KOSDAQ_Change']

        if not df_market_idx.empty: df_to_calculate_ta = df_to_calculate_ta.join(df_market_idx, how='left')
        else: 
            for col in market_cols_to_add: df_to_calculate_ta[col] = np.nan
        if not df_macro_fx.empty: df_to_calculate_ta = df_to_calculate_ta.join(df_macro_fx, how='left')
        else: 
            for col in MACRO_DATA_COLS_FOR_CSV: df_to_calculate_ta[col] = np.nan
        for col in INVESTOR_COLS_FOR_CSV: # Ensure investor columns exist and fill NaN with 0.0
            if col not in df_to_calculate_ta.columns: df_to_calculate_ta[col] = 0.0
            else: df_to_calculate_ta[col].fillna(0.0, inplace=True)
        
        # Add Fundamental Data (Marcap, PER, PBR)
        # These are snapshot values, applied to all rows for simplicity in CSV.
        # For more accurate time-series fundamental data, a different data source/method would be needed.
        stock_fundamentals = df_market_fundamentals.get(stock_code, {})
        for fund_col in FUNDAMENTAL_COLS_FOR_CSV: # e.g. 'Marcap', 'PER', 'PBR'
            df_to_calculate_ta[fund_col] = stock_fundamentals.get(fund_col, np.nan) # Get from pre-fetched dict

        df_to_calculate_ta.ffill(inplace=True) # ffill after adding all external data
        df_final_features = calculate_all_features(df_to_calculate_ta.copy(), market_name_upper=market_name_upper)
        
        current_market_feature_columns_ordered = BASE_OHLCV_COLS_FOR_CSV + EXISTING_TA_COLS_FOR_CSV + \
                                                 NEW_TA_COLS_FOR_CSV + market_cols_to_add + \
                                                 MACRO_DATA_COLS_FOR_CSV + INVESTOR_COLS_FOR_CSV + \
                                                 FUNDAMENTAL_COLS_FOR_CSV # Add fundamental columns
        
        for col_expected in current_market_feature_columns_ordered:
            if col_expected not in df_final_features.columns:
                print(f"    [Warning] Final feature set missing '{col_expected}'. Adding as NaN. ({stock_name})")
                df_final_features[col_expected] = np.nan
        
        df_final_features.reset_index(inplace=True) 
        if df_final_features.empty or 'Date' not in df_final_features.columns:
            print(f"    {stock_name}({stock_code}) Final features empty or Date column missing. Skipping.")
            return False
        
        cols_to_save_ordered = ['Date'] + current_market_feature_columns_ordered
        df_to_save = df_final_features[cols_to_save_ordered].copy()
        
        if len(df_to_save) > APPROX_5YR_TRADING_DAYS: df_to_save = df_to_save.tail(APPROX_5YR_TRADING_DAYS)
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')
        
        numeric_cols_for_fillna = [col for col in df_to_save.columns if col != 'Date']
        df_to_save[numeric_cols_for_fillna] = df_to_save[numeric_cols_for_fillna].fillna(0.0)

        if df_to_save.empty or df_to_save.drop(columns=['Date']).isnull().all().all():
            print(f"    {stock_name}({stock_code}) No data to save or all features are NaN. Skipping.")
            return False

        min_date_str = pd.to_datetime(df_to_save['Date']).min().strftime('%Y%m%d')
        max_date_str = pd.to_datetime(df_to_save['Date']).max().strftime('%Y%m%d')
        new_filename = f"{stock_code}{market_id_suffix_for_filename}daily_{min_date_str}_{max_date_str}_features_manualTA.csv"
        new_csv_path = os.path.join(base_csv_folder, new_filename)
        df_to_save.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
        print(f"    {stock_name}({stock_code}) CSV saved: {new_filename}")

        if original_csv_path_to_remove and original_csv_path_to_remove != new_csv_path and os.path.exists(original_csv_path_to_remove):
            try: os.remove(original_csv_path_to_remove); print(f"    Old file removed: {os.path.basename(original_csv_path_to_remove)}")
            except Exception as e_del: print(f"    [Error] Failed to remove old file ({original_csv_path_to_remove}): {e_del}")
        return True
    except Exception as e_main:
        print(f"    [CRITICAL ERROR] Major error processing {stock_name}({stock_code}): {e_main}")
        traceback.print_exc()
        return False

def run_daily_csv_update_tasks(market_config_list):
    print(f"Daily CSV data update task started...")
    total_updated_count = 0
    
    # Fetch market-wide fundamental data once
    print("Fetching market-wide fundamental data (Marcap, PER, PBR) via StockListing('KRX-DESC')...")
    df_all_fundamentals_raw = pd.DataFrame()
    try:
        df_all_fundamentals_raw = fdr.StockListing('KRX-DESC')
        if 'Code' in df_all_fundamentals_raw.columns: # Ensure 'Symbol' column for consistency
            df_all_fundamentals_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        # Create a dictionary for faster lookup: {stock_code: {Marcap: val, PER: val, PBR: val}}
        market_fundamentals_dict = {}
        if 'Symbol' in df_all_fundamentals_raw.columns:
            for _, row in df_all_fundamentals_raw.iterrows():
                code = row['Symbol']
                market_fundamentals_dict[code] = {
                    'Marcap': row.get('Marcap'),
                    'PER': row.get('PER'),
                    'PBR': row.get('PBR')
                }
            print(f"Successfully fetched and processed fundamental data for {len(market_fundamentals_dict)} stocks.")
        else:
            print("[Warning] 'Symbol' column not found in StockListing result. Fundamentals will be missing in CSVs.")
    except Exception as e_fund:
        print(f"[Error] Failed to fetch market-wide fundamental data: {e_fund}. Fundamentals will be missing in CSVs.")
        traceback.print_exc()
        market_fundamentals_dict = {} # Ensure it's a dict

    for market_info in market_config_list:
        market_name_display = market_info['name'] 
        market_csv_base_folder = market_info['csv_folder']
        market_file_id_suffix = market_info['id_suffix'] 

        os.makedirs(market_csv_base_folder, exist_ok=True)
        print(f"\n--- Processing {market_name_display} market CSVs ---")
        print(f"Target folder: {market_csv_base_folder}")

        stock_listing_raw = fdr.StockListing(market_name_display)
        if 'Symbol' not in stock_listing_raw.columns and 'Code' in stock_listing_raw.columns:
            stock_listing_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        
        if 'Symbol' not in stock_listing_raw.columns or 'Name' not in stock_listing_raw.columns:
            print(f"[Error] {market_name_display} stock list missing 'Symbol' or 'Name'. Skipping.")
            continue
            
        stock_listing = stock_listing_raw[stock_listing_raw['Symbol'].notna() & stock_listing_raw['Name'].notna()]
        num_stocks_in_market = len(stock_listing)
        market_updated_count = 0

        for i, stock_row in stock_listing.iterrows():
            stock_code_val = stock_row['Symbol']
            stock_name_val = stock_row['Name']
            
            print(f"[{market_name_display} {i+1}/{num_stocks_in_market}] Processing {stock_name_val}({stock_code_val})...")
            # Pass the pre-fetched fundamental data to the update function
            if update_stock_csv_with_all_features(stock_code_val, stock_name_val, market_name_display.upper(), market_csv_base_folder, market_file_id_suffix, market_fundamentals_dict):
                market_updated_count +=1
            
            if (i+1) % 20 == 0: 
                print(f"  [{market_name_display}] {i+1} stocks checked. {market_updated_count} updated so far.")
            time.sleep(0.05) # Reduced sleep as fundamental data is pre-fetched

        print(f"--- {market_name_display} market update complete: {market_updated_count} out of {num_stocks_in_market} stock CSVs updated/created ---")
        total_updated_count += market_updated_count

    print(f"\nDaily CSV data update task finished. Total {total_updated_count} CSV files updated/created.")

def run_daily_startup_tasks_main(enable_model_retraining=False): 
    print(f"Daily data update task (startup_tasks.py) started... (Model retraining: {'Enabled' if enable_model_retraining else 'Disabled'})")
    
    try:
        kosdaq_csv_folder_path = settings.KOSDAQ_TRAINING_DATA_DIR
        kospi_csv_folder_path = settings.KOSPI_TRAINING_DATA_DIR
        if not os.path.isdir(kosdaq_csv_folder_path): os.makedirs(kosdaq_csv_folder_path, exist_ok=True)
        if not os.path.isdir(kospi_csv_folder_path): os.makedirs(kospi_csv_folder_path, exist_ok=True)
    except AttributeError:
        print("[Error] KOSDAQ_TRAINING_DATA_DIR or KOSPI_TRAINING_DATA_DIR not defined in settings.py.")
        return

    markets_to_process = [
        {'name': 'KOSDAQ', 'csv_folder': kosdaq_csv_folder_path, 'id_suffix': '_kosdaq_'},
        {'name': 'KOSPI', 'csv_folder': kospi_csv_folder_path, 'id_suffix': '_kospi_'}
    ]
    run_daily_csv_update_tasks(markets_to_process)
    
    if enable_model_retraining:
        print("Model retraining should be handled by a separate script or Management Command (e.g., run_daily_updates --retrain-models).")
        print("This startup_tasks.py script only handles CSV updates.")
    
    print("Daily data update task (startup_tasks.py) finished.")

if __name__ == '__main__':
    print("Running startup_tasks.py directly (without Django settings)...")
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    mock_base_dir = os.path.join(current_script_dir, '..', '..') 
    class MockSettings:
        BASE_DIR = mock_base_dir
        KOSDAQ_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'predict_info', 'data', 'kosdaq_data_test') # Use test folders
        KOSPI_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'predict_info', 'data', 'kospi_data_test')
    global settings; settings = MockSettings()
    print(f"Mock KOSDAQ_TRAINING_DATA_DIR: {settings.KOSDAQ_TRAINING_DATA_DIR}")
    print(f"Mock KOSPI_TRAINING_DATA_DIR: {settings.KOSPI_TRAINING_DATA_DIR}")
    run_daily_startup_tasks_main(enable_model_retraining=False)
