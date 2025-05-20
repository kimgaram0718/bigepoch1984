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
import joblib
import tensorflow as tf
import traceback # 오류 로깅용

from .utils import calculate_all_features, get_market_macro_data
from django.conf import settings

BASE_FEATURE_COLUMNS_FOR_CSV = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change',
    'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14',
    'KOSPI_Close', 'KOSPI_Change',
    'KOSDAQ_Close', 'KOSDAQ_Change',
    'USD_KRW_Close', 'USD_KRW_Change',
    'Indi', 'Foreign', 'Organ'
]

TARGET_COLUMN_TRAINING = 'Close'
TIME_STEPS_TRAINING = 10
FUTURE_TARGET_DAYS_TRAINING = 5
APPROX_5YR_TRADING_DAYS = 5 * 252

def get_trading_day_before(date_reference, days_offset=1):
    if isinstance(date_reference, datetime):
        current_check_date = date_reference.date()
    elif isinstance(date_reference, date_type):
        current_check_date = date_reference
    else:
        try:
            current_check_date = pd.to_datetime(date_reference).date()
        except Exception as e:
            print(f"[오류] get_trading_day_before: 날짜 변환 실패 ({date_reference}): {e}. 오늘 날짜 사용.")
            current_check_date = timezone.now().date()
    
    kr_holidays = holidays.KR(years=[current_check_date.year -1, current_check_date.year, current_check_date.year + 1])
    trading_days_found = 0
    temp_date = current_check_date
    while trading_days_found < days_offset:
        temp_date -= timedelta(days=1)
        if temp_date.weekday() < 5 and temp_date not in kr_holidays:
            trading_days_found += 1
    return temp_date


def update_stock_csv_with_all_features(stock_code, stock_name, market_name_upper, base_csv_folder, market_id_suffix_for_filename):
    """
    개별 종목의 CSV 파일을 업데이트합니다. (모든 피처 포함)
    market_name_upper: 'KOSPI' 또는 'KOSDAQ'
    market_id_suffix_for_filename: 파일명에 사용될 접미사 (예: "_kospi_", "_kosdaq_")
    """
    print(f"  - {stock_name}({stock_code}) CSV 업데이트 시도...")

    general_pattern = os.path.join(base_csv_folder, f"{stock_code}{market_id_suffix_for_filename}*.csv")
    all_potential_files = glob.glob(general_pattern)

    candidate_files = []
    for f_path in all_potential_files:
        f_name = os.path.basename(f_path)
        if f_name.endswith("_features_manualTA.csv") or f_name.endswith("_extendedTA.csv"):
            candidate_files.append(f_path)
    
    found_csv_files = candidate_files
    
    df_existing_ohlcv = pd.DataFrame()
    last_csv_date = None
    original_csv_path_to_remove = None

    if found_csv_files:
        try:
            found_csv_files.sort(key=os.path.getmtime, reverse=True)
            original_csv_path_to_remove = found_csv_files[0]
            
            if os.path.exists(original_csv_path_to_remove) and os.path.getsize(original_csv_path_to_remove) > 0:
                df_existing_ohlcv = pd.read_csv(original_csv_path_to_remove)
                if 'Date' in df_existing_ohlcv.columns:
                    df_existing_ohlcv['Date'] = pd.to_datetime(df_existing_ohlcv['Date']).dt.date
                    last_csv_date = df_existing_ohlcv['Date'].max()
                else: 
                    df_existing_ohlcv = pd.DataFrame()
                    last_csv_date = None
            else: 
                 df_existing_ohlcv = pd.DataFrame()
                 last_csv_date = None
        except Exception as e_read:
            print(f"    [경고] 기존 CSV ({original_csv_path_to_remove}) 읽기 오류: {e_read}. 새로 데이터를 가져옵니다.")
            df_existing_ohlcv = pd.DataFrame()
            last_csv_date = None

    yesterday_trading_date = get_trading_day_before(timezone.now(), 1)
    
    if last_csv_date and last_csv_date >= yesterday_trading_date:
        print(f"    {stock_name}({stock_code}) CSV는 이미 최신({last_csv_date})입니다. 건너뜁니다.")
        return False

    if last_csv_date and last_csv_date < yesterday_trading_date:
        start_fetch_date_ohlcv = last_csv_date + timedelta(days=1)
    elif not last_csv_date: 
        start_fetch_date_ohlcv = yesterday_trading_date - timedelta(days=APPROX_5YR_TRADING_DAYS * 1.2) 
    else:
        return False


    try:
        df_new_ohlcv_raw = fdr.DataReader(stock_code, start=start_fetch_date_ohlcv, end=yesterday_trading_date)
        if df_new_ohlcv_raw.empty:
            return False 
        
        df_new_ohlcv_raw.index = pd.to_datetime(df_new_ohlcv_raw.index).date
        df_new_ohlcv_raw.rename_axis('Date', inplace=True)
        df_new_ohlcv_raw.reset_index(inplace=True)

        if 'Change' not in df_new_ohlcv_raw.columns:
            df_new_ohlcv_raw['Change'] = df_new_ohlcv_raw['Close'].pct_change()
        
        for col in ['Indi', 'Foreign', 'Organ']:
            if col not in df_new_ohlcv_raw.columns:
                 df_new_ohlcv_raw[col] = 0.0

        if not df_existing_ohlcv.empty and 'Date' in df_existing_ohlcv.columns:
            cols_to_keep_from_existing = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Indi', 'Foreign', 'Organ']
            valid_existing_cols = [col for col in cols_to_keep_from_existing if col in df_existing_ohlcv.columns]
            df_combined_base = pd.concat([df_existing_ohlcv[valid_existing_cols], df_new_ohlcv_raw], ignore_index=True)
        else:
            df_combined_base = df_new_ohlcv_raw.copy()

        df_combined_base = df_combined_base.drop_duplicates(subset=['Date'], keep='last')
        df_combined_base = df_combined_base.sort_values('Date').reset_index(drop=True)
        df_combined_base.set_index('Date', inplace=True)

        if df_combined_base.empty:
            print(f"    {stock_name}({stock_code}) 데이터 병합 후 비어있음.")
            return False

        min_date_for_other_data = df_combined_base.index.min()
        max_date_for_other_data = df_combined_base.index.max()
        
        market_fdr_code_param = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
        df_market_idx, df_macro_fx = get_market_macro_data(
            min_date_for_other_data, 
            max_date_for_other_data, 
            market_fdr_code=market_fdr_code_param 
        )

        df_to_calculate_ta = df_combined_base.copy()
        if not df_market_idx.empty:
            df_to_calculate_ta = df_to_calculate_ta.join(df_market_idx, how='left')
        if not df_macro_fx.empty:
            df_to_calculate_ta = df_to_calculate_ta.join(df_macro_fx, how='left')
        
        df_to_calculate_ta.ffill(inplace=True)

        # 여기가 수정된 부분입니다: market_name -> market_name_upper
        df_final_features = calculate_all_features(df_to_calculate_ta, market_name_upper=market_name_upper)
        
        if len(df_final_features) > APPROX_5YR_TRADING_DAYS:
            df_final_features = df_final_features.tail(APPROX_5YR_TRADING_DAYS)
        
        df_final_features.reset_index(inplace=True) 

        if df_final_features.empty or 'Date' not in df_final_features.columns:
            print(f"    {stock_name}({stock_code}) 최종 피처 생성 실패 또는 Date 컬럼 없음.")
            return False
        
        cols_to_save = ['Date'] + [col for col in df_final_features.columns if col != 'Date']
        
        df_to_save = df_final_features[cols_to_save].copy()
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d')

        if df_to_save.empty or df_to_save.drop(columns=['Date']).isnull().all().all():
            print(f"    {stock_name}({stock_code}) 저장할 데이터가 없거나 모든 피처가 NaN입니다.")
            return False

        min_date_str = pd.to_datetime(df_to_save['Date']).min().strftime('%Y%m%d')
        max_date_str = pd.to_datetime(df_to_save['Date']).max().strftime('%Y%m%d')
        new_filename = f"{stock_code}{market_id_suffix_for_filename}daily_{min_date_str}_{max_date_str}_features_manualTA.csv"
        new_csv_path = os.path.join(base_csv_folder, new_filename)

        df_to_save.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
        print(f"    {stock_name}({stock_code}) CSV 저장 완료: {new_filename}")

        if original_csv_path_to_remove and original_csv_path_to_remove != new_csv_path and os.path.exists(original_csv_path_to_remove):
            try:
                os.remove(original_csv_path_to_remove)
                print(f"    이전 파일 삭제: {os.path.basename(original_csv_path_to_remove)}")
            except Exception as e_del:
                print(f"    [오류] 이전 파일 삭제 실패 ({original_csv_path_to_remove}): {e_del}")
        return True

    except Exception as e_main:
        print(f"    [오류] {stock_name}({stock_code}) 처리 중 심각한 오류: {e_main}")
        traceback.print_exc()
        return False


def run_daily_csv_update_tasks(market_config_list):
    print(f"일일 CSV 데이터 업데이트 작업 시작...")
    total_updated_count = 0
    
    for market_info in market_config_list:
        market_name_display = market_info['name'] 
        market_csv_base_folder = market_info['csv_folder']
        market_file_id_suffix = market_info['id_suffix'] 

        os.makedirs(market_csv_base_folder, exist_ok=True)
        print(f"\n--- {market_name_display} 시장 CSV 업데이트 시작 ---")
        print(f"대상 폴더: {market_csv_base_folder}")

        stock_listing_raw = fdr.StockListing(market_name_display)
        if 'Symbol' not in stock_listing_raw.columns and 'Code' in stock_listing_raw.columns:
            stock_listing_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        
        if 'Symbol' not in stock_listing_raw.columns or 'Name' not in stock_listing_raw.columns:
            print(f"[오류] {market_name_display} 주식 목록에 'Symbol' 또는 'Name' 컬럼 없음. 건너뜁니다.")
            continue
            
        stock_listing = stock_listing_raw[stock_listing_raw['Symbol'].notna() & stock_listing_raw['Name'].notna()]
        num_stocks_in_market = len(stock_listing)
        market_updated_count = 0

        for i, stock_row in stock_listing.iterrows():
            stock_code_val = stock_row['Symbol']
            stock_name_val = stock_row['Name']
            
            print(f"[{market_name_display} {i+1}/{num_stocks_in_market}] {stock_name_val}({stock_code_val}) 처리 중...")
            if update_stock_csv_with_all_features(stock_code_val, stock_name_val, market_name_display.upper(), market_csv_base_folder, market_file_id_suffix):
                market_updated_count +=1
            
            if (i+1) % 20 == 0: 
                print(f"  [{market_name_display}] {i+1}개 종목 확인 완료. 현재까지 {market_updated_count}개 업데이트됨.")
            time.sleep(0.1) 

        print(f"--- {market_name_display} 시장 업데이트 완료: 총 {num_stocks_in_market}개 중 {market_updated_count}개 종목 CSV 업데이트/생성 ---")
        total_updated_count += market_updated_count

    print(f"\n일일 CSV 데이터 업데이트 작업 완료. 총 {total_updated_count}개 CSV 파일 업데이트/생성.")


def run_daily_startup_tasks_main(enable_model_retraining=False): 
    print(f"일일 데이터 업데이트 작업 (startup_tasks.py) 시작... (모델 재학습 비활성화)")
    
    try:
        kosdaq_csv_folder_path = settings.KOSDAQ_TRAINING_DATA_DIR
        kospi_csv_folder_path = settings.KOSPI_TRAINING_DATA_DIR
        
        if not os.path.isdir(kosdaq_csv_folder_path):
            print(f"[경고] KOSDAQ CSV 폴더 경로가 잘못되었거나 존재하지 않습니다: {kosdaq_csv_folder_path}")
            os.makedirs(kosdaq_csv_folder_path, exist_ok=True)
            print(f"폴더 생성 시도: {kosdaq_csv_folder_path}")
        if not os.path.isdir(kospi_csv_folder_path):
            print(f"[경고] KOSPI CSV 폴더 경로가 잘못되었거나 존재하지 않습니다: {kospi_csv_folder_path}")
            os.makedirs(kospi_csv_folder_path, exist_ok=True)
            print(f"폴더 생성 시도: {kospi_csv_folder_path}")

    except AttributeError:
        print("[오류] settings.py에 KOSDAQ_TRAINING_DATA_DIR 또는 KOSPI_TRAINING_DATA_DIR가 정의되지 않았습니다.")
        return

    markets_to_process = [
        {'name': 'KOSDAQ', 'csv_folder': kosdaq_csv_folder_path, 'id_suffix': '_kosdaq_'},
        {'name': 'KOSPI', 'csv_folder': kospi_csv_folder_path, 'id_suffix': '_kospi_'}
    ]
    
    run_daily_csv_update_tasks(markets_to_process)
    
    if enable_model_retraining:
        print("모델 재학습은 별도의 스크립트 또는 Management Command (예: run_daily_updates --retrain-models)로 실행해야 합니다.")
        print("이 startup_tasks.py에서는 CSV 업데이트만 수행합니다.")
    
    print("일일 데이터 업데이트 작업 (startup_tasks.py) 완료.")

if __name__ == '__main__':
    print("startup_tasks.py를 직접 실행합니다. (Django settings 없이 실행)")
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    mock_base_dir = os.path.join(current_script_dir, '..', '..') 

    class MockSettings:
        BASE_DIR = mock_base_dir
        KOSDAQ_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'predict_info', 'data', 'kosdaq_data')
        KOSPI_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'predict_info', 'data', 'kospi_data')

    global settings
    settings = MockSettings()
    
    print(f"Mock BASE_DIR: {settings.BASE_DIR}")
    print(f"Mock KOSDAQ_TRAINING_DATA_DIR: {settings.KOSDAQ_TRAINING_DATA_DIR}")
    print(f"Mock KOSPI_TRAINING_DATA_DIR: {settings.KOSPI_TRAINING_DATA_DIR}")

    run_daily_startup_tasks_main(enable_model_retraining=False)
