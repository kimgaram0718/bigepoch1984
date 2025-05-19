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
# from tensorflow.keras.models import load_model # 모델 직접 로드/학습은 여기서 제거 (views.py에서 처리)
# from tensorflow.keras.layers import Input, LSTM, Dense
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split # 여기서는 사용 안 함

# views.py와 동일한 피처 정의 가져오기 (또는 공통 파일로 분리)
# 여기서는 views.py의 get_feature_columns_for_market 함수를 직접 호출하기 어려우므로,
# 기본 27개 피처 목록을 정의하고, 시장별로 동적으로 컬럼명을 조정하는 로직을 포함해야 합니다.
# 또는, CSV 생성 시에는 모든 가능한 시장 지표 컬럼(KOSPI_Close, KOSDAQ_Close 등)을 다 넣어두고
# 모델 학습/예측 시에 선택적으로 사용하도록 할 수도 있습니다.
# 여기서는 views.py에서 정의한 BASE_FEATURE_COLUMNS와 유사하게 정의합니다.

from .utils import calculate_all_features, get_market_macro_data # 수정된 utils 임포트
from django.conf import settings

# --- 설정값 ---
# 이 FEATURE_COLUMNS_TRAINING은 CSV 파일에 저장될 컬럼 목록이며,
# 모델 학습 시 이 중에서 실제 사용할 27개를 선택하게 됩니다.
# calculate_all_features 함수가 생성하는 모든 컬럼을 포함하도록 확장합니다.
# pandas-ta가 생성하는 이름 그대로 사용합니다.
# 시장/거시/투자자 데이터는 calculate_all_features를 호출하기 전에 원본 DataFrame에 병합되어야 합니다.

# startup_tasks.py 에서는 CSV를 생성하는 역할이므로,
# views.py의 get_feature_columns_for_market(market_name_upper) 와 유사하게
# 해당 시장에 맞는 피처 컬럼 리스트를 사용해야 합니다.
# 여기서는 단순화를 위해 views.py에서 가져온 BASE_FEATURE_COLUMNS를 사용하고,
# CSV 저장 시 시장별 접두사를 가진 컬럼이 포함되도록 합니다.

BASE_FEATURE_COLUMNS_FOR_CSV = [ # CSV 저장 시 포함될 수 있는 모든 컬럼 (calculate_all_features 결과 기반)
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change',
    'ATR_14', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14', 'DMP_14', 'DMN_14',
    # 아래 시장/거시/투자자 컬럼은 calculate_all_features 호출 전에 원본 데이터에 병합되어야 함
    'KOSPI_Close', 'KOSPI_Change', # KOSPI용
    'KOSDAQ_Close', 'KOSDAQ_Change', # KOSDAQ용
    'USD_KRW_Close', 'USD_KRW_Change',
    'Indi', 'Foreign', 'Organ'
]
# 실제 저장 시에는 해당 시장에 맞는 컬럼만 선택하거나, 모든 시장 컬럼을 포함하고 학습 시 선택


TARGET_COLUMN_TRAINING = 'Close' # 모델의 타겟은 종가
TIME_STEPS_TRAINING = 10 # views.py와 일치
FUTURE_TARGET_DAYS_TRAINING = 5 # views.py와 일치
# EPOCHS_FOR_DAILY_RETRAIN = 3 # 이 파일에서는 모델 재학습 로직 제거 (management command로 분리 권장)
APPROX_5YR_TRADING_DAYS = 5 * 252

# --- Helper Functions (데이터 업데이트용) ---
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
    target_csv_pattern = os.path.join(base_csv_folder, f"{stock_code}{market_id_suffix_for_filename}*features_manualTA.csv") # 기존 파일명 패턴 유지
    found_csv_files = glob.glob(target_csv_pattern)
    
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
                else: # Date 컬럼 없으면 처음부터 다시
                    df_existing_ohlcv = pd.DataFrame() # 비워서 새로 받도록
                    last_csv_date = None
            else: # 파일이 비어있으면 새로
                 df_existing_ohlcv = pd.DataFrame()
                 last_csv_date = None
        except Exception as e_read:
            print(f"    [경고] 기존 CSV ({original_csv_path_to_remove}) 읽기 오류: {e_read}. 새로 데이터를 가져옵니다.")
            df_existing_ohlcv = pd.DataFrame()
            last_csv_date = None
            # original_csv_path_to_remove는 유지하여 나중에 삭제 시도

    yesterday_trading_date = get_trading_day_before(timezone.now(), 1)
    
    # 데이터 가져올 시작 날짜 결정
    if last_csv_date and last_csv_date < yesterday_trading_date:
        start_fetch_date_ohlcv = last_csv_date + timedelta(days=1)
    elif not last_csv_date: # 기존 데이터가 없거나 날짜를 알 수 없는 경우
        start_fetch_date_ohlcv = yesterday_trading_date - timedelta(days=APPROX_5YR_TRADING_DAYS * 1.2) # 최근 5년치 + 여유분
    else: # 이미 최신 데이터
        print(f"    {stock_name}({stock_code}) CSV는 이미 최신({last_csv_date})입니다. 건너뜁니다.")
        return False # 업데이트 안 함

    # 1. 개별 종목 OHLCV (+Change) 및 투자자별 매매동향 데이터 가져오기
    # 투자자별 매매동향은 pykrx 또는 DB에서 가져와야 함. 여기서는 FDR의 기본 데이터만 사용.
    try:
        df_new_ohlcv_raw = fdr.DataReader(stock_code, start=start_fetch_date_ohlcv, end=yesterday_trading_date)
        if df_new_ohlcv_raw.empty:
            # print(f"    {stock_name}({stock_code}) 신규 OHLCV 데이터 없음 ({start_fetch_date_ohlcv} ~ {yesterday_trading_date}).")
            return False # 업데이트 안 함
        
        df_new_ohlcv_raw.index = pd.to_datetime(df_new_ohlcv_raw.index).date # 인덱스를 date 객체로
        df_new_ohlcv_raw.rename_axis('Date', inplace=True)
        df_new_ohlcv_raw.reset_index(inplace=True)

        # 'Change' 컬럼이 없다면 추가
        if 'Change' not in df_new_ohlcv_raw.columns:
            df_new_ohlcv_raw['Change'] = df_new_ohlcv_raw['Close'].pct_change()
        
        # 투자자별 매매동향 데이터 (임시로 NaN 컬럼 추가, 실제로는 pykrx 등으로 채워야 함)
        # 이 데이터는 일별로 수집되어 StockPrice 모델 등에 저장되어 있어야 함.
        # 여기서는 CSV 생성 시점에 pykrx를 직접 호출하거나, DB에서 읽어와야 함.
        # 예시: df_investor = stock.get_market_trading_value_by_date(start_date_str, end_date_str, stock_code) 등
        # 여기서는 임시로 0으로 채움
        for col in ['Indi', 'Foreign', 'Organ']:
            if col not in df_new_ohlcv_raw.columns:
                 df_new_ohlcv_raw[col] = 0.0


        # 기존 데이터와 새 데이터 병합
        if not df_existing_ohlcv.empty and 'Date' in df_existing_ohlcv.columns:
            # 기존 데이터에서 OHLCV 및 투자자 컬럼만 선택 (다른 TA 컬럼은 새로 계산)
            cols_to_keep_from_existing = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Indi', 'Foreign', 'Organ']
            valid_existing_cols = [col for col in cols_to_keep_from_existing if col in df_existing_ohlcv.columns]
            df_combined_base = pd.concat([df_existing_ohlcv[valid_existing_cols], df_new_ohlcv_raw], ignore_index=True)
        else:
            df_combined_base = df_new_ohlcv_raw.copy()

        df_combined_base = df_combined_base.drop_duplicates(subset=['Date'], keep='last')
        df_combined_base = df_combined_base.sort_values('Date').reset_index(drop=True)
        df_combined_base.set_index('Date', inplace=True) # 인덱스를 다시 Date로

        if df_combined_base.empty:
            print(f"    {stock_name}({stock_code}) 데이터 병합 후 비어있음.")
            return False

        # 2. 시장 지수 및 환율 데이터 가져오기 (전체 기간에 대해)
        min_date_for_other_data = df_combined_base.index.min()
        max_date_for_other_data = df_combined_base.index.max()
        
        market_fdr_code = 'KS11' if market_name_upper == 'KOSPI' else 'KQ11'
        df_market_idx, df_macro_fx = get_market_macro_data(min_date_for_other_data, max_date_for_other_data, market_code=market_fdr_code)

        # 3. 모든 데이터 병합
        df_to_calculate_ta = df_combined_base.copy()
        if not df_market_idx.empty:
            df_to_calculate_ta = df_to_calculate_ta.join(df_market_idx, how='left')
        if not df_macro_fx.empty:
            df_to_calculate_ta = df_to_calculate_ta.join(df_macro_fx, how='left')
        
        # 병합 후 ffill 필수
        df_to_calculate_ta.ffill(inplace=True)


        # 4. 모든 기술적 지표 계산 (utils.calculate_all_features 사용)
        df_final_features = calculate_all_features(df_to_calculate_ta, market_name=market_name_upper)
        
        # 최근 5년치 데이터만 유지 (선택 사항)
        if len(df_final_features) > APPROX_5YR_TRADING_DAYS:
            df_final_features = df_final_features.tail(APPROX_5YR_TRADING_DAYS)
        
        df_final_features.reset_index(inplace=True) # Date 컬럼으로 복원

        if df_final_features.empty or 'Date' not in df_final_features.columns:
            print(f"    {stock_name}({stock_code}) 최종 피처 생성 실패 또는 Date 컬럼 없음.")
            return False
        
        # CSV 저장 시 포함할 컬럼 목록 (BASE_FEATURE_COLUMNS_FOR_CSV 와 유사하게, 실제 생성된 컬럼 기준)
        # calculate_all_features가 생성한 모든 컬럼 + Date
        cols_to_save = ['Date'] + [col for col in df_final_features.columns if col != 'Date']
        
        # 특정 컬럼만 선택해서 저장하고 싶다면 여기서 필터링
        # 예: 현재 시장에 맞는 피처만 선택 (views.py의 get_feature_columns_for_market 참고)
        # current_market_features = get_feature_columns_for_market(market_name_upper) # 이 함수를 startup_tasks에서도 사용 가능하게 만들어야 함
        # cols_to_save = ['Date'] + [col for col in current_market_features if col in df_final_features.columns]


        df_to_save = df_final_features[cols_to_save].copy()
        df_to_save['Date'] = pd.to_datetime(df_to_save['Date']).dt.strftime('%Y-%m-%d') # 날짜 형식 통일

        if df_to_save.empty or df_to_save.drop(columns=['Date']).isnull().all().all():
            print(f"    {stock_name}({stock_code}) 저장할 데이터가 없거나 모든 피처가 NaN입니다.")
            return False

        # 새 파일명 생성 (날짜 범위 포함)
        min_date_str = pd.to_datetime(df_to_save['Date']).min().strftime('%Y%m%d')
        max_date_str = pd.to_datetime(df_to_save['Date']).max().strftime('%Y%m%d')
        new_filename = f"{stock_code}{market_id_suffix_for_filename}daily_{min_date_str}_{max_date_str}_features_manualTA.csv"
        new_csv_path = os.path.join(base_csv_folder, new_filename)

        df_to_save.to_csv(new_csv_path, index=False, encoding='utf-8-sig')
        print(f"    {stock_name}({stock_code}) CSV 저장 완료: {new_filename}")

        # 이전 파일 삭제 (파일명이 변경된 경우)
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
    """
    지정된 시장 목록에 대해 모든 종목의 CSV 파일을 업데이트합니다.
    market_config_list: [{'name': 'KOSPI', 'csv_folder': 'path', 'id_suffix': '_kospi_'}, ...]
    """
    print(f"일일 CSV 데이터 업데이트 작업 시작...")
    total_updated_count = 0
    
    for market_info in market_config_list:
        market_name_display = market_info['name'] # 예: "KOSPI"
        market_csv_base_folder = market_info['csv_folder']
        market_file_id_suffix = market_info['id_suffix'] # 예: "_kospi_"

        os.makedirs(market_csv_base_folder, exist_ok=True)
        print(f"\n--- {market_name_display} 시장 CSV 업데이트 시작 ---")
        print(f"대상 폴더: {market_csv_base_folder}")

        stock_listing_raw = fdr.StockListing(market_name_display)
        if 'Symbol' not in stock_listing_raw.columns and 'Code' in stock_listing_raw.columns:
            stock_listing_raw.rename(columns={'Code': 'Symbol'}, inplace=True)
        
        if 'Symbol' not in stock_listing_raw.columns or 'Name' not in stock_listing_raw.columns:
            print(f"[오류] {market_name_display} 주식 목록에 'Symbol' 또는 'Name' 컬럼 없음. 건너<0xEB><0><0xA9>니다.")
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
            
            if (i+1) % 20 == 0: # 20개 종목마다 진행 상황 간단히 표시
                print(f"  [{market_name_display}] {i+1}개 종목 확인 완료. 현재까지 {market_updated_count}개 업데이트됨.")
            time.sleep(0.1) # FDR 호출 부하 감소

        print(f"--- {market_name_display} 시장 업데이트 완료: 총 {num_stocks_in_market}개 중 {market_updated_count}개 종목 CSV 업데이트/생성 ---")
        total_updated_count += market_updated_count

    print(f"\n일일 CSV 데이터 업데이트 작업 완료. 총 {total_updated_count}개 CSV 파일 업데이트/생성.")


# Django 앱 시작 시 또는 Management Command로 호출될 메인 함수
def run_daily_startup_tasks_main(enable_model_retraining=False): # 모델 재학습 로직은 제거됨
    print(f"일일 데이터 업데이트 작업 (startup_tasks.py) 시작... (모델 재학습 비활성화)")
    
    # 모델 및 스케일러 저장/로드 기본 경로는 Django settings에서 가져옴
    # ml_models_dir_main = os.path.join(settings.BASE_DIR, 'predict_info', 'ml_models')
    # os.makedirs(ml_models_dir_main, exist_ok=True)

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
        # 여기에 모델 재학습 로직을 직접 넣는 것보다 분리하는 것이 좋음.
        # 만약 넣는다면, 이전 `update_csv_and_retrain_market_model` 함수와 유사한 로직 필요.
        # 단, 해당 함수는 13개 피처 기준이었으므로, 27개 피처에 맞게 수정 필요.
        # 또한, 스케일러(scaler_X, scaler_y)도 27개 피처로 다시 학습하고 저장해야 함.
        # 예: from .model_trainer_module import retrain_market_models
        # retrain_market_models(markets_to_process, ml_models_dir_main, ...)
    
    print("일일 데이터 업데이트 작업 (startup_tasks.py) 완료.")

if __name__ == '__main__':
    # 이 스크립트를 직접 실행하는 경우는 보통 Django 환경 외부에서의 테스트 목적입니다.
    # Django settings를 로드할 수 있도록 설정하거나, 필요한 경로를 직접 지정해야 합니다.
    print("startup_tasks.py를 직접 실행합니다. (Django settings 없이 실행)")
    
    # 임시 경로 설정 (직접 실행 테스트용)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    mock_base_dir = os.path.join(current_script_dir, '..', '..') # 프로젝트 루트 추정

    class MockSettings:
        BASE_DIR = mock_base_dir
        KOSDAQ_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'data_kosdaq_csv')
        KOSPI_TRAINING_DATA_DIR = os.path.join(mock_base_dir, 'data_kospi_csv')

    # 실제 Django settings 대신 MockSettings 사용
    global settings
    settings = MockSettings()
    
    print(f"Mock BASE_DIR: {settings.BASE_DIR}")
    print(f"Mock KOSDAQ_TRAINING_DATA_DIR: {settings.KOSDAQ_TRAINING_DATA_DIR}")
    print(f"Mock KOSPI_TRAINING_DATA_DIR: {settings.KOSPI_TRAINING_DATA_DIR}")

    run_daily_startup_tasks_main(enable_model_retraining=False)
