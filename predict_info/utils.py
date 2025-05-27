import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta, date as date_type
import traceback
import holidays
from django.core.cache import cache
import os
import glob
from django.conf import settings

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    print("[INFO][predict_info/utils.py] pandas_ta 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("[CRITICAL ERROR][predict_info/utils.py] 'pandas_ta' 라이브러리를 찾을 수 없습니다. 기술적 지표 계산이 불가능합니다. 서버 환경에 'pip install pandas-ta'로 설치해주세요.")

CSV_BASE_DIR = "data_csv_for_models" # settings에서 관리하는 것이 더 좋음
CSV_FILENAME_SUFFIX = "_TA_extended.csv" 
CSV_FILENAME_DATE_FORMAT = "%Y%m%d"

# KOSPI 모델 학습 시 사용된 34개 피처 목록 (views.py에서도 정의되지만, utils에서도 참조용으로 유지 가능)
KOSPI_TRAINED_FEATURES_FALLBACK = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'ATR_14', 'BBL_20_2.0',
    'BBM_20_2.0', 'BBU_20_2.0', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    'MACDs_12_26_9', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'OBV', 'ADX_14',
    'DMP_14', 'DMN_14', 'KOSPI_Close', 'KOSPI_Change', 'USD_KRW_Close',
    'USD_KRW_Change', 'Indi', 'Foreign', 'Organ', 'MarketCap', 'PBR', 'PER',
    'MarketCap_is_nan', 'PBR_is_nan', 'PER_is_nan', 'PER_is_zero'
]


def get_kr_holidays(years):
    """지정된 연도들에 대한 한국 공휴일 목록을 반환합니다."""
    kr_holidays_list = []
    if not isinstance(years, list):
        years = [years] # 단일 연도도 리스트로 처리
    for year_val in years:
        try:
            kr_holidays_list.extend(holidays.KR(years=year_val).keys())
        except KeyError: 
            print(f"[WARN][utils.get_kr_holidays] No holiday data found for year {year_val}. Skipping.")
    return [pd.to_datetime(d).date() for d in kr_holidays_list]


def get_market_macro_data(start_date_str, end_date_str, market_name='KOSPI', other_market_name_for_index=None):
    """지정된 기간 동안의 주요 시장 지수 및 환율 데이터를 가져옵니다."""
    df_market_macro = pd.DataFrame()
    main_market_symbol, other_market_symbol = None, None
    primary_market_context = market_name.upper()

    if primary_market_context == 'KOSPI':
        main_market_symbol = 'KS11'
        other_market_symbol = 'KQ11' if (other_market_name_for_index and other_market_name_for_index.upper() == 'KOSDAQ') or not other_market_name_for_index else None
        if not other_market_name_for_index and other_market_symbol: other_market_name_for_index = 'KOSDAQ'
    elif primary_market_context == 'KOSDAQ':
        main_market_symbol = 'KQ11'
        other_market_symbol = 'KS11' if (other_market_name_for_index and other_market_name_for_index.upper() == 'KOSPI') or not other_market_name_for_index else None
        if not other_market_name_for_index and other_market_symbol: other_market_name_for_index = 'KOSPI'
    else: 
        print(f"[WARN][utils.get_market_macro_data] Unexpected market_name '{market_name}'. Defaulting to KOSPI context.")
        main_market_symbol, primary_market_context = 'KS11', 'KOSPI' # 기본값 설정
        other_market_symbol, other_market_name_for_index = 'KQ11', 'KOSDAQ'
        
    try:
        if main_market_symbol:
            df_main = fdr.DataReader(main_market_symbol, start_date_str, end_date_str)
            if not df_main.empty:
                df_main['Change'] = df_main['Close'].pct_change() if 'Change' not in df_main.columns and 'Close' in df_main.columns else df_main.get('Change')
                df_market_macro = df_main[['Close', 'Change']].copy().rename(columns={'Close': f"{primary_market_context}_Close", 'Change': f"{primary_market_context}_Change"})

        if other_market_symbol and other_market_name_for_index:
            df_other = fdr.DataReader(other_market_symbol, start_date_str, end_date_str)
            if not df_other.empty:
                df_other['Change'] = df_other['Close'].pct_change() if 'Change' not in df_other.columns and 'Close' in df_other.columns else df_other.get('Change')
                df_other_processed = df_other[['Close', 'Change']].copy().rename(columns={'Close': f"{other_market_name_for_index.upper()}_Close", 'Change': f"{other_market_name_for_index.upper()}_Change"})
                df_market_macro = df_market_macro.join(df_other_processed, how='outer') if not df_market_macro.empty else df_other_processed
        
        df_usd = fdr.DataReader('USD/KRW', start_date_str, end_date_str)
        if not df_usd.empty:
            df_usd['Change'] = df_usd['Close'].pct_change() if 'Change' not in df_usd.columns and 'Close' in df_usd.columns else df_usd.get('Change')
            cols_usd = [col for col in ['Close', 'Change'] if col in df_usd.columns]
            if cols_usd:
                df_usd_p = df_usd[cols_usd].copy().rename(columns={c: f"USD_KRW_{c}" for c in cols_usd})
                df_market_macro = df_market_macro.join(df_usd_p, how='outer') if not df_market_macro.empty else df_usd_p
        
        if not df_market_macro.empty:
            # 인덱스 타입을 datetime.date로 변환
            if isinstance(df_market_macro.index, pd.DatetimeIndex):
                 df_market_macro.index = df_market_macro.index.date
            else: # 다른 타입일 경우 (예: object) 변환 시도
                 df_market_macro.index = pd.to_datetime(df_market_macro.index, errors='coerce').date
            df_market_macro = df_market_macro.ffill().bfill() # NaN 채우기
        return df_market_macro

    except Exception as e:
        print(f"[ERROR][utils.get_market_macro_data] Error fetching market/macro data: {e}\n{traceback.format_exc()}")
        # 오류 발생 시 예상되는 컬럼을 가진 빈 DataFrame 반환
        expected_cols = []
        if primary_market_context: expected_cols.extend([f"{primary_market_context}_Close", f"{primary_market_context}_Change"])
        if other_market_name_for_index: expected_cols.extend([f"{other_market_name_for_index.upper()}_Close", f"{other_market_name_for_index.upper()}_Change"])
        expected_cols.extend(['USD_KRW_Close', 'USD_KRW_Change'])
        return pd.DataFrame(columns=expected_cols)


def calculate_technical_indicators(df_ohlcv, pandas_ta_available=PANDAS_TA_AVAILABLE):
    """OHLCV DataFrame에 기술적 지표를 계산하여 추가합니다."""
    if not pandas_ta_available:
        print("[WARN][calculate_technical_indicators] pandas_ta is not available. Skipping TA calculation.")
        return df_ohlcv.copy()

    df_ta_input = df_ohlcv.copy()
    # pandas_ta는 소문자 컬럼명을 선호
    std_col_names = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
    df_ta_input.rename(columns=std_col_names, inplace=True, errors='ignore')

    required_cols_lower = ['open', 'high', 'low', 'close']
    if not all(col in df_ta_input.columns for col in required_cols_lower):
        missing_cols = [col for col in required_cols_lower if col not in df_ta_input.columns]
        print(f"[ERROR][calculate_technical_indicators] Required OHLC columns (lower case: {missing_cols}) are missing. Cannot calculate TA.")
        return df_ohlcv.copy() # 원본 반환

    # 숫자형으로 변환, volume은 없을 수도 있음
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_ta_input.columns:
            df_ta_input[col] = pd.to_numeric(df_ta_input[col], errors='coerce')
    
    # volume이 float64인지 확인하고, 없거나 모두 NaN이면 float64 NaN 시리즈로 생성
    if 'volume' in df_ta_input.columns and df_ta_input['volume'].notna().any():
        df_ta_input['volume'] = df_ta_input['volume'].astype('float64')
    else:
        df_ta_input['volume'] = pd.Series(np.nan, index=df_ta_input.index, dtype='float64')


    if 'close' not in df_ta_input or df_ta_input['close'].isnull().all():
        print("[WARN][calculate_technical_indicators] All 'close' prices are NaN or column missing. TA calculations will result in NaNs.")
        return df_ohlcv.copy()

    df_with_ta_features = df_ohlcv.copy() # 원본 컬럼명 유지하며 결과 저장

    try:
        # ATR
        df_with_ta_features['ATR_14'] = ta.atr(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        # Bollinger Bands
        bbands = ta.bbands(df_ta_input['close'], length=20, std=2.0)
        if bbands is not None and not bbands.empty:
            for col_name in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']:
                df_with_ta_features[col_name] = bbands.get(col_name)
        # RSI
        for length in [14, 6, 28]: # 14가 주요, 나머지는 추가적
            df_with_ta_features[f'RSI_{length}'] = ta.rsi(df_ta_input['close'], length=length)
        # MACD
        macd = ta.macd(df_ta_input['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            for col_name in ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']:
                df_with_ta_features[col_name] = macd.get(col_name)
        # Stochastic
        stoch_slow = ta.stoch(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], k=14, d=3, smooth_k=3)
        if stoch_slow is not None and not stoch_slow.empty:
            df_with_ta_features['STOCHk_14_3_3'] = stoch_slow.get('STOCHk_14_3_3')
            df_with_ta_features['STOCHd_14_3_3'] = stoch_slow.get('STOCHd_14_3_3')
        stoch_fast = ta.stoch(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], k=14, d=3, smooth_k=1)
        if stoch_fast is not None and not stoch_fast.empty:
            df_with_ta_features['STOCHk_fast_14_3_1'] = stoch_fast.get(f'STOCHk_14_3_1') # pandas_ta 컬럼명 확인 필요
            df_with_ta_features['STOCHd_fast_14_3_1'] = stoch_fast.get(f'STOCHd_14_3_1')
        # OBV
        if df_ta_input['volume'].notna().any(): # volume 데이터가 있어야 계산 가능
            df_with_ta_features['OBV'] = ta.obv(df_ta_input['close'], df_ta_input['volume'])
        else:
            df_with_ta_features['OBV'] = np.nan
        # ADX, DMP, DMN
        adx_df = ta.adx(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            for col_name in ['ADX_14', 'DMP_14', 'DMN_14']:
                df_with_ta_features[col_name] = adx_df.get(col_name)
        # 이동평균선 (MA, EMA)
        for length in [5, 10, 20, 60, 120]:
            df_with_ta_features[f'MA_{length}'] = ta.sma(df_ta_input['close'], length=length)
            df_with_ta_features[f'EMA_{length}'] = ta.ema(df_ta_input['close'], length=length)
        # 기타 지표들
        df_with_ta_features['CCI_14_0.015'] = ta.cci(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14, c=0.015)
        if df_ta_input['volume'].notna().any():
            mfi_series = ta.mfi(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], df_ta_input['volume'].astype('float64'), length=14)
            df_with_ta_features['MFI_14'] = mfi_series.astype('float64') # float64로 명시적 변환
        else:
            df_with_ta_features['MFI_14'] = np.nan
        df_with_ta_features['WILLR_14'] = ta.willr(df_ta_input['high'], df_ta_input['low'], df_ta_input['close'], length=14)
        df_with_ta_features['MOM_10'] = ta.mom(df_ta_input['close'], length=10)
        df_with_ta_features['ROC_10'] = ta.roc(df_ta_input['close'], length=10)
        trix_df = ta.trix(df_ta_input['close'], length=14, signal=9) 
        if trix_df is not None and not trix_df.empty:
             df_with_ta_features['TRIX_14_9'] = trix_df.get('TRIX_14_9') # 또는 TRIXs_14_9_9
        if hasattr(ta, 'vr') and df_ta_input['volume'].notna().any(): # vr 함수 존재 및 volume 데이터 확인
            vr_series = ta.vr(df_ta_input['close'], df_ta_input['volume'].astype('float64'), length=20)
            df_with_ta_features['VR_20'] = vr_series.astype('float64')
        else: df_with_ta_features['VR_20'] = np.nan
        if hasattr(ta, 'psl'): # psl 함수 존재 확인
            df_with_ta_features['PSL_12'] = ta.psl(df_ta_input['close'], length=12)
        else: df_with_ta_features['PSL_12'] = np.nan
        
        # 로그 변환된 값 (모델 입력 피처로 사용될 수 있음)
        df_with_ta_features['Log_Close'] = np.log1p(df_ta_input['close'].clip(lower=0)).astype('float64')
        if df_ta_input['volume'].notna().any():
            log_volume_series = np.log1p(df_ta_input['volume'].fillna(0).clip(lower=0)) # fillna(0) for log1p of volume
            df_with_ta_features['Log_Volume'] = log_volume_series.astype('float64')
        else:
            df_with_ta_features['Log_Volume'] = np.nan
        
        df_with_ta_features.replace([np.inf, -np.inf], np.nan, inplace=True) # 로그 변환 후 inf 값 처리

    except Exception as e_ta:
        print(f"[ERROR][calculate_technical_indicators] Error during TA calculation: {e_ta}\n{traceback.format_exc()}")
        # 오류 발생 시에도 컬럼은 존재하도록 NaN으로 채울 수 있으나, 보통은 원본을 반환하거나 예외를 다시 발생시킴
    return df_with_ta_features


def calculate_all_features(stock_df_ohlcv, market_macro_data_df, investor_df, fundamental_df, pandas_ta_available=PANDAS_TA_AVAILABLE):
    """모든 피처를 계산하고 병합합니다."""
    if stock_df_ohlcv is None or stock_df_ohlcv.empty:
        print("[WARN][calculate_all_features] stock_df_ohlcv is empty. Cannot calculate features.")
        return pd.DataFrame()

    # 인덱스 타입 datetime.date로 통일
    def ensure_date_index(df_to_check, df_name="DataFrame"):
        if df_to_check is not None and not df_to_check.empty:
            if not (isinstance(df_to_check.index, pd.DatetimeIndex) and all(isinstance(i, date_type) for i in df_to_check.index if pd.notna(i))):
                try:
                    current_index_type = type(df_to_check.index)
                    if isinstance(df_to_check.index, pd.DatetimeIndex): # DatetimeIndex -> date
                        df_to_check.index = df_to_check.index.date
                    # object나 string 타입 인덱스 처리 강화
                    elif pd.api.types.is_object_dtype(df_to_check.index) or pd.api.types.is_string_dtype(df_to_check.index) or not all(isinstance(i, date_type) for i in df_to_check.index if pd.notna(i)):
                        df_to_check.index = pd.to_datetime(df_to_check.index, errors='coerce').date
                except Exception as e_idx_convert:
                    print(f"[WARN][ensure_date_index] Failed to convert index of {df_name} to date objects: {e_idx_convert}. Original index type: {current_index_type}, Index head: {df_to_check.index[:2]}")
        return df_to_check

    stock_df_ohlcv = ensure_date_index(stock_df_ohlcv, "stock_df_ohlcv")
    market_macro_data_df = ensure_date_index(market_macro_data_df, "market_macro_data_df")
    investor_df = ensure_date_index(investor_df, "investor_df")
    fundamental_df = ensure_date_index(fundamental_df, "fundamental_df")

    df_with_ta = calculate_technical_indicators(stock_df_ohlcv, pandas_ta_available)
    final_df = df_with_ta.copy()

    # 데이터프레임 병합
    for df_to_join, suffix, cols_to_ensure_if_empty in [
        (market_macro_data_df, '_macro_duplicate', ['KOSPI_Close', 'KOSPI_Change', 'KOSDAQ_Close', 'KOSDAQ_Change', 'USD_KRW_Close', 'USD_KRW_Change']),
        (investor_df, '_investor_duplicate', ['Indi', 'Foreign', 'Organ']),
        (fundamental_df, '_fundamental_duplicate', ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'ROE'])
    ]:
        if df_to_join is not None and not df_to_join.empty:
            final_df = final_df.join(df_to_join, how='left', rsuffix=suffix)
        else: # 병합할 데이터가 없으면, 해당 컬럼들이 존재하도록 NaN으로 채움
            for col_name in cols_to_ensure_if_empty:
                if col_name not in final_df.columns:
                    final_df[col_name] = np.nan
            
    # 중복된 이름의 컬럼 정리 (rsuffix로 인해 발생한 경우)
    cols_to_drop = [col for col in final_df.columns if '_duplicate' in col]
    final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
            
    # 인덱스 중복 제거 (병합 과정에서 발생 가능성)
    if final_df.index.has_duplicates:
        final_df = final_df[~final_df.index.duplicated(keep='first')] # 일반적으로 'first' 유지

    # 최종적으로 모든 컬럼에 대해 Python None을 np.nan으로 변환하고, 숫자형으로 시도
    for col in final_df.columns:
        if final_df[col].dtype == 'object': # object 타입 컬럼에 대해
            # None 값을 np.nan으로 먼저 바꾸고 숫자형 변환 시도
            try:
                # Series.replace(None, np.nan)은 효과가 없을 수 있음. apply 사용.
                final_df[col] = final_df[col].apply(lambda x: np.nan if x is None else x)
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            except Exception as e_final_conv: # 만약 to_numeric 변환 중 다른 오류 발생 시
                print(f"Warning: Final conversion of column {col} to numeric failed. Error: {e_final_conv}")
                # 이미 None은 np.nan으로 바뀌었으므로 추가 처리는 불필요할 수 있음
    return final_df


def add_fundamental_indicator_features(df):
    """펀더멘털 기반 파생 플래그 피처를 추가합니다."""
    df_processed = df.copy()
    
    # PER 관련 플래그
    if 'PER' in df_processed.columns:
        # PER 컬럼을 숫자형으로 먼저 변환 (이미 되어있을 수 있지만 안전하게)
        per_numeric = pd.to_numeric(df_processed['PER'], errors='coerce')
        df_processed['PER_is_nan'] = per_numeric.isna().astype(int)
        df_processed['PER_is_zero'] = (per_numeric == 0).astype(int)
        # 사용자가 요청한 34개 피처 목록에 PER_is_high, PER_is_low는 없음
    else: 
        df_processed['PER_is_nan'] = 1
        df_processed['PER_is_zero'] = 0

    # PBR 관련 플래그
    if 'PBR' in df_processed.columns:
        pbr_numeric = pd.to_numeric(df_processed['PBR'], errors='coerce')
        df_processed['PBR_is_nan'] = pbr_numeric.isna().astype(int)
        # 사용자가 요청한 34개 피처 목록에 PBR_is_zero, PBR_is_high, PBR_is_low는 없음
    else: 
        df_processed['PBR_is_nan'] = 1
        
    # MarketCap 관련 플래그
    if 'MarketCap' in df_processed.columns:
        mc_numeric = pd.to_numeric(df_processed['MarketCap'], errors='coerce')
        df_processed['MarketCap_is_nan'] = mc_numeric.isna().astype(int)
    else: 
        df_processed['MarketCap_is_nan'] = 1
        
    return df_processed


def get_feature_columns_for_model(market_name, model_type='technical'):
    """
    모델 학습에 사용될 피처 컬럼 목록을 반환합니다. (주로 fallback 용도)
    실제 예측 시에는 views.py의 DEFAULT_MODEL_PARAMS에 정의된 리스트를 우선 사용합니다.
    """
    if market_name.upper() == 'KOSPI' and model_type == 'technical':
        # KOSPI technical 모델의 경우, 사용자가 제공한 34개 피처 목록을 기본으로 반환
        return KOSPI_TRAINED_FEATURES_FALLBACK.copy()

    # 그 외의 경우, 일반적인 기술적 분석 피처 목록 구성 (예시)
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    technical_indicator_features = [ # 광범위한 TA 목록
        'MA_5', 'MA_10', 'MA_20', 'MA_60', 'MA_120', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_60', 'EMA_120',
        'BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'RSI_6', 'RSI_14', 'RSI_28',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCHk_fast_14_3_1', 'STOCHd_fast_14_3_1', 
        'ATR_14', 'ADX_14', 'DMP_14', 'DMN_14', 'CCI_14_0.015', 'MFI_14', 'OBV', 'WILLR_14',
        'MOM_10', 'ROC_10', 'TRIX_14_9', 'VR_20', 'PSL_12', 'Log_Close', 'Log_Volume'
    ]
    
    main_market_context = market_name.upper()
    other_market_context = 'KOSDAQ' if main_market_context == 'KOSPI' else 'KOSPI'

    market_macro_features = [
        f"{main_market_context}_Close", f"{main_market_context}_Change",
        f"{other_market_context}_Close", f"{other_market_context}_Change", 
        'USD_KRW_Close', 'USD_KRW_Change',
    ]

    feature_columns = []
    if model_type == 'technical':
        feature_columns = base_features + technical_indicator_features + market_macro_features
    elif model_type == 'comprehensive': # 'lstm' 또는 다른 일반 모델 타입에 해당될 수 있음
        investor_features = ['Indi', 'Foreign', 'Organ'] 
        fundamental_value_features = ['MarketCap', 'PBR', 'PER', 'EPS', 'BPS', 'DPS', 'ROE'] 
        fundamental_flag_features = [ # add_fundamental_indicator_features에서 생성됨
            'PER_is_nan', 'PER_is_zero', 
            'PBR_is_nan', 
            'MarketCap_is_nan'
        ]
        feature_columns = (base_features + technical_indicator_features + market_macro_features +
                           investor_features + fundamental_value_features + fundamental_flag_features)
    else: # 알 수 없는 타입이면 기술적 분석 기준으로
        feature_columns = base_features + technical_indicator_features + market_macro_features
    
    return list(dict.fromkeys(feature_columns)) # 중복 제거


def get_krx_stock_list(market='KOSPI,KOSDAQ', cache_ttl_seconds=3600):
    """KRX 전체 또는 특정 시장의 종목 목록을 반환합니다."""
    cache_key = f"krx_stock_list_{market.replace(',', '_')}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return cached_data
    try:
        krx_df = fdr.StockListing('KRX') # 전체 KRX 목록 조회
        
        markets_to_include_list = [m.strip().upper() for m in market.split(',') if m.strip()]
        
        # 'ALL'이 아니거나 markets_to_include_list가 비어있지 않으면 필터링
        if 'ALL' not in markets_to_include_list and markets_to_include_list:
            krx_df_filtered = krx_df[krx_df['Market'].isin(markets_to_include_list)]
        else: # 'ALL'이거나 market 인자가 없으면 전체 사용
            krx_df_filtered = krx_df
        
        # 필수 컬럼 확인
        if not all(col in krx_df_filtered.columns for col in ['Code', 'Name', 'Market']):
            print(f"[ERROR][get_krx_stock_list] FDR StockListing missing required columns for market: {market}")
            return []

        stock_list = krx_df_filtered[['Code', 'Name', 'Market']].to_dict('records')
        cache.set(cache_key, stock_list, cache_ttl_seconds)
        return stock_list
    except Exception as e:
        print(f"[ERROR][get_krx_stock_list] Failed to fetch stock list from FDR for market '{market}': {e}")
        return []

def get_csv_path_for_stock(stock_code, market_name, base_dir=None):
    """특정 종목의 최신 CSV 파일 경로를 찾습니다. (현재 예측 로직에서는 주로 DB 사용)"""
    if base_dir is None:
        base_dir = getattr(settings, 'CSV_BASE_DIR_PATH', CSV_BASE_DIR) # settings에 정의된 경로 사용 시도
    
    market_path_segment = "KOSPI_Market_Data_CSV" if market_name.upper() == "KOSPI" else "KOSDAQ_Market_Data_CSV"
    # 실제 디렉토리 구조에 맞게 수정 필요 (예: predict_info/data_csv_for_models/KOSPI_Market_Data_CSV)
    # stock_specific_dir = os.path.join(settings.BASE_DIR, 'predict_info', base_dir, market_path_segment) 
    stock_specific_dir = os.path.join(base_dir, market_path_segment) # base_dir가 이미 전체 경로를 포함한다고 가정
    
    filename_pattern = f"{stock_code}_{market_name.lower()}_daily_from_*_to_*{CSV_FILENAME_SUFFIX}"
    search_path = os.path.join(stock_specific_dir, filename_pattern)
    
    matching_files = glob.glob(search_path)
    if not matching_files:
        return None

    latest_file = None
    latest_to_date_in_filename = None

    for f_path in matching_files:
        try:
            filename_only = os.path.basename(f_path)
            parts = filename_only.replace(CSV_FILENAME_SUFFIX, "").split('_')
            if len(parts) >= 6 and parts[-2] == "to": # 파일명 형식: CODE_market_daily_from_YYYYMMDD_to_YYYYMMDD
                date_str_part = parts[-1] 
                current_file_date = datetime.strptime(date_str_part, CSV_FILENAME_DATE_FORMAT).date()
                if latest_to_date_in_filename is None or current_file_date > latest_to_date_in_filename:
                    latest_to_date_in_filename = current_file_date
                    latest_file = f_path
            elif latest_file is None: # 형식 안 맞으면 일단 첫번째 파일
                latest_file = f_path
        except (ValueError, IndexError): 
            if latest_file is None: latest_file = f_path # 오류 시에도 첫번째 파일
            continue
            
    return latest_file if latest_file else (matching_files[0] if matching_files else None)


def get_future_trading_dates_list(base_date, num_days, kr_holidays_list=None):
    """주어진 기준일로부터 미래 num_days 만큼의 거래일을 리스트로 반환합니다 (기준일 미포함)."""
    if not isinstance(base_date, (datetime, date_type)):
        try: base_date = pd.to_datetime(base_date).date()
        except Exception: raise ValueError("base_date must be a datetime object or convertible to one.")

    if kr_holidays_list is None:
        current_year = base_date.year
        years_to_fetch_holidays = list(range(current_year, current_year + int(num_days / 200) + 3)) # 대략적인 연도 범위
        kr_holidays_list = get_kr_holidays(years_to_fetch_holidays)
    
    future_dates = []
    current_date_iter = base_date # 기준일 다음날부터 시작
    while len(future_dates) < num_days:
        current_date_iter += timedelta(days=1)
        if current_date_iter.weekday() < 5 and current_date_iter not in kr_holidays_list: # 주말(토:5,일:6) 아니고, 공휴일 아니면
            future_dates.append(current_date_iter)
    return future_dates

def get_past_trading_dates_list(base_date, num_days, kr_holidays_list=None):
    """주어진 기준일로부터 과거 num_days 만큼의 거래일을 리스트로 반환합니다 (기준일 미포함, 날짜 오름차순)."""
    if not isinstance(base_date, (datetime, date_type)):
        try: base_date = pd.to_datetime(base_date).date()
        except Exception: raise ValueError("base_date must be a datetime object or convertible to one.")

    if kr_holidays_list is None:
        # 과거 날짜를 포함하도록 연도 범위 설정 (넉넉하게)
        years_range = list(range(base_date.year - int(num_days / 200) - 2, base_date.year + 1))
        kr_holidays_list = get_kr_holidays(years_range)
    
    past_dates = []
    current_dt_iter = base_date # 기준일 이전날부터 시작
    while len(past_dates) < num_days:
        current_dt_iter -= timedelta(days=1)
        if current_dt_iter.weekday() < 5 and current_dt_iter not in kr_holidays_list:
            past_dates.append(current_dt_iter)
    return sorted(past_dates) # 날짜 오름차순으로 정렬하여 반환
