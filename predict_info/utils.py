# predict_info/utils.py
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import traceback # 상세한 오류 로깅을 위해 추가

# pandas-ta 라이브러리 임포트
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    print("[INFO][predict_info/utils.py] pandas_ta 라이브러리가 성공적으로 로드되었습니다.")
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("[CRITICAL ERROR][predict_info/utils.py] 'pandas_ta' 라이브러리를 찾을 수 없습니다. 기술적 지표 계산이 불가능합니다. 서버 환경에 'pip install pandas-ta'로 설치해주세요.")

def get_market_macro_data(start_date, end_date, market_fdr_code='KS11', currency_pair='USD/KRW'):
    """
    지정된 기간의 시장 지수 및 환율 데이터를 가져옵니다.
    market_fdr_code: FDR에서 사용하는 시장 티커 (예: 'KS11' for KOSPI, 'KQ11' for KOSDAQ)
    반환되는 DataFrame의 시장 지표 컬럼명은 'KOSPI_Close', 'KOSDAQ_Change' 등으로 표준화됩니다.
    """
    df_market_processed = pd.DataFrame()
    df_macro_processed = pd.DataFrame()
    
    actual_market_prefix = "KOSPI" if market_fdr_code == 'KS11' else \
                           "KOSDAQ" if market_fdr_code == 'KQ11' else \
                           market_fdr_code 

    print(f"[DEBUG][utils.get_market_macro_data] 시장 데이터 조회 시작: {actual_market_prefix} ({market_fdr_code}), 기간: {start_date} ~ {end_date}")
    try:
        # 시장 지수 데이터
        df_market_raw = fdr.DataReader(market_fdr_code, start_date, end_date)
        if not df_market_raw.empty:
            df_market_processed[f'{actual_market_prefix}_Close'] = df_market_raw['Close']
            df_market_processed[f'{actual_market_prefix}_Change'] = df_market_raw['Change'] 
            df_market_processed.index = pd.to_datetime(df_market_raw.index)
            print(f"[DEBUG][utils.get_market_macro_data] {actual_market_prefix} 지수 데이터 {len(df_market_processed)}건 로드 완료.")
        else:
            print(f"[WARNING][utils.get_market_macro_data] {actual_market_prefix} 지수 데이터를 찾을 수 없습니다. ({start_date} ~ {end_date})")

        # 환율 데이터 (USD/KRW)
        df_macro_raw = fdr.DataReader(currency_pair, start_date, end_date)
        if not df_macro_raw.empty:
            df_macro_processed['USD_KRW_Close'] = df_macro_raw['Close']
            df_macro_processed['USD_KRW_Change'] = df_macro_raw['Close'].pct_change()
            df_macro_processed.index = pd.to_datetime(df_macro_raw.index)
            print(f"[DEBUG][utils.get_market_macro_data] 환율 데이터({currency_pair}) {len(df_macro_processed)}건 로드 완료.")
        else:
            print(f"[WARNING][utils.get_market_macro_data] 환율 데이터({currency_pair})를 찾을 수 없습니다. ({start_date} ~ {end_date})")

    except Exception as e:
        print(f"[ERROR][utils.get_market_macro_data] 시장/거시 데이터 가져오기 중 오류 발생: {e}")
        # traceback.print_exc() # 디버깅 시 상세 오류 확인
    
    return df_market_processed, df_macro_processed


def calculate_all_features(df_input_with_ohlcv_extras, market_name_upper="KOSPI"):
    """
    주어진 DataFrame (OHLCV + 외부 데이터)에 pandas-ta를 사용하여 모든 기술적 지표를 계산합니다.
    """
    if not PANDAS_TA_AVAILABLE:
        print("[CRITICAL ERROR][utils.calculate_all_features] pandas_ta 라이브러리가 없어 기술적 지표를 계산할 수 없습니다. 피처가 누락됩니다.")
        return df_input_with_ohlcv_extras # TA 컬럼 없이 반환

    if df_input_with_ohlcv_extras.empty:
        print("[WARNING][utils.calculate_all_features] 입력 DataFrame이 비어있어 TA 계산 불가.")
        return df_input_with_ohlcv_extras

    df = df_input_with_ohlcv_extras.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e_idx:
            print(f"[ERROR][utils.calculate_all_features] DataFrame 인덱스 Datetime 변환 오류: {e_idx}. TA 계산 불가.")
            return df_input_with_ohlcv_extras

    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
    if missing_ohlcv:
        print(f"[ERROR][utils.calculate_all_features] TA 계산에 필요한 기본 OHLCV 컬럼 누락: {missing_ohlcv}. TA 계산 불가.")
        return df

    print(f"[DEBUG][utils.calculate_all_features] TA 계산 시작. 입력 df shape: {df.shape}")
    try:
        # pandas-ta 전략 사용 예시 (여러 지표를 한 번에 계산)
        # 학습 시 사용한 지표와 파라미터를 정확히 맞춰야 함.
        # 예시: MyStrategy = ta.Strategy(...) df.ta.strategy(MyStrategy)
        # 여기서는 개별적으로 호출

        # ATR
        df.ta.atr(length=14, append=True) 

        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True) 

        # RSI
        df.ta.rsi(length=14, append=True) 

        # MACD
        df.ta.macd(fast=12, slow=26, signal=9, append=True) 

        # Stochastic Oscillator %K, %D
        stoch_df = df.ta.stoch(k=14, d=3, smooth_k=3, append=False)
        if stoch_df is not None and not stoch_df.empty:
            # pandas-ta 버전에 따라 컬럼명이 STOCHk_14_3_3 또는 STOCHk_14_3_0_3_0 등으로 나올 수 있음.
            # 학습 시 사용된 정확한 컬럼명으로 맞춰야 함.
            # 여기서는 views.py의 NEW_TA_COLS에 정의된 'STOCHk_14_3_3', 'STOCHd_14_3_3'를 가정.
            if 'STOCHk_14_3_3' in stoch_df.columns and 'STOCHd_14_3_3' in stoch_df.columns:
                 df = pd.concat([df, stoch_df[['STOCHk_14_3_3', 'STOCHd_14_3_3']]], axis=1)
            else: # 만약 컬럼명이 다르면, 여기서 rename 또는 다른 방식으로 처리 필요
                print(f"[WARNING][utils.calculate_all_features] Stochastic Oscillator 결과 컬럼명이 예상과 다릅니다: {stoch_df.columns.tolist()}")
                # 임시로 첫 두 컬럼을 사용하거나, 더 정확한 매핑 필요
                # df['STOCHk_14_3_3'] = stoch_df.iloc[:, 0]
                # df['STOCHd_14_3_3'] = stoch_df.iloc[:, 1]
        else:
            print("[WARNING][utils.calculate_all_features] Stochastic Oscillator 계산 결과가 비어있거나 None입니다.")
            df['STOCHk_14_3_3'] = np.nan # 누락 시 NaN으로 명시적 추가
            df['STOCHd_14_3_3'] = np.nan


        # OBV (On-Balance Volume)
        df.ta.obv(append=True) # 생성되는 컬럼명 확인 필요 (OBV 또는 OBV_EMA_xx 등)
                               # views.py의 NEW_TA_COLS에 'OBV'로 정의되어 있으므로, 필요시 rename
        if 'OBV' not in df.columns and any(col.startswith('OBV_') for col in df.columns): # 예시: OBV_EMA_10
            obv_col_found = [col for col in df.columns if col.startswith('OBV_')][0]
            df.rename(columns={obv_col_found: 'OBV'}, inplace=True)
            print(f"[DEBUG][utils.calculate_all_features] OBV 컬럼명 변경: {obv_col_found} -> OBV")
        elif 'OBV' not in df.columns:
            print("[WARNING][utils.calculate_all_features] OBV 컬럼이 생성되지 않았습니다.")
            df['OBV'] = np.nan


        # ADX (Average Directional Movement Index)
        adx_df = df.ta.adx(length=14, append=False)
        if adx_df is not None and not adx_df.empty:
            # views.py의 NEW_TA_COLS에 'ADX_14', 'DMP_14', 'DMN_14'로 정의됨
            cols_to_add_adx = ['ADX_14', 'DMP_14', 'DMN_14']
            missing_adx_cols = [col for col in cols_to_add_adx if col not in adx_df.columns]
            if not missing_adx_cols:
                df = pd.concat([df, adx_df[cols_to_add_adx]], axis=1)
            else:
                print(f"[WARNING][utils.calculate_all_features] ADX 관련 예상 컬럼 누락: {missing_adx_cols}. 사용 가능한 컬럼만 추가 시도.")
                available_adx_cols = [col for col in cols_to_add_adx if col in adx_df.columns]
                if available_adx_cols:
                    df = pd.concat([df, adx_df[available_adx_cols]], axis=1)
                for m_col in missing_adx_cols: df[m_col] = np.nan # 누락된 ADX 컬럼은 NaN으로
        else:
            print("[WARNING][utils.calculate_all_features] ADX 계산 결과가 비어있거나 None입니다.")
            for col in ['ADX_14', 'DMP_14', 'DMN_14']: df[col] = np.nan # 누락 시 NaN으로

        print(f"[INFO][utils.calculate_all_features] pandas_ta 기술적 지표 계산 완료. df shape: {df.shape}")

    except Exception as e:
        print(f"[ERROR][utils.calculate_all_features] pandas_ta 지표 계산 중 예외 발생: {e}")
        traceback.print_exc() 
        # 오류 발생 시, TA 컬럼들이 부분적으로만 생성되거나 아예 없을 수 있음.
        # 호출부에서 최종 피처 목록을 기준으로 누락 여부를 다시 한번 점검해야 함.
    
    return df
