# chart/data_fetcher.py

import FinanceDataReader as fdr
import pandas as pd
import pandas_ta as ta
import os

# 저장 디렉토리
save_dir = './chart/data'
os.makedirs(save_dir, exist_ok=True)

# 가져올 종목 리스트 (이름: 티커)
stocks = {
    '삼성전자': '005930',
    '카카오': '035720',
    '네이버': '035420',
}


# 시작/종료 기간 설정
start_date = '2020-04-01'
end_date = '2025-04-01'




# 반복 다운로드 및 저장
for name, code in stocks.items():
    try:
        # 데이터 불러오기
        df = fdr.DataReader(code, start_date, end_date)
        df.index.name = 'datetime'
        
        # 결측값 처리: NaN을 앞의 값으로 채우기
        df = df.ffill()  # fillna() 대신 ffill() 사용
        
        # 컬럼 이름 변경
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # 지표 추가 없이 데이터 저장
        save_path = os.path.join(save_dir, f'{name}_5min_50month.csv')
        df.to_csv(save_path, encoding='utf-8-sig')
        
        print(f"[✔] {name} 데이터 저장 완료 → {save_path}")
    except Exception as e:
        print(f"[X] {name} 저장 실패: {e}")