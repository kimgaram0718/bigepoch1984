from lstm_predictor import LSTMPredictor
import FinanceDataReader as fdr
import os

# ✅ 학습할 종목별 CSV 파일 경로 설정
target_files = {
    '삼성전자': './chart/data/삼성전자_5min_50month.csv',
    '카카오': './chart/data/카카오_5min_50month.csv',
    '네이버': './chart/data/네이버_5min_50month.csv',
    # 추가 종목 계속 확장 가능
}

# ✅ 모델 저장 디렉토리
model_dir = 'saved_model'
os.makedirs(model_dir, exist_ok=True)



# 종목별 학습 및 저장 루프
for name, path in target_files.items():
    try:
        print(f"[{name}] 데이터 로딩 및 전처리 중...")
        predictor = LSTMPredictor()
        predictor.load_and_preprocess({name: path})
        X, y = predictor.build_sequences()

        print(f"[{name}] 모델 학습 시작 (총 샘플: {len(X)})")
        predictor.train_model(X, y, epochs=10, batch_size=64)

        model_path = os.path.join(model_dir, f"{name}_model.keras")
        scaler_path = os.path.join(model_dir, f"{name}_scaler.pkl")

        predictor.save_model(model_path)
        predictor.save_scaler(scaler_path)

        print(f"[{name}] 모델 저장 완료 → {model_path}")
    except Exception as e:
        print(f"[{name}] ❌ 학습 실패: {e}")
        
        