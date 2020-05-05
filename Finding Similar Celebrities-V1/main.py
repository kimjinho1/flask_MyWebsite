from application import app

if __name__ == "__main__":
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run()