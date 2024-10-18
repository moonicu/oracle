import streamlit as st
import joblib
import pandas as pd
import os
 
# 모델 저장 경로 설정
model_save_dir = 'saved_models'

# 불러올 모델들의 이름과 예측할 타겟 (Y) 컬럼
model_names = ['RandomForest', 'XGBoost', 'LightGBM']  # 예시로 사용할 모델 이름들
Y_columns = ['RESU', 'RESUO', 'RESUP', 'RESUI', 'RESUH', 'RESUE', 'RESUC', 
                     'RDS', 'SFT', 'SFTW', 'SFTTH_M', 'DTH', 'SUPYN']

# 학습에 사용한 특성 이름 (X_columns) 순서
X_columns = ['MAGE', 'GRAN', 'PARN', 'AMNI', 'MULG', 'BIR', 'PREP', 'DM', 
             'HTN', 'CHOR', 'PROM', 'PROM_H', 'STER', 'STERP', 'ATBYN', 
             'DELM', 'GAD', 'BWEI']

# 변수명과 항목명 간의 매핑 정보 (예시)
variable_to_display_name = {
    'RESU': '분만시 소생술 필요 여부',
    'RESUO': '소생술 중 산소',
    'RESUP': '소생술 중 PPV',
    'RESUI': '소생술 중 Intubation',
    'RESUH': '소생술 중 Chest compression',
    'RESUE': '소생술 중 Epinephrine',
    'RESUC': '소생술 중 CPAP',
    'RDS': '신생아 호흡곤란증후군(RDS)',
    'SFT': '폐표면활성제 사용 여부',
    'SFTW': '폐표면활성제 분만장 투여',
    'SFTTH_M': '폐표면활성제 투여후 NIV(INSURE,LISA)',
    'DTH': '사망',
    'SUPYN': '퇴원시 의료 보조 장비 필요'
}

# Streamlit 앱 제목
st.title("NICU 환자 예측 모델")

# 입력 데이터 수집
st.header("입력 데이터")

# 입력받을 값들
GA_w = st.number_input("임신 나이 (주, GA_w)", min_value=20, max_value=45, value=28)
GA_d = st.number_input("임신 나이 (일, GA_d)", min_value=0, max_value=6, value=4)
GAD = (GA_w * 7) + GA_d  # GA_w와 GA_d로 GAD 값 계산
BWEI = st.number_input("출생 체중 (BWEI, g)", min_value=200, max_value=1500, value=1000)
DELM = st.selectbox("분만 방식 (DELM)", [0, 1], format_func=lambda x: "자연" if x == 0 else "제왕절개")
MAGE = st.number_input("산모 나이 (MAGE)", min_value=18, max_value=50, value=30)
GRAN = st.number_input("임신력 (GRAN)", min_value=0, max_value=10, value=1)
PARN = st.number_input("출산력 (PARN)", min_value=0, max_value=10, value=1)
AMNI = st.selectbox("양수 상태 (AMNI)", [0, 1], format_func=lambda x: "정상" if x == 0 else "비정상")
MULG = st.selectbox("다태아 여부 (MULG)", [1, 2], format_func=lambda x: "단태" if x == 1 else "다태")
BIR = st.number_input("출생 순서 (BIR)", min_value=1, max_value=10, value=1)
PREP = st.selectbox("분만 준비 상태 (PREP)", [0, 1], format_func=lambda x: "미비" if x == 0 else "준비됨")
DM = st.selectbox("당뇨병 (DM)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
HTN = st.selectbox("고혈압 (HTN)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
CHOR = st.selectbox("조직학적 융모양막염 (CHOR)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
PROM = st.selectbox("조기 양막 파열 (PROM)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
PROM_H = st.number_input("양막 파열 시간 (PROM_H)", min_value=0, max_value=48, value=0)
STER = st.selectbox("스테로이드 사용 여부 (STER)", [0, 1], format_func=lambda x: "사용 안함" if x == 0 else "사용")
STERP = st.selectbox("스테로이드 완료 여부 (STERP)", [0, 1], format_func=lambda x: "미완료" if x == 0 else "완료")
ATBYN = st.selectbox("항생제 사용 여부 (ATBYN)", [0, 1], format_func=lambda x: "사용 안함" if x == 0 else "사용")

# 입력 데이터를 X_columns 순서대로 바로 구성 (실제 입력값을 사용)
new_X_data = pd.DataFrame([[MAGE, GRAN, PARN, AMNI, MULG, BIR, PREP, DM, HTN, CHOR, PROM, PROM_H, STER, STERP, ATBYN, DELM, GAD, BWEI]], columns=X_columns)

# 모델 예측 수행
st.header("예측 결과")
if st.button("결과 예측"):
    results = []

    # 결과를 행이 Y_column, 열이 모델 이름으로 구성된 테이블로 정리
    result_table = pd.DataFrame(index=Y_columns, columns=model_names)

    # 각 모델에 대해 예측 수행
    for model_name in model_names:
        for y_col in Y_columns:
            # 저장된 모델 경로
            model_filename = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")

            try:
                # 모델 불러오기
                model = joblib.load(model_filename)

                # 예측 수행 (퍼센트로 변환)
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(new_X_data)
                    pred_percent = pred_proba[:, 1] * 100  # 클래스 1의 예측 확률을 퍼센트로 변환
                    
                    # 결과를 테이블에 저장
                    result_table.at[y_col, model_name] = f"{pred_percent[0]:.2f}%"
                else:
                    result_table.at[y_col, model_name] = "N/A"  # 예측 불가한 경우
            except FileNotFoundError:
                st.error(f"모델 파일 '{model_filename}'을 찾을 수 없습니다.")
                result_table.at[y_col, model_name] = "N/A"


    # 입력 데이터에 대한 매핑을 적용하여 항목명으로 변경
    result_table_display = result_table.rename(index=variable_to_display_name)

    # 결과 출력 (Streamlit 데이터프레임으로 표시)
    st.header("예측 결과")
    st.dataframe(result_table_display, height=500, width=800)