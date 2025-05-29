import streamlit as st
import joblib
import pandas as pd
import os
import io
from datetime import datetime

model_save_dir = 'saved_models'

# 변수 목록
display_columns = ['gaw', 'gawd'] + ['gad', 'bwei', 'sex',
             'mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor', 'prom',
             'ster', 'sterp', 'sterd', 'atbyn', 'delm']

x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
             'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']

# 5% 미만 제외 (resuh, resue는 유지, sftup 제외) - 27개
y_columns = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'rds', 'sft', 'sftw',
             'als', 'mph', 'ph', 'bpdyn', 'bpdm', 'pdad', 'acl', 'lbp', 'ivh2', 'ivh3', 'pvl', 'seps', 
             'ntet', 'pmio', 'eythtran', 'deathyn','supyn']

y_display_names = {
    'resu': '초기 소생술 필요 유무',
    'resuo': '초기 소생술 산소',
    'resup': '초기 소생술 양압 환기',
    'resui': '초기 소생술 기도 삽관',
    'resuh': '초기 소생술 심장마사지',
    'resue': '초기 소생술 Epinephrine',
    'resuc': '초기 소생술 CPAP',
    'rds': '신생아 호흡곤란증후군',
    'sft': '폐표면활성제 사용',
    'sftw': '폐활성제 출생 즉시 사용',
    'als': '공기누출증후군',
    'mph': '대량 폐출혈',
    'ph': '폐동맥 고혈압',
    'bpdyn': '기관지폐이형성증 여부(≥mild BPD)',
    'bpdm': '중등증 기관지폐이형성증(≥moderate BPD)',
    'pdad': 'PDA 약물 치료',
    'acl': '동맥관 결찰술',
    'lbp': '저혈압',
    'ivh2': '뇌실내출혈 (Grade≥2)',
    'ivh3': '중증 뇌실내출혈 (Grade≥3)',
    'pvl': '백질연화증',
    'seps': '패혈증',
    'ntet': '괴사성 장염',
    'pmio': '망막증 수술',
    'eythtran': '적혈구 수혈',
    'deathyn': 'NICU 입원중 사망',
    'supyn': '퇴원시 보조 장비 필요'
}

# 그룹 구분
group_resuscitation = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'sft', 'sftw']
group_complication = ['rds', 'als', 'mph', 'ph', 'bpdyn', 'bpdm', 'pdad', 'acl', 'lbp',
                      'ivh2', 'ivh3', 'pvl', 'seps', 'ntet', 'pmio', 'eythtran', 'deathyn', 'supyn']

# 최적 모델만 로드 (파일명: best_<ModelName>_<Target>.pkl)
@st.cache_resource
def load_best_models():
    best_models = {}
    for y_col in y_columns:
        for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
            model_path = os.path.join(model_save_dir, f"best_{model_name}_{y_col}.pkl")
            if os.path.exists(model_path):
                try:
                    best_models[y_col] = joblib.load(model_path)
                    break  # 가장 먼저 발견된 모델 하나만 사용
                except Exception as e:
                    st.warning(f"모델 로드 실패: {model_path} ({e})")
            
    return best_models

all_models = load_best_models()

st.title("NICU 예후 예측 대시보드")

st.header("입력 정보")
gaw = st.number_input("임신 주수", 20, 50, 28)
gawd = st.number_input("임신 일수", 0, 6, 0)
gad = gaw * 7 + gawd
bwei = st.number_input("출생 체중 (g)", 200, 5000, 1000)
sex = st.selectbox("성별", [1, 2, 3], format_func=lambda x: {1: "남아", 2: "여아", 3: "ambiguous"}[x])

# 기타 입력 항목
inputs = {
    'mage': st.number_input("산모 나이", 15, 99, 30),
    'gran': st.number_input("임신력", 0, 10, 0),
    'parn': st.number_input("출산력", 0, 10, 0),
    'amni': st.selectbox("양수량", [1, 2, 3, 4], format_func=lambda x: ["정상", "과소", "과다", "모름"][x-1]),
    'mulg': st.selectbox("다태 정보", [1, 2, 3, 4], format_func=lambda x: ["Singleton", "Twin", "Triplet", "Quad 이상"][x-1]),
    'bir': st.selectbox("출생 순서", [0, 1, 2, 3, 4], format_func=lambda x: ["단태", "1st", "2nd", "3rd", "4th 이상"][x]),
    'prep': st.selectbox("임신 과정", [1, 2], format_func=lambda x: ["자연임신", "IVF"][x-1]),
    'dm': st.selectbox("당뇨", [1, 2, 3], format_func=lambda x: ["없음", "GDM", "Overt DM"][x-1]),
    'htn': st.selectbox("고혈압", [1, 2, 3], format_func=lambda x: ["없음", "PIH", "Chronic HTN"][x-1]),
    'chor': st.selectbox("융모양막염", [1, 2, 3], format_func=lambda x: ["없음", "있음", "모름"][x-1]),
    'prom': st.selectbox("조기 양막 파열", [1, 2, 3], format_func=lambda x: ["없음", "있음", "모름"][x-1]),
    'ster': st.selectbox("스테로이드 사용", [1, 2, 3], format_func=lambda x: ["없음", "있음", "모름"][x-1]),
    'sterp': st.selectbox("스테로이드 완료 여부", [0, 1, 2, 3], format_func=lambda x: ["미투여", "미완료", "완료", "모름"][x]),
    'sterd': st.selectbox("스테로이드 약제", [0, 1, 2, 3, 4], format_func=lambda x: ["미투여", "Dexa", "Beta", "Dexa+Beta", "모름"][x]),
    'atbyn': st.selectbox("항생제 사용", [1, 2], format_func=lambda x: ["없음", "있음"][x-1]),
    'delm': st.selectbox("분만 방식", [1, 2], format_func=lambda x: ["질식분만", "제왕절개"][x-1])
}

# 입력 정리
data_values = [inputs[col] for col in x_columns if col in inputs] + [gad, sex, bwei]
new_X_data = pd.DataFrame([data_values], columns=x_columns)

# 환자 식별자
patient_id = st.text_input("환자 식별자 (예: patient123)", max_chars=20)

# 예측 수행
if st.button("예측 실행"):
    results = []
    for y_col in y_columns:
        if y_col not in all_models:
            continue
        model = all_models[y_col]
        try:
            prob = model.predict_proba(new_X_data)[0, 1]
            results.append({
                'Target': y_col,
                'Name': y_display_names.get(y_col, y_col),
                'Probability (%)': f"{prob * 100:.2f}%"
            })
        except:
            continue

    df_results = pd.DataFrame(results)
    df_results.set_index('Target', inplace=True)

    st.subheader("* 신생아 소생술 관련 예측")
    st.dataframe(df_results.loc[group_resuscitation])

    st.subheader("* 미숙아 합병증 및 예후 예측")
    st.dataframe(df_results.loc[group_complication])

    # 결과 다운로드
    if patient_id:
        buf = io.StringIO()
        buf.write(f"환자 식별자: {patient_id}\n")
        buf.write(f"예측일: {datetime.today().strftime('%Y-%m-%d')}\n\n")
        buf.write("[입력 데이터]\n")
        for var, val in zip(display_columns, [gaw, gawd, gad, bwei, sex] + list(inputs.values())):
            buf.write(f"{var}: {val}\n")

        buf.write("\n[예측 결과]\n")
        buf.write(df_results[['Name', 'Probability (%)']].to_string(index=False))

        st.download_button(
            label="📄 결과 TXT 다운로드",
            data=buf.getvalue(),
            file_name=f"{patient_id}_result.txt",
            mime="text/plain"
        )
