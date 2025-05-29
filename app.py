import streamlit as st
import joblib
import pandas as pd
import os
import io
from datetime import datetime

model_save_dir = 'saved_models'

# 언어 선택
lang = st.sidebar.radio("언어 / Language", ['한국어', 'English'])

# 변수 목록
x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
             'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']

display_columns = ['gaw', 'gawd', 'gad', 'bwei', 'sex',
                   'mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor', 'prom',
                   'ster', 'sterp', 'sterd', 'atbyn', 'delm']

# 5% 미만 제외 - 27개
y_columns = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'rds', 'sft', 'sftw',
             'als', 'mph', 'ph', 'bpdyn', 'bpdm', 'pdad', 'acl', 'lbp', 'ivh2', 'ivh3', 'pvl', 'seps', 
             'ntet', 'pmio', 'eythtran', 'deathyn','supyn']

threshold_df = pd.read_csv('model_thresholds.csv')
thresh_map = threshold_df.set_index('target')['threshold'].to_dict()

# 한글 이름
y_display_ko = {
    'resu': '초기 소생술 필요 유무', 'resuo': '초기 소생술 산소', 'resup': '초기 소생술 양압 환기', 'resui': '초기 소생술 기도 삽관',
    'resuh': '초기 소생술 심장마사지', 'resue': '초기 소생술 Epinephrine', 'resuc': '초기 소생술 CPAP',
    'rds': '신생아 호흡곤란증후군', 'sft': '폐표면활성제 사용', 'sftw': '폐활성제 출생 즉시 사용',
    'als': '공기누출증후군', 'mph': '대량 폐출혈', 'ph': '폐동맥 고혈압', 'bpdyn': '기관지폐이형성증 여부(≥mild BPD)',
    'bpdm': '중등증 기관지폐이형성증(≥moderate BPD)', 'pdad': 'PDA 약물 치료', 'acl': '동맥관 결찰술', 'lbp': '저혈압',
    'ivh2': '뇌실내출혈 (Grade≥2)', 'ivh3': '중증 뇌실내출혈 (Grade≥3)', 'pvl': '백질연화증', 'seps': '패혈증',
    'ntet': '괴사성 장염', 'pmio': '망막증 수술', 'eythtran': '적혈구 수혈', 'deathyn': 'NICU 입원중 사망', 'supyn': '퇴원시 보조 장비 필요'
}

# 영어 이름
y_display_en = {
    'resu': 'Resuscitation needed', 'resuo': 'Oxygen', 'resup': 'PPV', 'resui': 'Intubation',
    'resuh': 'Chest compression', 'resue': 'Epinephrine', 'resuc': 'CPAP',
    'rds': 'RDS', 'sft': 'Surfactant use', 'sftw': 'Immediate surfactant',
    'als': 'Air leak', 'mph': 'Massive pulmonary hemorrhage', 'ph': 'Pulmonary hypertension',
    'bpdyn': '≥ Mild BPD', 'bpdm': '≥ Moderate BPD', 'pdad': 'PDA medication', 'acl': 'PDA ligation', 'lbp': 'Hypotension',
    'ivh2': 'IVH (≥Grade 2)', 'ivh3': 'IVH (≥Grade 3)', 'pvl': 'PVL', 'seps': 'Sepsis',
    'ntet': 'NEC', 'pmio': 'ROP surgery', 'eythtran': 'RBC transfusion', 'deathyn': 'In-hospital death', 'supyn': 'Discharge support'
}

y_display_names = y_display_ko if lang == '한국어' else y_display_en

# 그룹 구분
resuscitation_targets = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'sft', 'sftw']
complication_targets = [y for y in y_columns if y not in resuscitation_targets]

@st.cache_resource
def load_best_models():
    best_models = {}
    for y_col in y_columns:
        for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
            path = os.path.join(model_save_dir, f"best_{model_name}_{y_col}.pkl")
            if os.path.exists(path):
                try:
                    best_models[y_col] = joblib.load(path)
                    break
                except:
                    continue
    return best_models

models = load_best_models()

st.title("NICU Outcome Prediction Dashboard" if lang == 'English' else "NICU 예후 예측 대시보드")

st.header("Input Information" if lang == 'English' else "입력 정보")
gaw = st.number_input("Gestational Weeks" if lang == 'English' else "임신 주수", 20, 50, 28)
gawd = st.number_input("Gestational Days" if lang == 'English' else "임신 일수", 0, 6, 0)
gad = gaw * 7 + gawd
bwei = st.number_input("Birth Weight (g)" if lang == 'English' else "출생 체중 (g)", 200, 5000, 1000)
sex_labels = {1: ("Male" if lang == 'English' else "남아"), 2: ("Female" if lang == 'English' else "여아"), 3: "Ambiguous"}
sex = st.selectbox("Sex" if lang == 'English' else "성별", [1, 2, 3], format_func=lambda x: sex_labels[x])


# 기타 입력 항목 (언어별 라벨 지정)
inputs = {
    'mage': st.number_input("Maternal Age" if lang == 'English' else "산모 나이", 15, 99, 30),
    'gran': st.number_input("Gravidity" if lang == 'English' else "임신력", 0, 10, 0),
    'parn': st.number_input("Parity" if lang == 'English' else "출산력", 0, 10, 0),
    'amni': get_selectbox("양수량", "Amniotic Fluid", [1,2,3,4], ["정상", "과소", "과다", "모름"], ["Normal", "Oligo", "Poly", "Unknown"]),
    'mulg': get_selectbox("다태 정보", "Multiplicity", [1,2,3,4], ["Singleton", "Twin", "Triplet", "Quad 이상"], ["Singleton", "Twin", "Triplet", "Quad+"]),
    'bir': get_selectbox("출생 순서", "Birth Order", [0,1,2,3,4], ["단태", "1st", "2nd", "3rd", "4th 이상"], ["Single", "1st", "2nd", "3rd", "4th+"]),
    'prep': get_selectbox("임신 과정", "Pregnancy Type", [1,2], ["자연임신", "IVF"], ["Natural", "IVF"]),
    'dm': get_selectbox("당뇨", "Diabetes", [1,2,3], ["없음", "GDM", "Overt DM"], ["None", "GDM", "Overt"]),
    'htn': get_selectbox("고혈압", "Hypertension", [1,2,3], ["없음", "PIH", "Chronic HTN"], ["None", "PIH", "Chronic"]),
    'chor': get_selectbox("융모양막염", "Chorioamnionitis", [1,2,3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'prom': get_selectbox("조기 양막 파열", "PROM", [1,2,3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'ster': get_selectbox("스테로이드 사용", "Steroid Use", [1,2,3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'sterp': get_selectbox("스테로이드 완료 여부", "Steroid Completion", [0,1,2,3], ["미투여", "미완료", "완료", "모름"], ["None", "Incomplete", "Complete", "Unknown"]),
    'sterd': get_selectbox("스테로이드 약제", "Steroid Type", [0,1,2,3,4], ["미투여", "Dexa", "Beta", "Dexa+Beta", "모름"], ["None", "Dexa", "Beta", "Dexa+Beta", "Unknown"]),
    'atbyn': get_selectbox("항생제 사용", "Antibiotics", [1,2], ["없음", "있음"], ["No", "Yes"]),
    'delm': get_selectbox("분만 방식", "Delivery Mode", [1,2], ["질식분만", "제왕절개"], ["Vaginal", "Cesarean"])
}

# 입력값 정리
data_values = [inputs[col] for col in x_columns if col in inputs] + [gad, sex, bwei]
new_X_data = pd.DataFrame([data_values], columns=x_columns)

# 환자 식별자
patient_id = st.text_input("Patient ID (for download)" if lang == 'English' else "환자 식별자 (예: 등록번호)", max_chars=20)

# 예측 실행
if st.button("Run Prediction" if lang == 'English' else "예측 실행"):
    results = []
    for y_col in y_columns:
        if y_col in models:
            try:
                prob = models[y_col].predict_proba(new_X_data)[0, 1]
                thresh = thresh_map.get(y_col, 0.5)
                mark = "★" if prob >= thresh else ""
                results.append({
                    'Code': y_col,
                    'Name': y_display_names.get(y_col, y_col),
                    'Probability (%)': f"{prob * 100:.2f}%",
                    'Flag': mark
                })
            except:
                continue

    df = pd.DataFrame(results).set_index('Code')

    st.subheader("* Resuscitation Predictions" if lang == 'English' else "* 신생아 소생술 관련 예측")
    st.dataframe(df.loc[resuscitation_targets])

    st.subheader("* Complication Predictions" if lang == 'English' else "* 미숙아 합병증 및 예후 예측")
    st.dataframe(df.loc[complication_targets])

    if patient_id:
        buf = io.StringIO()
        buf.write(f"ID: {patient_id}\nDate: {datetime.today().strftime('%Y-%m-%d')}\n\n")
        buf.write("[Inputs]\n")
        for k, v in zip(display_columns, [gaw, gawd, gad, bwei, sex] + list(inputs.values())):
            buf.write(f"{k}: {v}\n")
        buf.write("\n[Predictions]\n")
        buf.write(df[['Name', 'Probability (%)', 'Flag']].to_string(index=False))

        st.download_button(
            label="📄 Download Results TXT" if lang == 'English' else "📄 결과 TXT 다운로드",
            data=buf.getvalue(),
            file_name=f"{patient_id}_result.txt",
            mime="text/plain"
        )
