import streamlit as st
import joblib
import pandas as pd
import os
import io
from datetime import datetime

model_save_dir = 'saved_models'

# 전체 화면 넓게 설정
st.set_page_config(layout="wide")

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
thresh_map = threshold_df.set_index(['target', 'model'])['threshold'].to_dict()

# 그룹 구분
resuscitation_targets = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'sft', 'sftw']
complication_targets = [y for y in y_columns if y not in resuscitation_targets]

# 입력 항목
def get_selectbox(label_kr, label_en, options, labels_kr, labels_en):
    label = label_en if lang == 'English' else label_kr
    labels = labels_en if lang == 'English' else labels_kr
    return st.selectbox(label, options, format_func=lambda x: labels[options.index(x)])

# GAW 입력
gaw = st.number_input("임신 주수" if lang == '한국어' else "Gestational Weeks", 20, 50, 28)
gawd = st.number_input("임신 일수" if lang == '한국어' else "Gestational Days", 0, 6, 0)
gad = gaw * 7 + gawd

# 입력값 받기
sex = get_selectbox("성별", "Sex", [1, 2, 3], ["남아", "여아", "미분류"], ["Male", "Female", "Ambiguous"])
bwei = st.number_input("출생 체중 (g)" if lang == '한국어' else "Birth Weight (g)", 200, 5000, 1000)

# 기타 입력값
inputs = {
    'mage': st.number_input("산모 나이" if lang == '한국어' else "Maternal Age", 15, 99, 30),
    'gran': st.number_input("임신력" if lang == '한국어' else "Gravidity", 0, 10, 0),
    'parn': st.number_input("출산력" if lang == '한국어' else "Parity", 0, 10, 0),
    'amni': get_selectbox("양수량", "Amniotic Fluid", [1, 2, 3, 4], ["정상", "과소", "과다", "모름"], ["Normal", "Oligo", "Poly", "Unknown"]),
    'mulg': get_selectbox("다태 정보", "Multiplicity", [1, 2, 3, 4], ["Singleton", "Twin", "Triplet", "Quad 이상"], ["Singleton", "Twin", "Triplet", "Quad+"]),
    'bir': get_selectbox("출생 순서", "Birth Order", [0, 1, 2, 3, 4], ["단태", "1st", "2nd", "3rd", "4th 이상"], ["Single", "1st", "2nd", "3rd", "4th+"]),
    'prep': get_selectbox("임신 과정", "Pregnancy Type", [1, 2], ["자연임신", "IVF"], ["Natural", "IVF"]),
    'dm': get_selectbox("당뇨", "Diabetes", [1, 2, 3], ["없음", "GDM", "Overt DM"], ["None", "GDM", "Overt"]),
    'htn': get_selectbox("고혈압", "Hypertension", [1, 2, 3], ["없음", "PIH", "Chronic HTN"], ["None", "PIH", "Chronic"]),
    'chor': get_selectbox("융모양막염", "Chorioamnionitis", [1, 2, 3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'prom': get_selectbox("조기 양막 파열", "PROM", [1, 2, 3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'ster': get_selectbox("스테로이드 사용", "Steroid Use", [1, 2, 3], ["없음", "있음", "모름"], ["No", "Yes", "Unknown"]),
    'sterp': get_selectbox("스테로이드 완료 여부", "Steroid Completion", [0, 1, 2, 3], ["미투여", "미완료", "완료", "모름"], ["None", "Incomplete", "Complete", "Unknown"]),
    'sterd': get_selectbox("스테로이드 약제", "Steroid Type", [0, 1, 2, 3, 4], ["미투여", "Dexa", "Beta", "Dexa+Beta", "모름"], ["None", "Dexa", "Beta", "Dexa+Beta", "Unknown"]),
    'atbyn': get_selectbox("항생제 사용", "Antibiotics", [1, 2], ["없음", "있음"], ["No", "Yes"]),
    'delm': get_selectbox("분만 방식", "Delivery Mode", [1, 2], ["질식분만", "제왕절개"], ["Vaginal", "Cesarean"]),
    'gad': gad,
    'sex': sex,
    'bwei': bwei
}

data_values = [inputs[col] for col in x_columns if col in inputs] + [gad, sex, bwei]
new_X_data = pd.DataFrame([[inputs[col] for col in x_columns]], columns=x_columns)

# 환자 식별자
patient_id = st.text_input("환자 식별자 (예: 등록번호)" if lang == '한국어' else "Patient ID (for download)", max_chars=20)

# 예측 실행 및 결과 시각화
if st.button("예측 실행" if lang == '한국어' else "Run Prediction"):
    @st.cache_resource
    def load_best_models():
        best_models = {}
        for y_col in y_columns:
            for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
                path = os.path.join(model_save_dir, f"best_{model_name}_{y_col}.pkl")
                if os.path.exists(path):
                    try:
                        best_models[(y_col, model_name)] = joblib.load(path)
                    except:
                        continue
        return best_models

    models = load_best_models()
    predictions = {}
    for y_col in y_columns:
        for model_name in ['LightGBM', 'XGBoost', 'RandomForest']:
            model_key = (y_col, model_name)
            if model_key in models:
                model = models[model_key]
                try:
                    prob = model.predict_proba(new_X_data)[0, 1]
                    threshold = thresh_map.get((y_col, model_name), 0.5)
                    mark = "★" if prob >= threshold else ""
                    predictions[y_col] = {'Probability (%)': f"{prob*100:.2f}%", 'Flag': mark}
                    break
                except:
                    continue

    resuscitation_targets = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'sft', 'sftw']
    complication_targets = [y for y in y_columns if y not in resuscitation_targets]

    resus_df = pd.DataFrame.from_dict({k: v for k, v in predictions.items() if k in resuscitation_targets}, orient='index')
    comp_df = pd.DataFrame.from_dict({k: v for k, v in predictions.items() if k in complication_targets}, orient='index')

    st.subheader("* 신생아 소생술 관련 예측" if lang == '한국어' else "* Resuscitation Predictions")
    st.dataframe(resus_df)

    st.subheader("* 미숙아 합병증 및 예후 예측" if lang == '한국어' else "* Complication Predictions")
    st.dataframe(comp_df)

    if patient_id:
        output = io.StringIO()
        output.write(f"Patient ID: {patient_id}\nDate: {datetime.today().strftime('%Y-%m-%d')}\n\n")
        output.write("[입력 정보 / Input Information]\n")
        for col in display_columns:
            output.write(f"{col}: {inputs.get(col, '')}\n")
        output.write("\n[예측 결과 / Prediction Results]\n")
        output.write("[Resuscitation Predictions]\n")
        output.write(resus_df.to_string())
        output.write("\n\n[Complication Predictions]\n")
        output.write(comp_df.to_string())

        st.download_button(
            label="결과 TXT 다운로드" if lang == '한국어' else "Download Results TXT",
            data=output.getvalue(),
            file_name=f"{patient_id}_result.txt",
            mime="text/plain"
        )
