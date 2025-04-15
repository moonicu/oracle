import streamlit as st
import joblib
import pandas as pd
import os
import io
from datetime import datetime

model_save_dir = 'saved_models'
model_names = ['RandomForest', 'XGBoost', 'LightGBM']

# 변수 목록
display_columns = ['gaw', 'gawd'] + ['gad', 'bwei', 'sex',
             'mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor', 'prom',
             'ster', 'sterp', 'sterd', 'atbyn', 'delm']

x_columns = ['mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor',
             'prom', 'ster', 'sterp', 'sterd', 'atbyn', 'delm', 'gad', 'sex', 'bwei']

y_columns = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'rds', 'sft', 'sftup', 'sftw',
             'als', 'mph', 'ph', 'bpdyn', 'bpdm', 'pdad', 'acl', 'lbp', 'ivh2', 'ivh3', 'phh', 'pvl', 'ibif', 'seps', 'meni',
             'ntet', 'ntety', 'iperr', 'pmio', 'avegftr', 'eythtran', 'deathyn',
             'supyn', 'dcdhm1', 'dcdhm2', 'dcdhm3', 'dcdhm4', 'dcdhm5', 'dcdhm6', 'dcdhm7']

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
    'sftup': '폐활성제 예방적 사용',
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
    'phh': '수두증',
    'pvl': '백질연화증',
    'ibif': '선천성 감염',
    'seps': '패혈증',
    'meni': '뇌수막염',
    'ntet': '괴사성 장염',
    'ntety': '괴사 장염 수술',
    'iperr': '특발성 장천공',
    'pmio': '망막증 수술',
    'avegftr': 'Anti-VEGF 치료',
    'eythtran': '적혈구 수혈',
    'deathyn': 'NICU 입원중 사망',
    'supyn': '퇴원시 보조 장비 필요',
    'dcdhm1': '퇴원시 모니터링 필요',
    'dcdhm2': '퇴원시 산소 필요',
    'dcdhm3': '퇴원시 기관절개 유지',
    'dcdhm4': '퇴원시 장루 유지',
    'dcdhm5': '퇴원시 위관영양 필요',
    'dcdhm6': '퇴원시 기타 보조 필요',
    'dcdhm7': '퇴원시 인공호흡기 필요'
}


# 모델 로딩 - 캐시 사용
@st.cache_resource
def load_all_models():
    all_models = {}
    for model_name in model_names:
        for y_col in y_columns:
            model_path = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")
            if os.path.exists(model_path):
                try:
                    all_models[(model_name, y_col)] = joblib.load(model_path)
                except Exception as e:
                    st.warning(f"[{model_name}-{y_col}] 모델 로드 실패: {e}")
            else:
                st.warning(f"❗ 모델 파일 없음: {model_path}")
    return all_models

all_models = load_all_models()

# UI 입력 폼
st.title("NICU 환자 예측 모델")
st.header("입력 데이터")

gaw = st.number_input("임신 주수", 20, 50, 28)
gawd = st.number_input("임신 일수", 0, 6, 4)
gad = gaw * 7 + gawd
bwei = st.number_input("출생 체중 (g)", 200, 5000, 1000)
sex = st.selectbox("성별", [1, 2, 3], format_func=lambda x: {1: "남아", 2: "여아", 3: "ambiguous"}[x])

# 기타 입력 항목 (mage ~ delm까지)
inputs = {
    'mage': st.number_input("산모 나이", 15, 99, 30),
    'gran': st.number_input("임신력", 0, 10, 0),
    'parn': st.number_input("출산력", 0, 10, 0),
    'amni': st.selectbox("양수량", [1,2,3,4], format_func=lambda x: ["정상","과소","과다","모름"][x-1]),
    'mulg': st.selectbox("다태 정보", [1,2,3,4], format_func=lambda x: ["Singleton","Twin","Triplet","Quad 이상"][x-1]),
    'bir': st.selectbox("출생 순서", [0,1,2,3,4], format_func=lambda x: ["단태","1st","2nd","3rd","4th 이상"][x]),
    'prep': st.selectbox("임신 과정", [1,2], format_func=lambda x: ["자연임신","IVF"][x-1]),
    'dm': st.selectbox("당뇨", [1,2,3], format_func=lambda x: ["없음","GDM","Overt DM"][x-1]),
    'htn': st.selectbox("고혈압", [1,2,3], format_func=lambda x: ["없음","PIH","Chronic HTN"][x-1]),
    'chor': st.selectbox("융모양막염", [1,2,3], format_func=lambda x: ["없음","있음","모름"][x-1]),
    'prom': st.selectbox("조기 양막 파열", [1,2,3], format_func=lambda x: ["없음","있음","모름"][x-1]),
    'ster': st.selectbox("스테로이드 사용", [1,2,3], format_func=lambda x: ["없음","있음","모름"][x-1]),
    'sterp': st.selectbox("스테로이드 완료 여부", [0,1,2,3], format_func=lambda x: ["미투여","미완료","완료","모름"][x]),
    'sterd': st.selectbox("스테로이드 약제", [0,1,2,3,4], format_func=lambda x: ["미투여","Dexa","Beta","Dexa+Beta","모름"][x]),
    'atbyn': st.selectbox("항생제 사용", [1,2], format_func=lambda x: ["없음","있음"][x-1]),
    'delm': st.selectbox("분만 방식", [1,2], format_func=lambda x: ["질식분만","제왕절개"][x-1])
}

# 입력 정리
data_values = [inputs[col] for col in x_columns if col in inputs] + [gad, sex, bwei]
new_X_data = pd.DataFrame([data_values], columns=x_columns)

# 예측 수행
if st.button("결과 예측"):
    result_rows = []
    for (model_name, y_col), model in all_models.items():
        try:
            if hasattr(model, "predict_proba"):
                if model_name == "XGBoost" and hasattr(model, 'get_booster'):
                    X_input = new_X_data[model.get_booster().feature_names]
                else:
                    X_input = new_X_data

                prob = model.predict_proba(X_input)
                result_rows.append({
                    'Target': y_col,
                    'Model': model_name,
                    'Probability (%)': f"{prob[0, 1]*100:.2f}%"
                })
        except Exception as e:
            result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': None})

    # 피벗 정리
    df_result = pd.DataFrame(result_rows)
    pivot = df_result.pivot(index='Target', columns='Model', values='Probability (%)')
    pivot = pivot[model_names].reindex(y_columns)
    pivot.index = pivot.index.map(lambda x: y_display_names.get(x, x))
    st.dataframe(pivot, height=800)

    # 다운로드 준비
    patient_id = st.text_input("환자 식별자 (파일명)", max_chars=10)
    if patient_id:
        txt_buf = io.StringIO()
        txt_buf.write("[입력 데이터]\n")
        for var, val in zip(display_columns, [gaw, gawd, gad, bwei, sex] + list(inputs.values())):
            txt_buf.write(f"{var}: {val}\n")

        txt_buf.write("\n[예측 결과]\n")
        txt_buf.write(pivot.reset_index().to_string(index=False))

        st.download_button(
            label="📄 결과 TXT 다운로드",
            data=txt_buf.getvalue(),
            file_name=f"{patient_id}.txt",
            mime="text/plain"
        )
