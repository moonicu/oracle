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
    'bpdyn': '기관지폐이형성증 여부',
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
    'supyn': '퇴원시 보조 장비 필요 여부',
    'dcdhm1': '퇴원시 모니터링 필요',
    'dcdhm2': '퇴원시 산소 필요',
    'dcdhm3': '퇴원시 기관절개 유지',
    'dcdhm4': '퇴원시 장루 유지',
    'dcdhm5': '퇴원시 위관영양 필요',
    'dcdhm6': '퇴원시 기타 보조 필요',
    'dcdhm7': '퇴원시 인공호흡기 필요'
}


st.title("NICU 환자 예측 모델")

st.header("입력 데이터")

gaw = st.number_input("임신 주수", min_value=20, max_value=50, value=28)
gawd = st.number_input("임신 일수", min_value=0, max_value=6, value=4)
gad = gaw * 7 + gawd
bwei = st.number_input("출생 체중 (g)", min_value=200, max_value=5000, value=1000)
sex = st.selectbox("성별", [1, 2, 3], format_func=lambda x: {1: "남아", 2: "여아", 3: "ambiguous"}.get(x))

mage = st.number_input("산모 나이", min_value=15, max_value=99, value=30)
gran = st.number_input("임신력 (gravida)", min_value=0, max_value=10, value=0)
parn = st.number_input("출산력 (parity)", min_value=0, max_value=10, value=0)

amni = st.selectbox("양수량", [1, 2, 3, 4], format_func=lambda x: {1: "정상", 2: "과소증", 3: "과다증", 4: "모름"}.get(x))
mulg = st.selectbox("다태 정보", [1, 2, 3, 4], format_func=lambda x: {1: "Singleton", 2: "Twin", 3: "Triplet", 4: "Quad 이상"}.get(x))
bir = st.selectbox("출생 순서", [0, 1, 2, 3, 4], format_func=lambda x: {0: "단태", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th 이상"}.get(x))

prep = st.selectbox("임신과정", [1, 2], format_func=lambda x: {1: "자연임신", 2: "IVF"}.get(x))
dm = st.selectbox("당뇨", [1, 2, 3], format_func=lambda x: {1: "없음", 2: "GDM", 3: "Overt DM"}.get(x))
htn = st.selectbox("고혈압", [1, 2, 3], format_func=lambda x: {1: "없음", 2: "PIH", 3: "Chronic HTN"}.get(x))
chor = st.selectbox("융모양막염", [1, 2, 3], format_func=lambda x: {1: "없음", 2: "있음", 3: "모름"}.get(x))
prom = st.selectbox("조기 양막 파열", [1, 2, 3], format_func=lambda x: {1: "없음", 2: "있음", 3: "모름"}.get(x))
ster = st.selectbox("산전스테로이드 투여 여부", [1, 2, 3], format_func=lambda x: {1: "없음", 2: "있음", 3: "모름"}.get(x))
sterp = st.selectbox("스테로이드 완료 여부 (sterp)", [0, 1, 2, 3],
    format_func=lambda x: ["미투여", "투여했으나 미완료", "완료", "확인 불가"][x],
    help="분만 1주일 이내에 정해진 간격으로 정해진 코스의 스테로이드 치료를 모두 완료한 경우 완료(betamethasone 2회, dexamethasone 4회)"
)
sterd = st.selectbox("스테로이드 약제", [0, 1, 2, 4], format_func=lambda x: {0: "미투여", 1: "Dexamethasone", 2: "Betamethasone", 3: "Dexa+Beta", 4: "모름"}.get(x))
atbyn = st.selectbox("항생제 사용", [1, 2], format_func=lambda x: {1: "없음", 2: "있음"}.get(x))
delm = st.selectbox("분만 방식 (delm)", [1, 2], format_func=lambda x: {1: "질식분만", 2: "제왕절개"}.get(x))

new_X_data = pd.DataFrame([[mage, gran, parn, amni, mulg, bir, prep, dm, htn, chor,
                            prom, ster, sterp, sterd, atbyn, delm, gad, sex, bwei]], columns=x_columns)

# 환자 식별자 입력
patient_id = st.text_input("환자정보 (최대 10자), 추출시 파일명", max_chars=10)

if st.button("결과 예측"):
    result_rows = []

    for model_name in model_names:
        for y_col in y_columns:
            model_filename = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")
            if not os.path.exists(model_filename):
                st.warning(f"❗ 모델 파일 없음: {model_filename}")
                result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': None})
                continue

            try:
                model = joblib.load(model_filename)

                if hasattr(model, "predict_proba"):
                    if model_name == "XGBoost" and hasattr(model, 'get_booster'):
                        model_features = model.get_booster().feature_names
                        X_input = new_X_data[model_features]
                    else:
                        X_input = new_X_data

                    pred_proba = model.predict_proba(X_input)
                    pred_percent = round(float(pred_proba[0, 1]) * 100, 2)
                    result_rows.append({
                        'Target': y_col, 'Model': model_name, 'Probability (%)': f"{pred_percent:.2f}%"
                    })
                else:
                    result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': None})

            except Exception as e:
                st.warning(f"[{model_name} - {y_col}] 예측 실패: {e}")
                result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': None})

    # 결과 정리
    df_result = pd.DataFrame(result_rows)
    pivot_result = df_result.pivot(index='Target', columns='Model', values='Probability (%)')
    pivot_result = pivot_result[model_names]
    pivot_result = pivot_result.reindex(y_columns)
    pivot_result.index = pivot_result.index.map(lambda x: y_display_names.get(x, x))

    # Streamlit 화면에 결과 표시
    st.dataframe(pivot_result, height=900)

    # CSV 저장
    if patient_id:
        csv_buffer = io.StringIO()

        # 입력값
        input_values = [gaw, gawd, gad, bwei, sex, mage, gran, parn, amni, mulg, bir,
                        prep, dm, htn, chor, prom, ster, sterp, sterd, atbyn, delm]
        input_df = pd.DataFrame({'features': display_columns, 'input_values': input_values})

        # CSV 내용 작성
        csv_buffer.write("[KNN data]\n")
        input_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.write("\n[Prediction Results]\n")
        pivot_result.to_csv(csv_buffer, encoding='utf-8-sig')

        st.download_button(
            label="📥 CSV 다운로드 (입력값 + 예측결과)",
            data=csv_buffer.getvalue(),
            file_name=f"{patient_id}.csv",
            mime='text/csv'
        )
    else:
        st.info("⬅ 환자정보를 입력하면 결과를 CSV로 다운로드할 수 있습니다.")

