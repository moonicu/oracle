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
             'als', 'mph', 'ph', 'bpdyn', 'bdp', 'pdad', 'acl', 'lbp', 'inhg', 'phh', 'pvl', 'ibif', 'seps', 'meni',
             'invfpod', 'ntet', 'ntety', 'iperr', 'pmio', 'avegftr', 'eythtran', 'stday', 'dcd', 'deathyn',
             'supyn', 'dcdhm1', 'dcdhm2', 'dcdhm3', 'dcdhm4', 'dcdhm5', 'dcdhm6', 'dcdhm7', 'dcdwt']

y_display_names = {
    'resu': '초기 소생술 필요 유무', 'resuo': '초기 소생술 산소', 'resup': '초기 소생술 양압 환기',
    'resui': '초기 소생술 기도 삽관', 'resuh': '초기 소생술 심장 마사지', 'resue': '초기 소생술 Epinephrine',
    'resuc': '초기 소생술 CPAP', 'rds': '신생아 호흡곤란증후군', 'sft': '폐표면활성제 사용',
    'sftup': '폐표면활성제 사용 목적', 'sftw': '폐표면활성제 장소', 'als': '공기누출증후군', 'mph': '대량 폐출혈',
    'ph': '폐동맥고혈압', 'bpdyn': '기관지폐이형성증 여부', 'bdp': '기관지폐이형성증', 'pdad': 'PDA 약물 치료',
    'acl': '동맥관 결찰술', 'lbp': '저혈압', 'inhg': '뇌실내출혈', 'phh': '수두증', 'pvl': '백질연화증',
    'ibif': '선천성감염', 'seps': '패혈증', 'meni': '뇌수막염', 'invfpod': '정맥영양기간', 'ntet': '괴사성 장염',
    'ntety': '괴사 장염 수술', 'iperr': '특발성 장천공', 'pmio': '망막증 수술', 'avegftr': 'Anti-VEGF 치료',
    'eythtran': '적혈구 수혈', 'stday': '재원일수', 'dcd': '퇴원 형태', 'deathyn': '사망 여부',
    'supyn': '보조필요', 'dcdhm1': '보조장치-모니터링', 'dcdhm2': '보조장치-산소', 'dcdhm3': '기관절개',
    'dcdhm4': '장루술', 'dcdhm5': '위관영양', 'dcdhm6': '기타보조', 'dcdhm7': '인공호흡기', 'dcdwt': '퇴원체중'
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
    format_func=lambda x: ["미투여", "미완료", "완료", "확인 불가"][x],
    help="분만 1주일 이내에 정해진 간격으로 정해진 코스의 스테로이드 치료를 모두 완료한 경우 완료(betamethasone 2회, dexamethasone 4회)"
)

sterd = st.selectbox("스테로이드 약제", [0, 1, 2, 4], format_func=lambda x: {0: "미투여", 1: "Dexamethasone", 2: "Betamethasone", 4: "모름"}.get(x))
atbyn = st.selectbox("항생제 사용", [1, 2], format_func=lambda x: {1: "없음", 2: "있음"}.get(x))
delm = st.selectbox("분만 방식 (delm)", [1, 2], format_func=lambda x: {1: "질식분만", 2: "제왕절개"}.get(x))

new_X_data = pd.DataFrame([[mage, gran, parn, amni, mulg, bir, prep, dm, htn, chor, 
                            prom, ster, sterp, sterd, atbyn, delm, gad, sex, bwei]], columns=x_columns)

if st.button("결과 예측"):
    result_rows = []
    for model_name in model_names:
        for y_col in y_columns:
            model_filename = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")
            try:
                model = joblib.load(model_filename)
                if hasattr(model, "predict_proba"):
                    if model_name == "XGBoost" and hasattr(model, 'get_booster'):
                        model_features = model.get_booster().feature_names
                        X_input = new_X_data[model_features]
                    else:
                        X_input = new_X_data

                    pred_proba = model.predict_proba(X_input)
                    pred_percent = round(pred_proba[:, 1] * 100, 2)
                    result_rows.append({'Target': y_col, 'Model': model_name, 'Probability': pred_percent})
            except Exception:
                result_rows.append({'Target': y_col, 'Model': model_name, 'Probability': None})

    df_result = pd.DataFrame(result_rows)
    pivot_result = df_result.pivot(index='Target', columns='Model', values='Probability')
    pivot_result = pivot_result[model_names]

    # re-order by y_display_names
    pivot_result = pivot_result.reindex(y_columns)

    # Highlight best model per target
    highlight_df = pivot_result.copy()
    for idx in highlight_df.index:
        row = highlight_df.loc[idx]
        if row.notnull().any():
            max_idx = row.idxmax()
            highlight_df.at[idx, max_idx] = f"⭐ {row[max_idx]:.2f}%"

    # format percentages
    highlight_df = highlight_df.applymap(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)

    # apply display names
    highlight_df.index = highlight_df.index.map(lambda x: y_display_names.get(x, x))

    st.dataframe(highlight_df, height=900)

    # Excel Export
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        meta_info = pd.DataFrame({'항목': ['작성일자'], '값': [datetime.today().strftime('%Y-%m-%d')]})
        meta_info.to_excel(writer, sheet_name='입력 데이터', startrow=0, index=False)

        display_data = [gaw, gawd] + new_X_data.iloc[0].tolist()
        input_df = pd.DataFrame({'입력 변수명': display_columns, '입력값': display_data})
        input_df.to_excel(writer, sheet_name='입력 데이터', startrow=3, index=False)

        highlight_df.to_excel(writer, sheet_name='예측 결과')
        
        processed_data = output.getvalue()

    st.download_button(
        label="입력값 + 예측결과 엑셀로 다운로드",
        data=processed_data,
        file_name='predictions_with_input.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
