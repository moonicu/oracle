import streamlit as st
import joblib
import pandas as pd
import os
import io
from datetime import datetime

model_save_dir = 'saved_models'
model_names = ['RandomForest', 'XGBoost', 'LightGBM']

# 변수 목록 (순서 포함)
x_columns = ['gaw', 'gawd', 'gad', 'bwei', 'sex',
             'mage', 'gran', 'parn', 'amni', 'mulg', 'bir', 'prep', 'dm', 'htn', 'chor', 'prom',
             'ster', 'sterp', 'sterd', 'atbyn', 'delm']

y_columns = ['resu', 'resuo', 'resup', 'resui', 'resuh', 'resue', 'resuc', 'rds', 'sft', 'sftup', 'sftw',
             'als', 'mph', 'ph', 'bpdyn', 'bdp', 'pdad', 'acl', 'lbp', 'inhg', 'phh', 'pvl', 'ibif', 'seps', 'meni',
             'invfpod', 'ntet', 'ntety', 'iperr', 'pmio', 'avegftr', 'eythtran', 'stday', 'dcd', 'deathyn',
             'supyn', 'dcdhm1', 'dcdhm2', 'dcdhm3', 'dcdhm4', 'dcdhm5', 'dcdhm6', 'dcdhm7', 'dcdwt']

# y컬럼 출력용 한글명은 이전 그대로 사용 (y_display_names)

st.title("NICU 환자 예측 모델")

# 입력 폼
st.header("입력 데이터")

gaw = st.number_input("임신 주수 (gaw, 주)", min_value=20, max_value=50, value=28)
gawd = st.number_input("임신 일수 (gawd, 일)", min_value=0, max_value=6, value=4)
gad = (gaw * 7) + gawd  # 자동 계산
st.write(f"계산된 GAD (일 기준): {gad} 일")

bwei = st.number_input("출생 체중 (bwei, g)", min_value=200, max_value=1500, value=1000)
sex = st.selectbox("성별 (sex)", [0, 1], format_func=lambda x: "남아" if x == 0 else "여아")

mage = st.number_input("산모 나이 (mage)", min_value=18, max_value=50, value=30)
gran = st.number_input("임신력 (gran)", min_value=0, max_value=10, value=1)
parn = st.number_input("출산력 (parn)", min_value=0, max_value=10, value=1)
amni = st.selectbox("양수 상태 (amni)", [0, 1], format_func=lambda x: "정상" if x == 0 else "비정상")
mulg = st.selectbox("다태아 여부 (mulg)", [1, 2], format_func=lambda x: "단태" if x == 1 else "다태")
bir = st.number_input("출생 순서 (bir)", min_value=1, max_value=10, value=1)
prep = st.selectbox("분만 준비 상태 (prep)", [0, 1], format_func=lambda x: "미비" if x == 0 else "준비됨")
dm = st.selectbox("당뇨병 (dm)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
htn = st.selectbox("고혈압 (htn)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
chor = st.selectbox("조직학적 융모양막염 (chor)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
prom = st.selectbox("조기 양막 파열 (prom)", [0, 1], format_func=lambda x: "없음" if x == 0 else "있음")
ster = st.selectbox("스테로이드 사용 여부 (ster)", [0, 1], format_func=lambda x: "사용 안함" if x == 0 else "사용")
sterp = st.selectbox("스테로이드 완료 여부 (sterp)", [0, 1], format_func=lambda x: "미완료" if x == 0 else "완료")
sterd = st.selectbox("스테로이드 투여 후 7일 경과 여부 (sterd)", [0, 1], format_func=lambda x: "미경과" if x == 0 else "7일 경과")
atbyn = st.selectbox("항생제 사용 여부 (atbyn)", [0, 1], format_func=lambda x: "사용 안함" if x == 0 else "사용")
delm = st.selectbox("분만 방식 (delm)", [0, 1], format_func=lambda x: "자연" if x == 0 else "제왕절개")

# 입력 데이터를 순서에 맞게 구성
new_X_data = pd.DataFrame([[gaw, gawd, gad, bwei, sex, mage, gran, parn, amni, mulg, bir,
                            prep, dm, htn, chor, prom, ster, sterp, sterd, atbyn, delm]], columns=x_columns)

# 예측
st.header("예측 결과")
if st.button("결과 예측"):
    result_table = pd.DataFrame(index=y_columns, columns=model_names)

    for model_name in model_names:
        for y_col in y_columns:
            model_filename = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")
            try:
                model = joblib.load(model_filename)
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(new_X_data)
                    pred_percent = pred_proba[:, 1] * 100
                    result_table.at[y_col, model_name] = f"{pred_percent[0]:.2f}%"
                else:
                    result_table.at[y_col, model_name] = "N/A"
            except FileNotFoundError:
                st.error(f"모델 파일 '{model_filename}'을 찾을 수 없습니다.")
                result_table.at[y_col, model_name] = "N/A"

    result_table.index = result_table.index.map(lambda x: y_display_names.get(x, x))
    st.dataframe(result_table, height=700, width=900)

    # Excel Export + 작성일자 추가
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        input_df = pd.DataFrame(new_X_data.T)
        input_df.columns = ['입력값']
        input_df['입력 변수명'] = input_df.index
        input_df = input_df[['입력 변수명', '입력값']].reset_index(drop=True)

        # 작성일자
        today = datetime.today().strftime('%Y-%m-%d')
        meta_info = pd.DataFrame({'항목': ['작성일자'], '값': [today]})

        meta_info.to_excel(writer, sheet_name='입력 데이터', startrow=0, index=False)
        input_df.to_excel(writer, sheet_name='입력 데이터', startrow=3, index=False)

        result_table.to_excel(writer, sheet_name='예측 결과')
        writer.save()
        processed_data = output.getvalue()

    st.download_button(
        label="입력값 + 예측결과 엑셀로 다운로드",
        data=processed_data,
        file_name='predictions_with_input.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
