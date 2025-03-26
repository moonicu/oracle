import streamlit as st
import joblib
import pandas as pd
import os
import io

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

# ▼ 입력
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
sterd = st.selectbox("스테로이드 약제", [0, 1, 2, 3, 4], format_func=lambda x: {0: "미투여", 1: "Dexamethasone", 2: "Betamethasone", 3: "Dexa+Beta", 4: "모름"}.get(x))
atbyn = st.selectbox("항생제 사용", [1, 2], format_func=lambda x: {1: "없음", 2: "있음"}.get(x))
delm = st.selectbox("분만 방식 (delm)", [1, 2], format_func=lambda x: {1: "질식분만", 2: "제왕절개"}.get(x))

# ▼ 예측용 데이터 생성
new_X_data = pd.DataFrame([[mage, gran, parn, amni, mulg, bir, prep, dm, htn, chor,
                            prom, ster, sterp, sterd, atbyn, delm, gad, sex, bwei]], columns=x_columns)

# ▼ 전역 상태: 예측 결과 보관
if 'pivot_result' not in st.session_state:
    st.session_state.pivot_result = pd.DataFrame()

# ▼ 예측 수행
if st.button("결과 예측"):
    result_rows = []
    for model_name in model_names:
        for y_col in y_columns:
            model_path = os.path.join(model_save_dir, f"{model_name}_{y_col}.pkl")
            if not os.path.exists(model_path):
                result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': None})
                continue
            model = joblib.load(model_path)
            X_input = new_X_data
            if model_name == "XGBoost" and hasattr(model, 'get_booster'):
                model_features = model.get_booster().feature_names
                X_input = new_X_data[model_features]
            pred_proba = model.predict_proba(X_input)
            prob = round(pred_proba[0, 1] * 100, 2)
            result_rows.append({'Target': y_col, 'Model': model_name, 'Probability (%)': f"{prob:.2f}%"})

    df_result = pd.DataFrame(result_rows)
    pivot = df_result.pivot(index='Target', columns='Model', values='Probability (%)')
    pivot = pivot[model_names].reindex(y_columns)
    pivot.index = pivot.index.map(lambda x: y_display_names.get(x, x))
    st.dataframe(pivot, height=900)
    st.session_state.pivot_result = pivot  # 저장

# ▼ 환자 ID
patient_id = st.text_input("환자정보 (파일명)", max_chars=10)

# ▶ 입력값 수집
input_values = [gaw, gawd, gad, bwei, sex, mage, gran, parn, amni, mulg, bir,
                prep, dm, htn, chor, prom, ster, sterp, sterd, atbyn, delm]

# ▶ 입력값과 변수 정보 매핑
input_variable_info = {
    'gaw': ('임신 주수', lambda x: f"{x}주"),
    'gawd': ('임신 일수', lambda x: f"{x}일"),
    'gad': ('재태연령 (일)', lambda x: f"{x}일"),
    'bwei': ('출생 체중 (g)', str),
    'sex': ('성별', lambda x: {1: '남아', 2: '여아', 3: 'ambiguous'}.get(x, '')),
    'mage': ('산모 나이', str),
    'gran': ('임신력', str),
    'parn': ('출산력', str),
    'amni': ('양수량', lambda x: {1: '정상', 2: '과소증', 3: '과다증', 4: '모름'}.get(x, '')),
    'mulg': ('다태 정보', lambda x: {1: 'Singleton', 2: 'Twin', 3: 'Triplet', 4: 'Quad 이상'}.get(x, '')),
    'bir': ('출생 순서', lambda x: {0: '단태', 1: '1st', 2: '2nd', 3: '3rd', 4: '4th 이상'}.get(x, '')),
    'prep': ('임신 과정', lambda x: {1: '자연임신', 2: 'IVF'}.get(x, '')),
    'dm': ('당뇨', lambda x: {1: '없음', 2: 'GDM', 3: 'Overt DM'}.get(x, '')),
    'htn': ('고혈압', lambda x: {1: '없음', 2: 'PIH', 3: 'Chronic HTN'}.get(x, '')),
    'chor': ('융모양막염', lambda x: {1: '없음', 2: '있음', 3: '모름'}.get(x, '')),
    'prom': ('조기 양막 파열', lambda x: {1: '없음', 2: '있음', 3: '모름'}.get(x, '')),
    'ster': ('산전스테로이드 투여 여부', lambda x: {1: '없음', 2: '있음', 3: '모름'}.get(x, '')),
    'sterp': ('스테로이드 완료 여부', lambda x: {0: '미투여', 1: '미완료', 2: '완료', 3: '확인 불가'}.get(x, '')),
    'sterd': ('스테로이드 약제', lambda x: {0: '미투여', 1: 'Dexamethasone', 2: 'Betamethasone', 3: 'Dexa+Beta', 4: '모름'}.get(x, '')),
    'atbyn': ('항생제 사용', lambda x: {1: '없음', 2: '있음'}.get(x, '')),
    'delm': ('분만 방식', lambda x: {1: '질식분만', 2: '제왕절개'}.get(x, ''))
}

# ▶ 입력값 정리 (한글 설명 포함)
input_data_rows = []
for var, value in zip(display_columns, input_values):
    name, decode = input_variable_info.get(var, (var, str))
    input_data_rows.append({'변수 코드': var, '변수명(한글)': name, '입력값': value, '값 설명': decode(value)})
input_df = pd.DataFrame(input_data_rows)

# ▼ TXT 구성
txt_buffer = io.StringIO()
txt_buffer.write("📌 [입력 데이터]\n")
txt_buffer.write(input_df.to_string(index=False))
txt_buffer.write("\n\n📌 [예측 결과]\n")
if not st.session_state.pivot_result.empty:
    txt_buffer.write(st.session_state.pivot_result.reset_index().to_string(index=False))
else:
    txt_buffer.write("예측 결과가 없습니다. '결과 예측' 버튼을 먼저 눌러주세요.\n")

# ▼ 다운로드 버튼 항상 표시
if patient_id:
    st.download_button("📥 TXT 다운로드", data=txt_buffer.getvalue(), file_name=f"{patient_id}.txt", mime='text/plain')
else:
    st.warning("⬅ 환자정보를 입력해야 다운로드가 가능합니다.")
