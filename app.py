import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# ========== تحميل النموذج والـ Scaler ==========
model = joblib.load('football_match_predictor.pkl')
scaler = joblib.load('scaler.pkl')

features = [
    'home_team_fifa_rank', 'away_team_fifa_rank',
    'home_team_total_fifa_points', 'away_team_total_fifa_points',
    'home_team_score', 'away_team_score',
    'home_team_goalkeeper_score', 'away_team_goalkeeper_score',
    'home_team_mean_defense_score', 'away_team_mean_defense_score',
    'home_team_mean_offense_score', 'away_team_mean_offense_score',
    'home_team_mean_midfield_score', 'away_team_mean_midfield_score',
    'year'
]

labels = {0: 'Lose', 1: 'Draw', 2: 'Win'}

# ========== خلفية مخصصة ==========
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# set_background("stadium_bg.png")  # الصورة لازم تكون بنفس الاسم في نفس الفولدر

# ========== عنوان التطبيق ==========
st.markdown("<h1 style='text-align: center; color: white;'>⚽ Football Match Result Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# ========== اختيار طريقة التنبؤ ==========
option = st.radio("اختر طريقة التنبؤ:", ['📥 إدخال يدوي', '📤 رفع ملف CSV'])

# ========== إدخال يدوي ==========
if option == '📥 إدخال يدوي':
    st.subheader("📝 أدخل بيانات المباراة:")
    input_data = []
    for feat in features:
        val = st.number_input(f"{feat.replace('_', ' ').capitalize()}", step=1.0)
        input_data.append(val)

    if st.button("🔮 توقع النتيجة"):
        X_scaled = scaler.transform([input_data])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        # نتيجة مفسّرة
        if pred == 2:
            result_text = "Expected Winner: 🏠 Home Team"
        elif pred == 0:
            result_text = "Expected Winner: 🛫 Away Team"
        else:
            result_text = "Expected Result: 🤝 Draw"

        # عرض النتيجة بشكل مميز
        st.markdown(
            f"<h2 style='text-align: center; color: white; font-weight: bold;'>{result_text}</h2>",
            unsafe_allow_html=True
        )

        # عرض الاحتمالات
        st.markdown("### 📊 الاحتمالات:")
        for i, label in labels.items():
            st.write(f"- {label}: {prob[i]*100:.2f}%")

# ========== رفع ملف CSV ==========
else:
    st.subheader("📂 ارفع ملف CSV يحتوي على بيانات المباريات")
    uploaded = st.file_uploader("اختر ملف CSV", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)

        if not all(col in df.columns for col in features):
            st.error("❌ الملف لا يحتوي على الأعمدة المطلوبة!")
        else:
            df_clean = df[features].fillna(df[features].mean())
            df_scaled = scaler.transform(df_clean)
            preds = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)

            df['Expected Result'] = [labels[p] for p in preds]
            df['Probability_Lose'] = probs[:, 0]
            df['Probability_Draw'] = probs[:, 1]
            df['Probability_Win'] = probs[:, 2]

            # تفسير النتيجة
            def interpret_result(row):
                if row['Expected Result'] == 'Win':
                    return "🏠 Home Team"
                elif row['Expected Result'] == 'Lose':
                    return "🛫 Away Team"
                else:
                    return "🤝 Draw"

            df['Winner Interpretation'] = df.apply(interpret_result, axis=1)

            st.success("✅ تم تنفيذ التوقعات بنجاح!")
            st.write(df[['Expected Result', 'Winner Interpretation', 'Probability_Lose', 'Probability_Draw', 'Probability_Win']])

            # تحميل النتائج
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ تحميل النتائج كـ CSV", data=csv, file_name='predictions.csv', mime='text/csv')
