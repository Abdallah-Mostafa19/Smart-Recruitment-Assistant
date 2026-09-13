import streamlit as st
import pandas as pd
import numpy as np
import joblib

# إعداد الصفحة وتنسيق المظهر
st.set_page_config(
    page_title="Smart Recruitment Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Smart Recruitment Assistant Dashboard")
st.markdown("نظام الفرز الذكي لتقييم المتقدمين للوظائف وتحديد درجة الملاءمة وترتيب الكفاءات.")

# ----------------------------------------------------
# تحميل النموذج والبيانات المجهزة
# ----------------------------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('recruitment_model.pkl')
        feature_names = joblib.load('model_features.pkl')
        return model, feature_names
    except Exception:
        return None, None

model, feature_names = load_ml_assets()

# ----------------------------------------------------
# تبويبات لوحة التحكم
# ----------------------------------------------------
tab_eval, tab_top10, tab_batch = st.tabs([
    "🎯 تقييم مرشح فردي", 
    "🏆 قائمة أفضل 10 مرشحين (Top 10)", 
    "📁 تقييم دفعة سير ذاتية (Batch CSV)"
])

# ====================================================
# التبويب الأول: تقييم مرشح فردي
# ====================================================
with tab_eval:
    if model is None:
        st.warning("⚠️ لم يتم العثور على ملف النموذج `recruitment_model.pkl`. يرجى التأكد من تشغيل كود التدريب وحفظ النموذج أولاً.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📋 بيانات المرشح")
        
        city_dev = st.slider(
            "مؤشر تنمية المدينة (City Development Index):",
            min_value=0.40, max_value=1.00, value=0.75, step=0.01
        )
        
        experience = st.slider(
            "سنوات الخبرة (Years of Experience):",
            min_value=0, max_value=25, value=5
        )
        
        training_hours = st.number_input(
            "ساعات التدريب السابقة (Training Hours):",
            min_value=1, max_value=350, value=45
        )
        
        last_new_job = st.selectbox(
            "عدد السنوات منذ آخر تغيير وظيفي (Last New Job):",
            options=[0, 1, 2, 3, 4, 5],
            format_func=lambda x: "لم يغير وظيفته بعد" if x == 0 else ("> 4 سنوات" if x == 5 else f"{x} سنوات")
        )
        
        relevent_exp = st.selectbox("هل لديه خبرة ذات صلة بالمجال؟", ["Has relevent experience", "No relevent experience"])
        education_level = st.selectbox("المستوى التعليمي:", ["Graduate", "Masters", "Phd", "High School", "Primary School"])
        
        evaluate_btn = st.button("🚀 تقييم المرشح الآن", use_container_width=True)

    with col2:
        st.subheader("📊 النتيجة والتوصية")
        
        if evaluate_btn:
            if model is not None and feature_names is not None:
                # إنشاء سجل متوافق مع تمثيل الميزات
                input_df = pd.DataFrame(0, index=[0], columns=feature_names)
                
                # تعبئة القيم المباشرة
                if 'city_development_index' in input_df.columns:
                    input_df['city_development_index'] = city_dev
                if 'experience' in input_df.columns:
                    input_df['experience'] = experience
                if 'training_hours' in input_df.columns:
                    input_df['training_hours'] = training_hours
                if 'last_new_job' in input_df.columns:
                    input_df['last_new_job'] = last_new_job
                
                # تعبئة الفئات المرمزة
                rel_col = f"relevent_experience_{relevent_exp}"
                if rel_col in input_df.columns:
                    input_df[rel_col] = 1
                
                edu_col = f"education_level_{education_level}"
                if edu_col in input_df.columns:
                    input_df[edu_col] = 1
                
                # حساب الاحتمالية
                prob = model.predict_proba(input_df)[0][1] * 100
            else:
                # تقدير بديل في حال عدم تحميل النموذج
                prob = min(100, int((city_dev * 45) + (experience * 1.5) + (training_hours * 0.08) + (last_new_job * 2)))

            # عرض الدرجة التقييمية
            st.metric(
                label="درجة ملاءمة المرشح (Suitability Score)",
                value=f"{prob:.1f}%"
            )
            
            st.progress(int(prob))
            
            if prob >= 65:
                st.success("✅ **التوصية:** مؤهل بقوة — الانتقال للمقابلة الفنية مباشرة (Advance to Next Stage)")
            elif prob >= 50:
                st.info("🔎 **التوصية:** قيد المراجعة — يتطلب تقييماً إضافياً أو مقابلة استكشافية (Hold / Review)")
            else:
                st.warning("⚠️ **التوصية:** غير مؤهل حالياً — حفظ السيرة الذاتية للأرشيف (Archive)")

# ====================================================
# التبويب الثاني: ترتيب أفضل 10 مرشحين (Top 10)
# ====================================================
with tab_top10:
    st.subheader("🏆 أفضل 10 مرشحين من قاعدة البيانات")
    st.markdown("يتم استخراج المرشحين بناءً على أعلى درجات الملاءمة المحسوبة بواسطة النموذج.")
    
    try:
        data = pd.read_csv('aug_train.csv')
        
        # تجهيز البيانات لتوقع الاحتمالية
        temp_df = data.copy()
        temp_df['experience'] = temp_df['experience'].replace({'>20': '21', '<1': '0'})
        temp_df['experience'] = pd.to_numeric(temp_df['experience'], errors='coerce').fillna(5)
        
        temp_df['last_new_job'] = temp_df['last_new_job'].replace({'>4': '5', 'never': '0'})
        temp_df['last_new_job'] = pd.to_numeric(temp_df['last_new_job'], errors='coerce').fillna(1)
        
        if model is not None and feature_names is not None:
            # تجهيز الميزات لتطابق مدخلات النموذج
            encoded = pd.get_dummies(temp_df.drop(columns=['enrollee_id', 'city', 'target'], errors='ignore'), drop_first=True)
            encoded = encoded.reindex(columns=feature_names, fill_value=0)
            data['Suitability_Score_%'] = (model.predict_proba(encoded)[:, 1] * 100).round(2)
        else:
            data['Suitability_Score_%'] = (
                (data['city_development_index'].fillna(0.7) * 45) + 
                (temp_df['experience'] * 1.5) + 
                (data['training_hours'].fillna(30) * 0.08)
            ).round(2)
        
        # استخراج أعلى 10 مرشحين
        top_10_df = data.sort_values(by='Suitability_Score_%', ascending=False).head(10)
        
        # اختيار الأعمدة المعروضة
        display_cols = [
            'enrollee_id', 'education_level', 'experience', 
            'training_hours', 'last_new_job', 'Suitability_Score_%'
        ]
        available_cols = [c for c in display_cols if c in top_10_df.columns]
        
        st.dataframe(
            top_10_df[available_cols].reset_index(drop=True),
            use_container_width=True
        )
        
    except FileNotFoundError:
        st.info("💡 ضع ملف `aug_train.csv` في نفس مسار المشروع لعرض ترتيب المرشحين الفعلي تلقائياً.")

# ====================================================
# التبويب الثالث: تقييم ملف CSV بالكامل (Batch Scoring)
# ====================================================
with tab_batch:
    st.subheader("📁 رفع قائمة مرشحين جديدة وتصفيتها")
    uploaded_file = st.file_uploader("قم برفع ملف المتقدمين (CSV):", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"تم تحميل البيانات بنجاح: {len(batch_df)} مرشح")
        
        if st.button("فرز وتصنيف القائمة بالكامل"):
            if model is not None and feature_names is not None:
                # معالجة سريعة للملف المرفوع
                b_clean = batch_df.copy()
                if 'experience' in b_clean.columns:
                    b_clean['experience'] = b_clean['experience'].replace({'>20': '21', '<1': '0'})
                    b_clean['experience'] = pd.to_numeric(b_clean['experience'], errors='coerce').fillna(0)
                if 'last_new_job' in b_clean.columns:
                    b_clean['last_new_job'] = b_clean['last_new_job'].replace({'>4': '5', 'never': '0'})
                    b_clean['last_new_job'] = pd.to_numeric(b_clean['last_new_job'], errors='coerce').fillna(0)
                
                b_encoded = pd.get_dummies(b_clean, drop_first=True)
                b_encoded = b_encoded.reindex(columns=feature_names, fill_value=0)
                
                batch_df['Suitability_Score_%'] = (model.predict_proba(b_encoded)[:, 1] * 100).round(2)
                batch_df['Recommendation'] = np.where(
                    batch_df['Suitability_Score_%'] >= 65, 'Advance',
                    np.where(batch_df['Suitability_Score_%'] >= 50, 'Review', 'Archive')
                )
                
                st.dataframe(batch_df.sort_values(by='Suitability_Score_%', ascending=False), use_container_width=True)
                
                # زر تحميل النتائج
                csv_result = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 تحميل قائمة النتائج والتوصيات (CSV)", data=csv_result, file_name="recruitment_screened_results.csv", mime="text/csv")