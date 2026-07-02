# Laptop Price Lab — Streamlit Dashboard

داشبورد تفاعلي كامل مبني على مشروع توقع أسعار اللابتوبات (Tuwaiq ML).
6 صفحات: Overview · Data Cleaning · Feature Engineering · EDA · Model Showdown · Price Predictor

## التشغيل — مباشرة بدون أي خطوة إضافية
الداتا (`laptopData.csv`) موجودة داخل الملف المضغوط جنب `app.py`، فبس:
```bash
pip install -r requirements.txt
streamlit run app.py
```
لو تبين تستخدمين نسختك من Kaggle، استبدلي الملف بنفس الاسم — البايبلاين يتعامل مع القيم الفاضية و`?` تلقائياً.

## العملة
الأسعار الأصلية بالداتا سيت **روبية هندية (INR)** لأن الداتا من السوق الهندي.
فيه مبدّل عملة بالسايدبار (ريال سعودي افتراضياً / دولار / روبية) — التحويل تقريبي، والموديل نفسه يتدرب على INR.

## النشر على Streamlit Cloud (نفس طريقة FIFA dashboard)
1. Repo جديد على GitHub وارفعي: `app.py` + `requirements.txt` + `laptopData.csv` + مجلد `.streamlit/`
2. من share.streamlit.io → New app → اختاري الريبو → Main file: `app.py` → Deploy

## ملاحظات
- البايبلاين نسخة حرفية من النوت بوك: نفس التنظيف والهندسة، نفس `random_state=42`،
  نفس LR+PCA(0.95) مقابل RF(300, depth=20) — النتائج تطابق نتائجك.
- التصميم: hero متحرك، كروت زجاجية بأنيميشن، عداد سعر نيون، بدون إيموجي.
