# ============================================================
#  LAPTOP PRICE LAB — ML Dashboard
#  Built on Ritaj Albogami's laptop price prediction pipeline
# ============================================================

import glob

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Laptop Price Lab", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------
# Design system
# ------------------------------------------------------------
NAVY = "#0A1628"
CARD = "#12213A"
PURPLE = "#8B5CF6"
ORANGE = "#FF8C42"
CYAN = "#38BDF8"
TEXT = "#E8EDF5"
MUTED = "#8FA3BF"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT),
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(bgcolor=CARD, font_color=TEXT, bordercolor=PURPLE),
)
COLOR_SEQ = [PURPLE, ORANGE, CYAN, "#F472B6", "#34D399", "#FBBF24", "#A78BFA", "#60A5FA"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }}

/* ---------- Aurora background ---------- */
.stApp {{
    background:
        radial-gradient(900px 500px at 85% -10%, rgba(139,92,246,0.14), transparent 60%),
        radial-gradient(800px 500px at -10% 30%, rgba(56,189,248,0.10), transparent 60%),
        radial-gradient(700px 600px at 60% 110%, rgba(255,140,66,0.08), transparent 60%),
        {NAVY};
}}

/* ---------- Keyframes ---------- */
@keyframes gradientShift {{ 0%{{background-position:0% 50%}} 50%{{background-position:100% 50%}} 100%{{background-position:0% 50%}} }}
@keyframes fadeUp {{ from{{opacity:0; transform:translateY(18px)}} to{{opacity:1; transform:translateY(0)}} }}
@keyframes floatOrb {{ 0%,100%{{transform:translate(0,0) scale(1)}} 50%{{transform:translate(-30px,25px) scale(1.15)}} }}
@keyframes pulseGlow {{ 0%,100%{{box-shadow:0 0 22px rgba(255,140,66,0.25)}} 50%{{box-shadow:0 0 45px rgba(255,140,66,0.55)}} }}
@keyframes shimmer {{ 0%{{background-position:-400px 0}} 100%{{background-position:400px 0}} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}

/* page content entrance */
div[data-testid="stMainBlockContainer"] {{ animation: fadeUp .55s cubic-bezier(.22,1,.36,1); }}

/* ---------- Hero ---------- */
.hero {{
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, {CARD}, #1A1035, #0E2A46, {CARD});
    background-size: 300% 300%;
    animation: gradientShift 14s ease infinite;
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 20px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.4rem;
}}
.hero::before, .hero::after {{
    content: ''; position: absolute; border-radius: 50%;
    filter: blur(65px); opacity: .45; pointer-events: none;
    animation: floatOrb 10s ease-in-out infinite;
}}
.hero::before {{ width: 320px; height: 320px; background: {PURPLE}; top: -130px; right: -70px; }}
.hero::after  {{ width: 260px; height: 260px; background: {ORANGE}; bottom: -140px; left: -50px; animation-delay: -5s; }}
.hero h1 {{
    font-size: 2.7rem; margin: 0 0 .45rem 0; position: relative; z-index: 1;
    background: linear-gradient(90deg, {ORANGE}, {PURPLE}, {CYAN}, {ORANGE});
    background-size: 300% auto;
    animation: gradientShift 8s linear infinite;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: 0; position: relative; z-index: 1; max-width: 62rem; }}

/* ---------- Glass metric cards ---------- */
.metric-card {{
    background: linear-gradient(160deg, rgba(18,33,58,0.9), rgba(18,33,58,0.55));
    backdrop-filter: blur(8px);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 16px;
    padding: 1.15rem 1.35rem;
    height: 100%;
    animation: fadeUp .6s cubic-bezier(.22,1,.36,1) both;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}}
.metric-card:hover {{
    transform: translateY(-5px);
    border-color: {PURPLE};
    box-shadow: 0 12px 32px rgba(139,92,246,0.28);
}}
.metric-card .label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; }}
.metric-card .value {{ font-family: 'Space Grotesk'; font-size: 1.95rem; font-weight: 700; color: {TEXT}; }}
.metric-card .sub {{ color: {ORANGE}; font-size: .85rem; }}
.metric-card.win {{ border: 1px solid {ORANGE}; animation: fadeUp .6s both, pulseGlow 3.2s ease-in-out infinite; }}
.metric-card.d1 {{ animation-delay: .05s }} .metric-card.d2 {{ animation-delay: .12s }}
.metric-card.d3 {{ animation-delay: .19s }} .metric-card.d4 {{ animation-delay: .26s }}

/* ---------- Chips / misc ---------- */
.step-chip {{
    display: inline-block; background: rgba(56,189,248,0.12); color: {CYAN};
    border: 1px solid rgba(56,189,248,0.35); border-radius: 999px;
    padding: .18rem .85rem; font-size: .78rem; margin-bottom: .5rem;
    letter-spacing: .03em;
}}

/* ---------- Sidebar nav as pills ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0C1B31, {NAVY});
    border-right: 1px solid rgba(139,92,246,0.2);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    background: rgba(18,33,58,0.6);
    border: 1px solid transparent;
    border-radius: 12px;
    padding: .55rem .9rem;
    margin-bottom: .35rem;
    width: 100%;
    transition: all .2s ease;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    border-color: {CYAN};
    transform: translateX(4px);
}}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
    background: linear-gradient(90deg, rgba(139,92,246,0.25), rgba(255,140,66,0.18));
    border: 1px solid {PURPLE};
}}

div[data-testid="stSidebarNav"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Currency (dataset prices are in Indian Rupees — INR)
# ------------------------------------------------------------
CURRENCIES = {
    "SAR — ريال سعودي": ("SAR", 0.045),
    "USD — دولار": ("$", 0.012),
    "INR — روبية هندية (الأصلية)": ("₹", 1.0),
}

# ------------------------------------------------------------
# Data pipeline (mirrors the notebook exactly)
# ------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_raw():
    for path in ["laptopData.csv", "laptop_data.csv", "data/laptopData.csv"]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    others = [p for p in glob.glob("*.csv")]
    if others:
        return pd.read_csv(others[0])
    return None


def cpu_category(cpu: str) -> str:
    cpu = cpu.lower()
    if "core i7" in cpu:
        return "Core i7"
    elif "core i5" in cpu:
        return "Core i5"
    elif "core i3" in cpu:
        return "Core i3"
    elif "celeron" in cpu or "pentium" in cpu or "atom" in cpu:
        return "Intel Basic"
    elif "amd" in cpu:
        return "AMD"
    else:
        return "Other"  # Core M, Xeon, ARM


INCH_TYPO_FIXES = {35.6: 15.6, 27.3: 17.3, 31.6: 11.6, 25.6: 15.6, 24.0: 14.0, 33.5: 13.5}


@st.cache_data(show_spinner="Cleaning + engineering features...")
def clean_and_engineer(raw: pd.DataFrame):
    audit = {}
    df = raw.copy()
    audit["raw_shape"] = df.shape

    for col in [df.columns[0]]:
        if str(col).startswith("Unnamed"):
            df = df.drop(columns=[col])

    audit["question_marks"] = (df.astype(str) == "?").sum()
    audit["question_marks"] = audit["question_marks"][audit["question_marks"] > 0]

    n_before = len(df)
    df = df.dropna(how="all")
    audit["empty_rows_dropped"] = n_before - len(df)
    df = df.dropna(subset=["Price"])

    n_before = len(df)
    df = df.drop_duplicates()
    audit["duplicates_dropped"] = n_before - len(df)

    # ScreenResolution -> 7 features
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    res = df["ScreenResolution"].str.split().str[-1]
    df[["Resolution_X", "Resolution_Y"]] = res.str.split("x", expand=True).astype(int)
    df["ScreenSize"] = df["Resolution_X"] * df["Resolution_Y"]
    df["UltraHD"] = df["ScreenResolution"].str.contains("4K").astype(int)
    df["FullHD"] = df["ScreenResolution"].str.contains("Full").astype(int)
    df["Quad"] = df["ScreenResolution"].str.contains("Quad").astype(int)

    # Inches
    df["Inches"] = pd.to_numeric(df["Inches"].replace("?", np.nan), errors="coerce")
    df["Inches"] = df["Inches"].fillna(df["Inches"].median())
    audit["inch_fixes"] = []
    for wrong, right in INCH_TYPO_FIXES.items():
        n = int((df["Inches"] == wrong).sum())
        if n:
            audit["inch_fixes"].append({"wrong": wrong, "fixed_to": right, "rows": n})
        df["Inches"] = df["Inches"].replace(wrong, right)

    # CPU
    df["CPU_Brand"] = df["Cpu"].str.split().str[0]
    df["CPU_Speed"] = df["Cpu"].str.split().str[-1].str.replace("GHz", "").astype(float)
    df["CPU_Name"] = df["Cpu"].apply(cpu_category)

    # Ram
    df["Ram"] = df["Ram"].astype(str).str.replace("GB", "").astype(int)

    # Memory -> SSD / HDD / Flash / Hybrid
    mem = (df["Memory"].astype(str)
           .str.replace("1.0TB", "1024GB", regex=False)
           .str.replace("2TB", "2048GB", regex=False)
           .str.replace("1TB", "1024GB", regex=False)
           .replace("?", np.nan).replace("nan", np.nan))
    for c in ["SSD", "HDD", "Flash", "Hybrid"]:
        df[c] = np.nan
    for i in df.index:
        if pd.isna(mem.loc[i]):
            continue
        df.loc[i, ["SSD", "HDD", "Flash", "Hybrid"]] = 0
        for part in str(mem.loc[i]).split("+"):
            part = part.strip()
            try:
                size = float(part.split("GB")[0].strip())
            except ValueError:
                continue
            if "SSD" in part:
                df.loc[i, "SSD"] += size
            elif "HDD" in part:
                df.loc[i, "HDD"] += size
            elif "Flash" in part:
                df.loc[i, "Flash"] += size
            elif "Hybrid" in part:
                df.loc[i, "Hybrid"] += size
    for c in ["SSD", "HDD", "Flash", "Hybrid"]:
        df[c] = df[c].fillna(df[c].median())

    # GPU / OpSys / Weight
    df["GPU_Brand"] = df["Gpu"].str.split().str[0]
    df["OpSys"] = df["OpSys"].replace({
        "Windows 10": "Windows", "Windows 7": "Windows", "Windows 10 S": "Windows",
        "Mac OS X": "Mac", "macOS": "Mac", "Android": "No OS"})
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str).str.replace("kg", "").replace("?", np.nan),
                                 errors="coerce")
    df["Weight"] = df["Weight"].fillna(df["Weight"].median())

    audit["clean_shape"] = df.shape
    return df.reset_index(drop=True), audit


@st.cache_data(show_spinner=False)
def encode(clean: pd.DataFrame):
    df = clean.copy()
    df = pd.get_dummies(df, columns=["CPU_Brand", "CPU_Name"], dtype=int)
    df = pd.get_dummies(df, columns=["GPU_Brand"], dtype=int)
    df = pd.concat([df, pd.get_dummies(df["OpSys"], dtype=int)], axis=1)
    df = pd.get_dummies(df, columns=["Company", "TypeName"], dtype=int)
    df = df.drop(columns=["ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys"])
    X = df.drop(columns=["Price"])
    y = df["Price"]
    return X, y


CONTINUOUS_COLS = ["Inches", "Ram", "CPU_Speed", "SSD", "HDD", "Flash", "Hybrid",
                   "Weight", "Resolution_X", "Resolution_Y", "ScreenSize"]


@st.cache_resource(show_spinner="Training models (Linear Regression + Random Forest)...")
def train_models(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    binary_cols = [c for c in X_train.columns if c not in CONTINUOUS_COLS]
    scaler = StandardScaler()
    Xtr_cont = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    Xte_cont = scaler.transform(X_test[CONTINUOUS_COLS])
    pca = PCA(n_components=0.95)
    Xtr_pca = pca.fit_transform(Xtr_cont)
    Xte_pca = pca.transform(Xte_cont)
    Xtr_final = np.hstack([Xtr_pca, X_train[binary_cols].values])
    Xte_final = np.hstack([Xte_pca, X_test[binary_cols].values])

    lr = LinearRegression().fit(Xtr_final, y_train)
    pred_lr = lr.predict(Xte_final)

    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=2,
                               random_state=42, n_jobs=-1).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    def metrics(y_true, y_pred):
        return {"MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mean_squared_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred)}

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"rf": rf, "y_test": y_test, "pred_lr": pred_lr, "pred_rf": pred_rf,
            "m_lr": metrics(y_test, pred_lr), "m_rf": metrics(y_test, pred_rf),
            "pca_var": pca.explained_variance_ratio_, "n_components": pca.n_components_,
            "importances": importances, "feature_cols": list(X.columns)}


# ------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------
def card(label, value, sub="", win=False, delay=1):
    cls = f"metric-card d{delay}" + (" win" if win else "")
    st.markdown(f'<div class="{cls}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def fig_show(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def animated_price(value: float, symbol: str):
    """Neon count-up price display."""
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">
    <div style="display:flex;align-items:baseline;gap:.5rem;">
      <span id="price" style="font-family:'Space Grotesk',sans-serif;font-size:54px;font-weight:700;
        color:{ORANGE};text-shadow:0 0 14px rgba(255,140,66,.55), 0 0 40px rgba(255,140,66,.30);"></span>
      <span style="font-family:'Space Grotesk',sans-serif;font-size:22px;color:{MUTED};">{symbol}</span>
    </div>
    <script>
      const target = {value:.0f};
      const el = document.getElementById('price');
      const dur = 900; const t0 = performance.now();
      function tick(t) {{
        const p = Math.min((t - t0) / dur, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * eased).toLocaleString('en-US');
        if (p < 1) requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    </script>
    """, height=78)


# ------------------------------------------------------------
# Pages
# ------------------------------------------------------------
def page_home(clean, audit, results, fmt):
    hero("Laptop Price Lab",
         "An end-to-end machine learning story: from a messy CSV to a live price predictor — "
         "data cleaning, feature engineering, EDA, and a model showdown between Linear Regression + PCA and Random Forest.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Raw rows", f"{audit['raw_shape'][0]:,}", "as loaded from the CSV", delay=1)
    with c2:
        card("Clean rows", f"{audit['clean_shape'][0]:,}",
             f"-{audit['empty_rows_dropped']} empty · -{audit['duplicates_dropped']} duplicates", delay=2)
    with c3:
        card("Engineered features", f"{len(results['feature_cols'])}", "from just 11 raw columns", delay=3)
    with c4:
        card("Best model R²", f"{results['m_rf']['R2']:.3f}", "Random Forest", win=True, delay=4)

    st.markdown("### The journey")
    st.markdown(
        """
| Stage | What happened |
|---|---|
| **Cleaning** | Dropped empty rows and exact duplicates, hunted `?` placeholders, and fixed screen-size typos (like `35.6"` → `15.6"`) by comparing against each brand's real lineup |
| **Feature engineering** | One `ScreenResolution` string became **7 features** (Touchscreen, IPS, X/Y resolution, pixel count, 4K/FHD/QHD flags). `Memory` was split into SSD / HDD / Flash / Hybrid in GB |
| **EDA** | Price distributions, brand comparisons, and correlations — before and after engineering |
| **Models** | Linear Regression (scaled + PCA on continuous features) vs Random Forest on raw features |
| **Predictor** | Build a laptop spec-by-spec and get a live price from the trained Random Forest |
        """
    )

    st.markdown("### A quick taste of the data")
    d = clean.copy()
    d["PriceDisp"] = d["Price"].map(fmt["rate"].__mul__)
    fig = px.scatter(d, x="Ram", y="PriceDisp", color="TypeName",
                     color_discrete_sequence=COLOR_SEQ, opacity=0.65,
                     title="RAM vs price — the single strongest signal in the dataset",
                     labels={"Ram": "RAM (GB)", "PriceDisp": f"Price ({fmt['sym']})"})
    fig_show(fig)


def page_cleaning(raw, clean, audit):
    hero("Data Cleaning",
         "You can't trust a model trained on data you haven't questioned. "
         "Every fix here has a reason — nothing was deleted or changed blindly.")

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Fully empty rows removed", audit["empty_rows_dropped"],
             "impossible to recover — no information at all", delay=1)
    with c2:
        card("Exact duplicates removed", audit["duplicates_dropped"],
             "verified with df.loc[a].equals(df.loc[b])", delay=2)
    with c3:
        n_q = int(audit["question_marks"].sum()) if len(audit["question_marks"]) else 0
        card("'?' placeholders found", n_q, "replaced with NaN, then filled with the median", delay=3)

    left, right = st.columns(2)
    with left:
        st.markdown('<span class="step-chip">Missing values strategy</span>', unsafe_allow_html=True)
        st.markdown(
            "**Median, not mean.** Numeric gaps (`Inches`, `Weight`, storage) were filled with the **median** "
            "because these columns contain outliers — the mean gets dragged toward them, the median stays robust."
        )
        if len(audit["question_marks"]):
            qm = audit["question_marks"].reset_index()
            qm.columns = ["Column", "'?' count"]
            st.dataframe(qm, hide_index=True, use_container_width=True)
        else:
            st.info("This copy of the CSV contains no '?' placeholders — the pipeline still checks every column for them.")
    with right:
        st.markdown('<span class="step-chip">Detective work: screen-size typos</span>', unsafe_allow_html=True)
        st.markdown(
            "Filtering for screens **< 10\" or > 19\"** exposed impossible values. Comparing each one against the "
            "same brand's real lineup showed they were **data-entry typos** — a single wrong digit:"
        )
        fixes = pd.DataFrame([
            {"Company": "Asus", "Typo": '35.6"', "Fixed to": '15.6"', "Clue": "3 typed instead of 1"},
            {"Company": "Dell", "Typo": '27.3"', "Fixed to": '17.3"', "Clue": '27 rows of 17.3" exist at Dell'},
            {"Company": "Dell", "Typo": '31.6"', "Fixed to": '11.6"', "Clue": "3 typed instead of 1"},
            {"Company": "Lenovo", "Typo": '25.6" / 35.6"', "Fixed to": '15.6"', "Clue": "15.6 is Lenovo's most common size"},
            {"Company": "Lenovo", "Typo": '24"', "Fixed to": '14"', "Clue": 'no 24" laptop exists'},
            {"Company": "Microsoft", "Typo": '33.5"', "Fixed to": '13.5"', "Clue": 'Surface only ships 13.5"'},
        ])
        st.dataframe(fixes, hide_index=True, use_container_width=True)

    st.markdown("### Inches — before vs after")
    raw_in = pd.to_numeric(raw.get("Inches"), errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Box(x=raw_in, name="Before", marker_color=MUTED))
    fig.add_trace(go.Box(x=clean["Inches"], name="After", marker_color=ORANGE))
    fig.update_layout(title="Screen size distribution — the impossible outliers disappear after the fixes",
                      xaxis_title="Inches", height=320)
    fig_show(fig)

    with st.expander("Peek at the raw data"):
        st.dataframe(raw.head(30), use_container_width=True)


def page_engineering(clean):
    hero("Feature Engineering",
         "Raw text columns hide numbers a model can actually learn from. "
         "This is where 11 messy columns became a rich numeric feature set.")

    st.markdown('<span class="step-chip">Try it live</span>', unsafe_allow_html=True)
    st.markdown("Pick any laptop and watch its raw strings turn into model-ready features:")

    idx = st.slider("Laptop row", 0, len(clean) - 1, 42)
    row = clean.iloc[idx]

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Raw strings")
        st.code(
            f'Company:          {row["Company"]}\n'
            f'ScreenResolution: {row["ScreenResolution"]}\n'
            f'Cpu:              {row["Cpu"]}\n'
            f'Memory:           {row["Memory"]}\n'
            f'Gpu:              {row["Gpu"]}\n'
            f'Weight:           (was text like "1.37kg")',
            language=None)
    with right:
        st.markdown("#### Engineered features")
        eng = pd.DataFrame({
            "Feature": ["Touchscreen", "IPS", "Resolution_X", "Resolution_Y", "ScreenSize (pixels)",
                        "UltraHD", "FullHD", "Quad", "CPU_Brand", "CPU_Name", "CPU_Speed (GHz)",
                        "SSD (GB)", "HDD (GB)", "Flash (GB)", "Hybrid (GB)", "GPU_Brand", "Weight (kg)"],
            "Value": [row["Touchscreen"], row["IPS"], row["Resolution_X"], row["Resolution_Y"],
                      f'{row["ScreenSize"]:,}', row["UltraHD"], row["FullHD"], row["Quad"],
                      row["CPU_Brand"], row["CPU_Name"], row["CPU_Speed"],
                      row["SSD"], row["HDD"], row["Flash"], row["Hybrid"],
                      row["GPU_Brand"], row["Weight"]],
        })
        eng["Value"] = eng["Value"].astype(str)
        st.dataframe(eng, hide_index=True, use_container_width=True, height=420)

    st.markdown("### Why it matters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1 string → 7 features**  \n`ScreenResolution` alone produced Touchscreen, IPS, "
                    "X/Y resolution, total pixel count, and 4K / Full HD / QHD flags.")
    with c2:
        st.markdown("**Storage untangled**  \n`\"128GB SSD + 1TB HDD\"` became four clean numeric columns "
                    "(SSD / HDD / Flash / Hybrid), with TB converted to GB first.")
    with c3:
        st.markdown("**Categories grouped**  \nCPUs mapped to 6 tiers (i7 / i5 / i3 / Intel Basic / AMD / Other) "
                    "and operating systems merged (Windows 7 & 10 → Windows) so the model isn't fooled by rare labels.")


def page_eda(clean, fmt):
    hero("Exploratory Data Analysis",
         "Interactive versions of the notebook's EDA — filter, hover, and zoom.")

    d = clean.copy()
    d["PriceDisp"] = d["Price"] * fmt["rate"]
    companies = st.multiselect("Filter by company", sorted(d["Company"].unique()), default=[])
    if companies:
        d = d[d["Company"].isin(companies)]
    price_label = f"Price ({fmt['sym']})"

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(d, x="PriceDisp", nbins=50, color_discrete_sequence=[PURPLE],
                           labels={"PriceDisp": price_label},
                           title="Price distribution — right-skewed: most laptops are mid-range, a few are very expensive")
        fig_show(fig)
    with c2:
        by = st.selectbox("Box plot: price by...", ["TypeName", "Company", "CPU_Name", "GPU_Brand", "OpSys", "Ram"])
        fig = px.box(d, x=by, y="PriceDisp", color=by, color_discrete_sequence=COLOR_SEQ,
                     labels={"PriceDisp": price_label}, title=f"Price by {by}")
        fig.update_layout(showlegend=False)
        fig_show(fig)

    c3, c4 = st.columns(2)
    with c3:
        counts = d["Company"].value_counts().reset_index()
        counts.columns = ["Company", "Laptops"]
        fig = px.bar(counts, x="Company", y="Laptops", color_discrete_sequence=[CYAN],
                     title="Laptops per company — Dell, Lenovo and HP dominate the market")
        fig_show(fig)
    with c4:
        fig = px.scatter(d, x="Ram", y="PriceDisp", color="TypeName", color_discrete_sequence=COLOR_SEQ,
                         opacity=0.65, labels={"Ram": "RAM (GB)", "PriceDisp": price_label},
                         title="RAM vs price — a clear upward staircase")
        fig_show(fig)

    st.markdown("### Correlation heatmap")
    num_cols = ["Price", "Ram", "CPU_Speed", "SSD", "HDD", "Inches", "Weight",
                "ScreenSize", "Touchscreen", "IPS", "FullHD", "UltraHD"]
    corr = d[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title="RAM (+0.7) and SSD lead the pack; HDD correlates negatively — HDD-only laptops are the budget tier")
    fig.update_layout(height=560)
    fig_show(fig)


def page_models(results, fmt):
    hero("Model Showdown",
         "Two philosophies: Linear Regression with scaled features + PCA, versus Random Forest on raw features. "
         "Same train/test split, same test set — a fair fight.")

    m_lr, m_rf = results["m_lr"], results["m_rf"]
    rate, sym = fmt["rate"], fmt["sym"]
    st.markdown("### The scoreboard")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: card("LR · R²", f"{m_lr['R2']:.3f}", delay=1)
    with c2: card("LR · MAE", f"{m_lr['MAE']*rate:,.0f} {sym}", delay=2)
    with c3: card("LR · MSE", f"{m_lr['MSE']:.2e}", "in INR²", delay=3)
    with c4: card("RF · R²", f"{m_rf['R2']:.3f}", "winner", win=True, delay=1)
    with c5: card("RF · MAE", f"{m_rf['MAE']*rate:,.0f} {sym}", "lower error", win=True, delay=2)
    with c6: card("RF · MSE", f"{m_rf['MSE']:.2e}", "in INR²", win=True, delay=3)

    st.markdown("")
    y_test = results["y_test"] * rate
    line = go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                      mode="lines", line=dict(color=MUTED, dash="dash"), name="Perfect prediction")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([go.Scatter(x=y_test, y=results["pred_lr"] * rate, mode="markers",
                                    marker=dict(color=PURPLE, opacity=0.55), name="LR + PCA"), line])
        fig.update_layout(title="Linear Regression + PCA — actual vs predicted",
                          xaxis_title=f"Actual price ({sym})", yaxis_title=f"Predicted price ({sym})")
        fig_show(fig)
    with c2:
        fig = go.Figure([go.Scatter(x=y_test, y=results["pred_rf"] * rate, mode="markers",
                                    marker=dict(color=ORANGE, opacity=0.55), name="Random Forest"), line])
        fig.update_layout(title="Random Forest — points hug the diagonal more tightly",
                          xaxis_title=f"Actual price ({sym})", yaxis_title=f"Predicted price ({sym})")
        fig_show(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### PCA — what it kept")
        var = results["pca_var"]
        cum = np.cumsum(var)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var))], y=var,
                             name="Per component", marker_color=CYAN))
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(var))], y=cum, name="Cumulative",
                                 mode="lines+markers", line=dict(color=ORANGE)))
        fig.update_layout(title=f"{results['n_components']} components kept from 11 continuous features "
                                f"({cum[-1]*100:.1f}% variance)", yaxis_title="Explained variance ratio")
        fig_show(fig)
    with c4:
        st.markdown("### What drives the price (Random Forest)")
        top = results["importances"].head(10).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", color_discrete_sequence=[PURPLE],
                     labels={"x": "Importance", "y": ""},
                     title=f"RAM alone explains {results['importances'].iloc[0]*100:.0f}% of the model's decisions")
        fig_show(fig)

    st.markdown("### Verdict")
    st.markdown(
        f"""
**Random Forest wins** (R² **{m_rf['R2']:.3f}** vs **{m_lr['R2']:.3f}**, plus lower MAE and MSE).
The relationship between laptop specs and price is **non-linear** — RAM upgrades don't add a fixed amount, and brand premiums
interact with specs — and tree ensembles capture those interactions naturally. PCA didn't help here: it compresses variance,
but the variance it discards still carried price signal, and Random Forest never needed the compression (it splits on raw values,
not distances). Scaling was applied **only to continuous features** — binary 0/1 dummies were deliberately left untouched,
since StandardScaler is meant for continuous values.
        """
    )


RESOLUTION_PRESETS = {
    "HD 1366×768": (1366, 768, 0, 0, 0),
    "Full HD 1920×1080": (1920, 1080, 0, 1, 0),
    "QHD 2560×1440": (2560, 1440, 0, 0, 1),
    "Retina 2560×1600": (2560, 1600, 0, 0, 0),
    "QHD+ 3200×1800": (3200, 1800, 0, 0, 1),
    "4K UHD 3840×2160": (3840, 2160, 1, 0, 0),
}


def page_predictor(clean, results, fmt):
    hero("Price Predictor",
         "Build a laptop spec-by-spec — the trained Random Forest (300 trees) prices it instantly.")

    rf, cols = results["rf"], results["feature_cols"]
    rate, sym = fmt["rate"], fmt["sym"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Identity")
        company = st.selectbox("Company", sorted(clean["Company"].unique()),
                               index=sorted(clean["Company"].unique()).index("Dell"))
        typename = st.selectbox("Type", sorted(clean["TypeName"].unique()),
                                index=sorted(clean["TypeName"].unique()).index("Notebook"))
        opsys = st.selectbox("Operating system", sorted(clean["OpSys"].unique()),
                             index=sorted(clean["OpSys"].unique()).index("Windows"))
        weight = st.slider("Weight (kg)", 0.7, 4.5, 1.8, 0.1)
    with c2:
        st.markdown("#### Display")
        inches = st.slider("Screen size (inches)", 10.1, 18.4, 15.6, 0.1)
        res_name = st.selectbox("Resolution", list(RESOLUTION_PRESETS.keys()), index=1)
        touch = st.toggle("Touchscreen")
        ips = st.toggle("IPS panel", value=True)
    with c3:
        st.markdown("#### Performance")
        cpu_brand = st.selectbox("CPU brand", sorted(clean["CPU_Brand"].unique()),
                                 index=sorted(clean["CPU_Brand"].unique()).index("Intel"))
        cpu_name = st.selectbox("CPU tier", ["Core i7", "Core i5", "Core i3", "Intel Basic", "AMD", "Other"], index=1)
        cpu_speed = st.slider("CPU speed (GHz)", 0.9, 3.6, 2.5, 0.1)
        ram = st.select_slider("RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)
        ssd = st.select_slider("SSD (GB)", options=[0, 128, 256, 512, 1024, 2048], value=256)
        hdd = st.select_slider("HDD (GB)", options=[0, 500, 1024, 2048], value=0)
        gpu_brand = st.selectbox("GPU brand", sorted(clean["GPU_Brand"].unique()),
                                 index=sorted(clean["GPU_Brand"].unique()).index("Intel"))

    rx, ry, uhd, fhd, quad = RESOLUTION_PRESETS[res_name]
    spec = {"Inches": inches, "Ram": ram, "Weight": weight,
            "Touchscreen": int(touch), "IPS": int(ips),
            "Resolution_X": rx, "Resolution_Y": ry, "ScreenSize": rx * ry,
            "UltraHD": uhd, "FullHD": fhd, "Quad": quad,
            "CPU_Speed": cpu_speed, "SSD": ssd, "HDD": hdd, "Flash": 0, "Hybrid": 0,
            f"CPU_Brand_{cpu_brand}": 1, f"CPU_Name_{cpu_name}": 1,
            f"GPU_Brand_{gpu_brand}": 1, opsys: 1,
            f"Company_{company}": 1, f"TypeName_{typename}": 1}
    X_row = pd.DataFrame([spec]).reindex(columns=cols, fill_value=0)
    price_inr = float(rf.predict(X_row)[0])
    price = price_inr * rate

    st.markdown("---")
    left, right = st.columns([1, 1.2])
    with left:
        animated_price(price, sym)
        st.caption("Dataset prices are in Indian Rupees (INR) — conversion rates are approximate.")
        max_p = float(clean["Price"].max()) * rate
        q1, q2 = clean["Price"].quantile(0.33) * rate, clean["Price"].quantile(0.66) * rate
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=price,
            number={"valueformat": ",.0f", "suffix": f" {sym}"},
            gauge={"axis": {"range": [0, max_p], "tickformat": "~s"},
                   "bar": {"color": ORANGE},
                   "steps": [{"range": [0, q1], "color": "rgba(56,189,248,0.18)"},
                             {"range": [q1, q2], "color": "rgba(139,92,246,0.18)"},
                             {"range": [q2, max_p], "color": "rgba(255,140,66,0.18)"}]}))
        fig.update_layout(height=270, margin=dict(l=30, r=30, t=30, b=10))
        fig_show(fig)
        pct = (clean["Price"] < price_inr).mean() * 100
        st.markdown(f"This build is priced above **{pct:.0f}%** of the {len(clean):,} laptops in the dataset.")

    with right:
        st.markdown("#### What if you change the RAM?")
        ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
        rows = []
        for r in ram_options:
            s = dict(spec)
            s["Ram"] = r
            rows.append(s)
        X_sens = pd.DataFrame(rows).reindex(columns=cols, fill_value=0)
        sens_pred = rf.predict(X_sens) * rate
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ram_options, y=sens_pred, mode="lines+markers",
                                 line=dict(color=CYAN, width=3, shape="spline"), name="Predicted price"))
        fig.add_trace(go.Scatter(x=[ram], y=[price], mode="markers",
                                 marker=dict(color=ORANGE, size=16, symbol="star"), name="Your build"))
        fig.update_layout(title="Same build, different RAM — the non-linear jumps the Random Forest learned",
                          xaxis_title="RAM (GB)", yaxis_title=f"Predicted price ({sym})", height=330)
        fig_show(fig)

        st.markdown("#### Closest real laptops in the dataset")
        spec_cols = ["Ram", "SSD", "HDD", "Inches", "CPU_Speed", "Weight"]
        ref = clean[spec_cols].copy()
        stds = ref.std().replace(0, 1)
        target = pd.Series({"Ram": ram, "SSD": ssd, "HDD": hdd, "Inches": inches,
                            "CPU_Speed": cpu_speed, "Weight": weight})
        dist = (((ref - target) / stds) ** 2).sum(axis=1)
        nearest = clean.loc[dist.nsmallest(5).index,
                            ["Company", "TypeName", "CPU_Name", "Ram", "SSD", "HDD", "Inches", "Price"]].copy()
        nearest["Price"] = (nearest["Price"] * rate).map(lambda p: f"{p:,.0f} {sym}")
        st.dataframe(nearest, hide_index=True, use_container_width=True)


# ------------------------------------------------------------
# App shell
# ------------------------------------------------------------
def main():
    raw = load_raw()
    if raw is None:
        st.markdown('<div class="hero"><h1>Laptop Price Lab</h1>'
                    '<p>laptopData.csv was not found next to app.py — place the CSV in the same folder and refresh.</p></div>',
                    unsafe_allow_html=True)
        st.stop()

    clean, audit = clean_and_engineer(raw)
    X, y = encode(clean)
    results = train_models(X, y)

    with st.sidebar:
        st.markdown("## Laptop Price Lab")
        st.caption("Laptop price prediction · end-to-end ML")
        page = st.radio("Navigate", ["Overview", "Data Cleaning", "Feature Engineering",
                                     "EDA", "Model Showdown", "Price Predictor"],
                        label_visibility="collapsed")
        st.markdown("---")
        currency = st.selectbox("Currency / العملة", list(CURRENCIES.keys()), index=0)
        sym, rate = CURRENCIES[currency]
        st.caption("الأسعار الأصلية بالروبية الهندية — الداتا سيت من السوق الهندي، والتحويل تقريبي.")
        st.markdown("---")
        st.markdown(f"**Best model:** Random Forest  \n**R²:** {results['m_rf']['R2']:.3f} · "
                    f"**MAE:** {results['m_rf']['MAE']*rate:,.0f} {sym}")
        st.caption("Built by Ritaj Albogami · Tuwaiq Academy")

    fmt = {"sym": sym, "rate": rate}

    if page == "Overview":
        page_home(clean, audit, results, fmt)
    elif page == "Data Cleaning":
        page_cleaning(raw, clean, audit)
    elif page == "Feature Engineering":
        page_engineering(clean)
    elif page == "EDA":
        page_eda(clean, fmt)
    elif page == "Model Showdown":
        page_models(results, fmt)
    else:
        page_predictor(clean, results, fmt)


if __name__ == "__main__":
    main()
