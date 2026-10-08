
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI Talent Retention Analyzer",
    page_icon="🧠",
    layout="wide",
)

COMPANY = "EV-Tech Mobility & Energy"
DEFAULT_FILE = "AI_Talent_Retention_Synthetic_Data.xlsx"

REQUIRED_SHEETS = {
    "Historico_Empleados",
    "Empleados_Actuales",
    "Entrevistas_Salida",
}

TARGET = "Voluntary_Attrition_12m"
ID_COL = "Employee_ID"

SPANISH_STOPWORDS = {
    "a","al","algo","algunas","algunos","ante","antes","como","con","contra","cual",
    "cuando","de","del","desde","donde","durante","e","el","ella","ellas","ellos","en",
    "entre","era","eramos","es","esa","esas","ese","eso","esos","esta","estaba","estaban",
    "este","esto","estos","fue","ha","habia","hacia","hasta","hay","la","las","le","les",
    "lo","los","mas","me","mi","mis","muy","ni","no","nos","o","para","pero","por","porque",
    "que","se","si","sin","sobre","su","sus","tambien","tenia","un","una","uno","unos","unas",
    "y","ya","yo","trabajo","trabajar","empresa","equipo","persona","personas","parte","aun",
    "aunque","ser","estar","haber","tener","hacer","mucho","mucha","muchos","muchas"
}

ACTION_MAP = {
    "Crecimiento profesional": [
        ("Career_Growth_Sat", "satisfacción con crecimiento"),
        ("Months_Since_Promotion", "tiempo desde la última promoción"),
        ("Internal_Move_24m", "movilidad interna"),
        ("Learning_Opportunity_Sat", "oportunidades de aprendizaje"),
    ],
    "Manager / liderazgo": [
        ("Manager_Sat", "satisfacción con liderazgo"),
        ("Recognition_Sat", "reconocimiento"),
    ],
    "Carga de trabajo / balance": [
        ("Work_Life_Balance", "balance vida-trabajo"),
        ("Overtime_Hours_Month", "horas extra"),
        ("Avg_Weekly_Hours", "horas semanales"),
    ],
    "Compensación / beneficios": [
        ("Total_Comp_Sat", "satisfacción con compensación"),
        ("Salary_Market_Index", "posición salarial frente al mercado"),
    ],
    "Reconocimiento": [
        ("Recognition_Sat", "reconocimiento"),
        ("Manager_Sat", "liderazgo"),
    ],
    "Oportunidad externa": [
        ("Recruiter_Contacts_6m", "contactos de reclutadores"),
        ("Career_Growth_Sat", "crecimiento interno"),
    ],
    "Reorganización / estabilidad": [
        ("Reorg_12m", "reorganizaciones recientes"),
        ("Engagement_Score", "engagement"),
    ],
}


def clean_label(text):
    return str(text).replace("_", " ").strip().title()


@st.cache_data(show_spinner=False)
def load_default_workbook(path_str):
    xls = pd.ExcelFile(path_str)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


@st.cache_data(show_spinner=False)
def load_uploaded_workbook(file_bytes):
    bio = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(bio)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


def validate_workbook(book):
    missing = REQUIRED_SHEETS - set(book.keys())
    if missing:
        raise ValueError(
            "Faltan las hojas requeridas: " + ", ".join(sorted(missing))
        )
    hist = book["Historico_Empleados"]
    current = book["Empleados_Actuales"]
    exits = book["Entrevistas_Salida"]

    hist_required = {ID_COL, TARGET, "Department"}
    current_required = {ID_COL, "Department"}
    exit_required = {
        ID_COL, "Department", "Primary_Exit_Reason",
        "Exit_Comment", "What_Would_Have_Made_You_Stay"
    }

    for name, df, req in [
        ("Historico_Empleados", hist, hist_required),
        ("Empleados_Actuales", current, current_required),
        ("Entrevistas_Salida", exits, exit_required),
    ]:
        missing_cols = req - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"En {name} faltan columnas: {', '.join(sorted(missing_cols))}"
            )


def _fit_design_matrix(df, feature_cols):
    """Preprocesamiento explícito y estable para evitar dependencias de pipelines complejos."""
    X = df[feature_cols].copy()
    categorical = [
        c for c in feature_cols
        if X[c].dtype == "object" or str(X[c].dtype).startswith("category")
    ]
    numeric = [c for c in feature_cols if c not in categorical]

    medians = {}
    means = {}
    stds = {}
    modes = {}

    for col in numeric:
        vals = pd.to_numeric(X[col], errors="coerce")
        median = float(vals.median()) if vals.notna().any() else 0.0
        vals = vals.fillna(median).astype(float)
        mean = float(vals.mean())
        std = float(vals.std())
        if not np.isfinite(std) or std == 0:
            std = 1.0
        X[col] = (vals - mean) / std
        medians[col] = median
        means[col] = mean
        stds[col] = std

    for col in categorical:
        values = X[col].astype("string")
        mode = values.dropna().mode()
        fill = str(mode.iloc[0]) if len(mode) else "Sin dato"
        X[col] = values.fillna(fill).astype(str)
        modes[col] = fill

    X_enc = pd.get_dummies(X, columns=categorical, dummy_na=False, dtype=float)
    X_enc = X_enc.astype(float)

    prep = {
        "feature_cols": feature_cols,
        "categorical": categorical,
        "numeric": numeric,
        "medians": medians,
        "means": means,
        "stds": stds,
        "modes": modes,
        "encoded_columns": X_enc.columns.tolist(),
    }
    return X_enc, prep


def _transform_design_matrix(df, prep):
    X = df.copy()
    for col in prep["feature_cols"]:
        if col not in X.columns:
            X[col] = np.nan
    X = X[prep["feature_cols"]].copy()

    for col in prep["numeric"]:
        vals = pd.to_numeric(X[col], errors="coerce")
        vals = vals.fillna(prep["medians"][col]).astype(float)
        X[col] = (vals - prep["means"][col]) / prep["stds"][col]

    for col in prep["categorical"]:
        X[col] = (
            X[col]
            .astype("string")
            .fillna(prep["modes"][col])
            .astype(str)
        )

    X_enc = pd.get_dummies(
        X,
        columns=prep["categorical"],
        dummy_na=False,
        dtype=float,
    )
    X_enc = X_enc.reindex(columns=prep["encoded_columns"], fill_value=0.0)
    return X_enc.astype(float)


def train_attrition_model(hist):
    feature_cols = [c for c in hist.columns if c not in {ID_COL, TARGET}]
    y = hist[TARGET].astype(int)

    auc = np.nan
    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(hist)),
            test_size=0.25,
            stratify=y,
            random_state=42,
        )
        X_train, prep_test = _fit_design_matrix(hist.iloc[train_idx], feature_cols)
        X_test = _transform_design_matrix(hist.iloc[test_idx], prep_test)
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        temp_model = LogisticRegression(max_iter=3000, random_state=42)
        temp_model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, temp_model.predict_proba(X_test)[:, 1])
    except Exception:
        # La app puede continuar aun si no es posible calcular el hold-out AUC.
        auc = np.nan

    X_all, prep = _fit_design_matrix(hist, feature_cols)
    lr = LogisticRegression(max_iter=3000, random_state=42)
    lr.fit(X_all, y)
    bundle = {"model": lr, "prep": prep}
    return bundle, feature_cols, prep["numeric"], prep["categorical"], auc


def score_current(model_bundle, current, feature_cols):
    scored = current.copy()
    X_current = _transform_design_matrix(scored, model_bundle["prep"])
    scored["Risk_Score"] = model_bundle["model"].predict_proba(X_current)[:, 1]

    low_cut = scored["Risk_Score"].quantile(0.40)
    high_cut = scored["Risk_Score"].quantile(0.80)
    scored["Risk_Band"] = np.select(
        [
            scored["Risk_Score"] >= high_cut,
            scored["Risk_Score"] >= low_cut,
        ],
        ["Prioridad alta", "Prioridad media"],
        default="Prioridad baja",
    )
    return scored, low_cut, high_cut

def numeric_attrition_drivers(hist):
    rows = []
    y = hist[TARGET].astype(int)
    for col in hist.columns:
        if col in {ID_COL, TARGET}:
            continue
        if pd.api.types.is_numeric_dtype(hist[col]):
            leave = hist.loc[y.eq(1), col].mean()
            stay = hist.loc[y.eq(0), col].mean()
            sd = hist[col].std()
            effect = (leave - stay) / sd if pd.notna(sd) and sd > 0 else 0
            rows.append({
                "Variable": col,
                "Media_salida": leave,
                "Media_permanece": stay,
                "Efecto_estandarizado": effect,
                "Magnitud": abs(effect),
            })
    return (
        pd.DataFrame(rows)
        .sort_values("Magnitud", ascending=False)
        .reset_index(drop=True)
    )


def model_coefficients(model_bundle, numeric, categorical):
    lr = model_bundle["model"]
    names = model_bundle["prep"]["encoded_columns"]
    coefs = lr.coef_[0]
    imp = pd.DataFrame({
        "Feature": names,
        "Coefficient": coefs,
        "Magnitude": np.abs(coefs),
    })
    imp["Direction"] = np.where(
        imp["Coefficient"] > 0,
        "Asociado con mayor riesgo",
        "Asociado con menor riesgo",
    )
    return imp.sort_values("Magnitude", ascending=False).reset_index(drop=True)

def top_tfidf_terms(text_series, top_n=15):
    texts = (
        text_series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-záéíóúüñ0-9\s]", " ", regex=True)
        .tolist()
    )
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return pd.DataFrame(columns=["Término", "Relevancia"])

    vec = TfidfVectorizer(
        stop_words=list(SPANISH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )
    try:
        mat = vec.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["Término", "Relevancia"])

    scores = np.asarray(mat.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({
        "Término": terms[idx],
        "Relevancia": scores[idx],
    })


def safe_mode(series):
    m = series.dropna().mode()
    return m.iloc[0] if len(m) else (series.dropna().iloc[0] if series.dropna().size else np.nan)


def make_typical_profile(current, feature_cols, department=None):
    base = current.copy()
    if department and department != "Todos":
        subset = base[base["Department"].eq(department)]
        if len(subset) >= 3:
            base = subset

    row = {}
    for col in feature_cols:
        if col not in base.columns:
            row[col] = np.nan
        elif pd.api.types.is_numeric_dtype(base[col]):
            row[col] = float(base[col].median())
        else:
            row[col] = safe_mode(base[col])
    return row


def scenario_score(model_bundle, feature_cols, profile):
    frame = pd.DataFrame([{c: profile.get(c, np.nan) for c in feature_cols}])
    X_frame = _transform_design_matrix(frame, model_bundle["prep"])
    return float(model_bundle["model"].predict_proba(X_frame)[0, 1])


def fmt_pct(x):
    return f"{x * 100:.1f}%"


def risk_band_from_score(score, low_cut, high_cut):
    if score >= high_cut:
        return "Prioridad alta"
    if score >= low_cut:
        return "Prioridad media"
    return "Prioridad baja"


def insight_cards(hist, scored, exits):
    drivers = numeric_attrition_drivers(hist)
    top_reasons = exits["Primary_Exit_Reason"].value_counts(normalize=True)
    cards = []

    for reason, share in top_reasons.head(4).items():
        mapped = ACTION_MAP.get(reason, [])
        candidate = None
        for variable, label in mapped:
            found = drivers[drivers["Variable"].eq(variable)]
            if not found.empty:
                candidate = (variable, label, found.iloc[0])
                break

        if candidate:
            variable, label, row = candidate
            direction = row["Efecto_estandarizado"]
            if variable in scored.columns:
                high = scored[scored["Risk_Band"].eq("Prioridad alta")][variable].mean()
                overall = scored[variable].mean()
            else:
                high = overall = np.nan

            cards.append({
                "reason": reason,
                "share": share,
                "variable": variable,
                "label": label,
                "effect": direction,
                "high": high,
                "overall": overall,
            })
    return cards


# -------------------- SIDEBAR / LOAD --------------------
with st.sidebar:
    st.title("🧠 Talent Retention")
    st.caption(COMPANY)
    st.divider()
    uploaded = st.file_uploader(
        "Opcional: sustituir el dataset",
        type=["xlsx"],
        help="Debe mantener las hojas y columnas del archivo sintético.",
    )
    st.caption("Si no subes un archivo, la app utiliza el dataset sintético incluido.")
    st.divider()
    st.info(
        "Uso académico. Los registros incluidos son sintéticos y el modelo no debe usarse "
        "para decisiones individuales de empleo."
    )

try:
    if uploaded is not None:
        book = load_uploaded_workbook(uploaded.getvalue())
        source_label = f"Archivo cargado: {uploaded.name}"
    else:
        default_path = Path(__file__).with_name(DEFAULT_FILE)
        if not default_path.exists():
            st.error(
                f"No encuentro {DEFAULT_FILE}. Súbelo desde la barra lateral o colócalo junto a app.py."
            )
            st.stop()
        book = load_default_workbook(str(default_path))
        source_label = "Dataset sintético incluido"

    validate_workbook(book)
except Exception as exc:
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

hist = book["Historico_Empleados"].copy()
current = book["Empleados_Actuales"].copy()
exits = book["Entrevistas_Salida"].copy()

model, feature_cols, numeric_cols, categorical_cols, auc = train_attrition_model(hist)
scored, low_cut, high_cut = score_current(model, current, feature_cols)
drivers = numeric_attrition_drivers(hist)
coeffs = model_coefficients(model, numeric_cols, categorical_cols)

# -------------------- HEADER --------------------
st.title("AI Talent Retention Analyzer")
st.caption(f"{COMPANY} · {source_label}")
st.write(
    "Herramienta de People Analytics para identificar patrones de rotación, "
    "analizar entrevistas de salida y orientar acciones de retención."
)

tabs = st.tabs([
    "📊 Executive Overview",
    "🎯 Retention Risk",
    "💬 Exit Interviews",
    "💡 Retention Insights",
    "🧪 What-if Simulator",
    "ℹ️ Metodología",
])

# -------------------- OVERVIEW --------------------
with tabs[0]:
    attrition_rate = hist[TARGET].mean()
    high_count = int((scored["Risk_Band"] == "Prioridad alta").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Histórico", f"{len(hist):,}")
    c2.metric("Rotación histórica", fmt_pct(attrition_rate))
    c3.metric("Empleados actuales", f"{len(current):,}")
    c4.metric("Entrevistas de salida", f"{len(exits):,}")
    c5.metric("Segmento prioridad alta", f"{high_count}")

    st.divider()

    left, right = st.columns(2)
    with left:
        dept = (
            hist.groupby("Department", as_index=False)[TARGET]
            .mean()
            .rename(columns={TARGET: "Rotación"})
            .sort_values("Rotación", ascending=False)
        )
        fig = px.bar(
            dept,
            x="Rotación",
            y="Department",
            orientation="h",
            text=dept["Rotación"].map(lambda x: f"{x:.1%}"),
            title="Rotación histórica por departamento",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_drv = drivers.head(8).copy()
        top_drv["Etiqueta"] = top_drv["Variable"].map(clean_label)
        top_drv["Dirección"] = np.where(
            top_drv["Efecto_estandarizado"] > 0,
            "Mayor en quienes salieron",
            "Menor en quienes salieron",
        )
        fig = px.bar(
            top_drv.sort_values("Magnitud"),
            x="Magnitud",
            y="Etiqueta",
            orientation="h",
            color="Dirección",
            title="Variables con mayores diferencias históricas",
        )
        fig.update_layout(xaxis_title="Magnitud estandarizada", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lectura ejecutiva")
    top_reason = exits["Primary_Exit_Reason"].value_counts().index[0]
    top_dept = dept.iloc[0]["Department"]
    st.write(
        f"En los datos sintéticos, **{top_reason}** es el motivo de salida más frecuente. "
        f"El departamento con mayor rotación histórica observada es **{top_dept}**. "
        "Estas señales deben interpretarse como patrones organizativos, no como causalidad."
    )

# -------------------- RETENTION RISK --------------------
with tabs[1]:
    st.subheader("Riesgo relativo por segmentos")
    st.caption(
        "El modelo genera un score relativo. La app agrupa personas para evitar usar el resultado "
        "como una decisión individual de empleo."
    )

    f1, f2, f3 = st.columns(3)
    dept_options = ["Todos"] + sorted(scored["Department"].dropna().astype(str).unique().tolist())
    level_options = ["Todos"] + sorted(scored["Level"].dropna().astype(str).unique().tolist())
    mode_options = ["Todos"] + sorted(scored["Work_Mode"].dropna().astype(str).unique().tolist())

    dep_filter = f1.selectbox("Departamento", dept_options)
    lvl_filter = f2.selectbox("Nivel", level_options)
    mode_filter = f3.selectbox("Modalidad", mode_options)

    filtered = scored.copy()
    if dep_filter != "Todos":
        filtered = filtered[filtered["Department"].eq(dep_filter)]
    if lvl_filter != "Todos":
        filtered = filtered[filtered["Level"].eq(lvl_filter)]
    if mode_filter != "Todos":
        filtered = filtered[filtered["Work_Mode"].eq(mode_filter)]

    if filtered.empty:
        st.warning("No hay registros con esa combinación de filtros.")
    else:
        a, b, c = st.columns(3)
        a.metric("Personas en el segmento", len(filtered))
        b.metric("Score medio", fmt_pct(filtered["Risk_Score"].mean()))
        c.metric(
            "Prioridad alta",
            f"{(filtered['Risk_Band'].eq('Prioridad alta')).sum()} "
            f"({(filtered['Risk_Band'].eq('Prioridad alta')).mean():.1%})"
        )

        summary = (
            filtered.groupby("Department", as_index=False)
            .agg(
                Personas=("Employee_ID", "count"),
                Score_Medio=("Risk_Score", "mean"),
                Overtime_Medio=("Overtime_Hours_Month", "mean"),
                Manager_Sat=("Manager_Sat", "mean"),
                WLB=("Work_Life_Balance", "mean"),
                Career_Growth=("Career_Growth_Sat", "mean"),
            )
            .sort_values("Score_Medio", ascending=False)
        )
        fig = px.bar(
            summary,
            x="Department",
            y="Score_Medio",
            text=summary["Score_Medio"].map(lambda x: f"{x:.1%}"),
            title="Score medio por departamento (filtros aplicados)",
        )
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="", yaxis_title="Score relativo")
        st.plotly_chart(fig, use_container_width=True)

        show = summary.copy()
        for col in ["Score_Medio"]:
            show[col] = show[col].map(lambda x: f"{x:.1%}")
        for col in ["Overtime_Medio", "Manager_Sat", "WLB", "Career_Growth"]:
            show[col] = show[col].map(lambda x: f"{x:.1f}")
        st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Factores del modelo")
    st.caption(
        "Coeficientes de regresión logística: muestran asociación dentro del modelo, no causalidad."
    )
    topcoef = coeffs.head(12).copy()
    topcoef["Feature"] = topcoef["Feature"].map(clean_label)
    fig = px.bar(
        topcoef.sort_values("Magnitude"),
        x="Magnitude",
        y="Feature",
        orientation="h",
        color="Direction",
    )
    fig.update_layout(xaxis_title="Importancia relativa", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- EXIT INTERVIEWS --------------------
with tabs[2]:
    st.subheader("Exit Interview Analyzer")
    departments_exit = ["Todos"] + sorted(exits["Department"].dropna().astype(str).unique().tolist())
    exit_dep = st.selectbox("Filtrar entrevistas por departamento", departments_exit, key="exit_dep")

    exit_view = exits.copy()
    if exit_dep != "Todos":
        exit_view = exit_view[exit_view["Department"].eq(exit_dep)]

    if exit_view.empty:
        st.warning("No hay entrevistas para ese departamento.")
    else:
        reasons = (
            exit_view["Primary_Exit_Reason"]
            .value_counts()
            .rename_axis("Motivo")
            .reset_index(name="Entrevistas")
        )
        reasons["Porcentaje"] = reasons["Entrevistas"] / reasons["Entrevistas"].sum()

        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            fig = px.bar(
                reasons.sort_values("Entrevistas"),
                x="Entrevistas",
                y="Motivo",
                orientation="h",
                text="Entrevistas",
                title="Motivos principales de salida",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            recommend = (
                exit_view["Would_Recommend"]
                .value_counts()
                .rename_axis("Respuesta")
                .reset_index(name="Entrevistas")
            )
            fig = px.pie(
                recommend,
                names="Respuesta",
                values="Entrevistas",
                hole=0.45,
                title="¿Recomendaría la empresa?",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("NLP: temas frecuentes en comentarios abiertos")
        combined = (
            exit_view["Exit_Comment"].fillna("")
            + " "
            + exit_view["What_Would_Have_Made_You_Stay"].fillna("")
        )
        terms = top_tfidf_terms(combined, 15)
        if terms.empty:
            st.info("No hay suficiente texto para calcular términos relevantes.")
        else:
            fig = px.bar(
                terms.sort_values("Relevancia"),
                x="Relevancia",
                y="Término",
                orientation="h",
                title="Términos y expresiones con mayor peso TF-IDF",
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Ver ejemplos anonimizados de comentarios"):
            sample_cols = [
                "Department",
                "Primary_Exit_Reason",
                "Positive_Aspect",
                "What_Would_Have_Made_You_Stay",
                "Exit_Comment",
            ]
            st.dataframe(
                exit_view[sample_cols].head(20),
                use_container_width=True,
                hide_index=True,
            )

# -------------------- RETENTION INSIGHTS --------------------
with tabs[3]:
    st.subheader("Cruce de evidencia: salida histórica + plantilla actual")
    st.write(
        "Esta sección conecta lo que expresaron quienes salieron con variables presentes "
        "en la plantilla actual para orientar preguntas e intervenciones de RH."
    )

    cards = insight_cards(hist, scored, exits)
    if not cards:
        st.info("No se pudieron generar insights con la estructura actual.")
    else:
        for i, item in enumerate(cards, start=1):
            reason = item["reason"]
            share = item["share"]
            label = item["label"]
            high = item["high"]
            overall = item["overall"]

            st.markdown(f"### Insight {i:02d} · {reason}")
            c1, c2 = st.columns(2)
            c1.metric("Presencia en entrevistas", fmt_pct(share))
            if pd.notna(high) and pd.notna(overall):
                c2.metric(
                    f"{clean_label(item['variable'])} · prioridad alta vs total",
                    f"{high:.2f} vs {overall:.2f}",
                )

            if reason == "Crecimiento profesional":
                action = (
                    "Revisar rutas de carrera, conversaciones de desarrollo, movilidad interna "
                    "y tiempo desde la última promoción en los segmentos de mayor prioridad."
                )
            elif reason == "Manager / liderazgo":
                action = (
                    "Priorizar diagnóstico de liderazgo, calidad de feedback, reconocimiento "
                    "y soporte del manager en las áreas con señales más fuertes."
                )
            elif reason == "Carga de trabajo / balance":
                action = (
                    "Revisar distribución de horas extra, dotación y prioridades antes de "
                    "implementar intervenciones generales de bienestar."
                )
            elif reason == "Compensación / beneficios":
                action = (
                    "Contrastar satisfacción de compensación con posición salarial frente al mercado "
                    "y con otras causas de salida; evitar asumir que salario explica todo el fenómeno."
                )
            else:
                action = (
                    f"Profundizar en {label} y contrastarlo con entrevistas, métricas de equipo "
                    "y contexto operativo antes de decidir una intervención."
                )
            st.success("Acción sugerida para RH: " + action)
            st.divider()

# -------------------- WHAT IF --------------------
with tabs[4]:
    st.subheader("What-if Simulator")
    st.caption(
        "Simulación académica: modifica factores de un perfil típico y observa cómo cambia "
        "el score del modelo. No es una predicción causal ni una recomendación individual."
    )

    departments = ["Todos"] + sorted(current["Department"].dropna().astype(str).unique().tolist())
    sim_dep = st.selectbox("Perfil típico del departamento", departments, key="sim_dep")
    base_profile = make_typical_profile(current, feature_cols, sim_dep)

    controls = {}
    c1, c2 = st.columns(2)

    with c1:
        controls["Manager_Sat"] = st.slider(
            "Satisfacción con liderazgo (1–5)",
            1, 5, int(round(base_profile.get("Manager_Sat", 3)))
        )
        controls["Career_Growth_Sat"] = st.slider(
            "Satisfacción con crecimiento (1–5)",
            1, 5, int(round(base_profile.get("Career_Growth_Sat", 3)))
        )
        controls["Work_Life_Balance"] = st.slider(
            "Work-life balance (1–5)",
            1, 5, int(round(base_profile.get("Work_Life_Balance", 3)))
        )
        controls["Total_Comp_Sat"] = st.slider(
            "Satisfacción con compensación (1–5)",
            1, 5, int(round(base_profile.get("Total_Comp_Sat", 3)))
        )

    with c2:
        controls["Overtime_Hours_Month"] = st.slider(
            "Horas extra al mes",
            0, 80, int(round(base_profile.get("Overtime_Hours_Month", 20)))
        )
        controls["Months_Since_Promotion"] = st.slider(
            "Meses desde última promoción",
            0, 72, int(round(base_profile.get("Months_Since_Promotion", 18)))
        )
        controls["Engagement_Score"] = st.slider(
            "Engagement score (0–100)",
            0, 100, int(round(base_profile.get("Engagement_Score", 70)))
        )
        controls["Internal_Move_24m"] = st.selectbox(
            "Movilidad interna en últimos 24 meses",
            [0, 1],
            index=int(round(base_profile.get("Internal_Move_24m", 0))),
            format_func=lambda x: "Sí" if x == 1 else "No",
        )

    baseline_score = scenario_score(model, feature_cols, base_profile)
    scenario = dict(base_profile)
    scenario.update(controls)
    scenario_score_value = scenario_score(model, feature_cols, scenario)

    m1, m2, m3 = st.columns(3)
    m1.metric("Score perfil típico", fmt_pct(baseline_score))
    m2.metric(
        "Score escenario",
        fmt_pct(scenario_score_value),
        delta=f"{(scenario_score_value - baseline_score) * 100:+.1f} pp",
        delta_color="inverse",
    )
    m3.metric(
        "Banda relativa",
        risk_band_from_score(scenario_score_value, low_cut, high_cut),
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=scenario_score_value * 100,
        number={"suffix": "%"},
        title={"text": "Score relativo del escenario"},
        gauge={"axis": {"range": [0, 100]}},
    ))
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- METHODOLOGY --------------------
with tabs[5]:
    st.subheader("Metodología y límites")
    st.markdown(
        f"""
**Caso:** {COMPANY}, empresa ficticia del sector de movilidad eléctrica, energía y manufactura avanzada.

**Datos:** {len(hist)} registros históricos, {len(current)} empleados actuales y {len(exits)} entrevistas de salida.  
Todos los registros son **sintéticos**.

**Modelo:** regresión logística con variables numéricas estandarizadas y variables categóricas codificadas one-hot.

**Evaluación técnica:** AUC en una partición hold-out de 25%: **{auc:.2f}** si el cálculo fue posible.

**NLP:** extracción de términos mediante TF-IDF sobre respuestas abiertas de entrevistas de salida.

**Uso correcto:** identificar patrones y segmentos que ameritan análisis adicional.

**No debe usarse para:** decidir despidos, promociones, compensación o acciones adversas sobre una persona.  
El score refleja asociaciones del dataset sintético y **no demuestra causalidad**.
"""
    )

    st.subheader("Variables excluidas deliberadamente")
    st.write(
        "El identificador del empleado no se usa como predictor. La aplicación tampoco incorpora "
        "atributos sensibles como raza, religión, discapacidad, orientación sexual o afiliación política."
    )

    if "Diccionario" in book:
        with st.expander("Diccionario de variables"):
            st.dataframe(book["Diccionario"], use_container_width=True, hide_index=True)

    if "Fuentes" in book:
        with st.expander("Fuentes utilizadas para diseñar el escenario sintético"):
            st.dataframe(book["Fuentes"], use_container_width=True, hide_index=True)
