import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard de materias primas asistido por IA",
    layout="wide"
)

# Rutas de archivos (usan la carpeta data/ del repo)
PRICE_FILES = {
    "Urea": "data/UREAFI.xlsx",
    "Metanol": "data/METANOLF.xlsx",
    "Madera": "data/MADERAF.xlsx",
}
IS_FILE = "data/Edo de Res.xlsx"   # tu estado de resultados


# ---------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------
@st.cache_data
def load_price_data(path: str) -> pd.DataFrame:
    """Carga datos de precios y calcula rendimientos logarítmicos."""
    df = pd.read_excel(path)
    cols = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Detectar columnas de fecha y precio
    date_col = [c for c in df.columns if "date" in c][0]
    price_col_candidates = [c for c in df.columns if "px_last" in c or "price" in c]
    price_col = price_col_candidates[0]

    df = df[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Price"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["log_ret"] = np.log(df["Price"] / df["Price"].shift(1))
    return df


def compute_price_metrics(df: pd.DataFrame) -> dict:
    """Calcula volatilidad, kurtosis y skewness a partir de rendimientos."""
    rets = df["log_ret"].dropna()
    vol_daily = rets.std()
    vol_annual = vol_daily * np.sqrt(252)
    kurtosis = rets.kurt()
    skew = rets.skew()

    metrics = {
        "price_last": df["Price"].iloc[-1],
        "price_min": df["Price"].min(),
        "price_max": df["Price"].max(),
        "start": df["Date"].min(),
        "end": df["Date"].max(),
        "vol_daily": vol_daily,
        "vol_annual": vol_annual,
        "kurtosis": kurtosis,
        "skew": skew,
    }
    return metrics


@st.cache_data
def load_income_statement(path: str) -> pd.DataFrame:
    """
    Carga tu estado de resultados simple:
    Columna 0: nombres (Revenue, Cost Of Goods Sold, Gross Profit,
                       Op expense, Op Income, [fila vacía de impuestos], Net Income)
    Columna 1: valores
    """
    return pd.read_excel(path)


def compute_base_is_from_template(df: pd.DataFrame) -> dict:
    col_name = df.columns[0]
    col_val = df.columns[1]

    def get_val(row_name):
        return float(df.loc[df[col_name] == row_name, col_val].iloc[0])

    revenue = get_val("Revenue")
    cogs = get_val("Cost Of Goods Sold")
    gross_profit = get_val("Gross Profit")
    op_expense = get_val("Op expense")
    op_income = get_val("Op Income")
    net_income = get_val("Net Income")

    # Fila de impuestos: nombre NaN
    tax_row = df[df[col_name].isna()]
    if not tax_row.empty:
        tax = float(tax_row[col_val].iloc[0])
    else:
        tax = op_income - net_income

    tax_rate = tax / op_income if op_income != 0 else 0.0

    return {
        "sales": revenue,
        "cogs": cogs,
        "opex": op_expense,
        "gross": gross_profit,
        "op": op_income,
        "tax": tax,
        "net": net_income,
        "tax_rate": tax_rate,
    }


def simulate_scenario(base: dict, share_cogs: float, shock_pct: float) -> dict:
    """
    Aplica un shock al % del COGS asociado al material.
    share_cogs: proporción (0–1) del COGS de ese material.
    shock_pct: shock en % (ej. 0.30 = +30%).
    """
    ventas = base["sales"]
    cogs_base = base["cogs"]
    opex = base["opex"]
    tax_rate = base["tax_rate"]

    delta_cogs = cogs_base * share_cogs * shock_pct
    cogs_scn = cogs_base + delta_cogs

    gross_scn = ventas - cogs_scn
    op_scn = gross_scn - opex
    tax_scn = op_scn * tax_rate
    net_scn = op_scn - tax_scn

    return {
        "sales": ventas,
        "cogs": cogs_scn,
        "opex": opex,
        "gross": gross_scn,
        "op": op_scn,
        "tax": tax_scn,
        "net": net_scn,
    }


def build_summary_table(base: dict, scenario: dict) -> pd.DataFrame:
    """Tabla con Base vs Escenario y cambios en márgenes."""
    rows = []
    labels = [
        ("gross", "Gross Profit"),
        ("op", "Op Income"),
        ("net", "Net Income"),
    ]
    sales = base["sales"]

    for key, label in labels:
        base_val = base[key]
        scn_val = scenario[key]
        delta_abs = scn_val - base_val
        delta_pct = delta_abs / base_val if base_val != 0 else np.nan
        margin_base = base_val / sales if sales != 0 else np.nan
        margin_scn = scn_val / sales if sales != 0 else np.nan
        delta_margin_pp = (margin_scn - margin_base) * 100

        rows.append(
            {
                "Métrica": label,
                "Base": base_val,
                "Escenario": scn_val,
                "Δ absoluta": delta_abs,
                "Δ %": delta_pct,
                "Margen base": margin_base,
                "Margen esc.": margin_scn,
                "Δ margen (p.p.)": delta_margin_pp,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# UI PRINCIPAL
# ---------------------------------------------------------
st.title("📊 Dashboard de materias primas – Volatilidad & Impacto en EERR")

st.markdown(
    """
    Este dashboard usa tu plantilla de **estado de resultados simple** y los precios
    históricos de **Urea, Metanol y Madera** guardados en la carpeta `data/`.
    """
)

# ---------- ESTADO DE RESULTADOS BASE ----------
is_df = load_income_statement(IS_FILE)
base_is = compute_base_is_from_template(is_df)

st.sidebar.header("Estado de resultados base")
st.sidebar.dataframe(is_df, use_container_width=True)
st.sidebar.metric("Revenue", f"{base_is['sales']:,.0f}")
st.sidebar.metric("Net Income", f"{base_is['net']:,.0f}")

# ---------- SECCIONES POR MATERIAL ----------
for material, path in PRICE_FILES.items():
    st.markdown("---")
    st.header(f"🔹 {material}")

    # Carga de datos de precios
    df_prices = load_price_data(path)
    metrics = compute_price_metrics(df_prices)

    # --- 1) Evolución de precios ---
    col_chart, col_risk = st.columns([2.2, 1.0])

    with col_chart:
        st.subheader("Evolución histórica de precios")
        st.line_chart(
            df_prices.set_index("Date")["Price"],
            height=300,
        )

    # --- 2) Medidas de riesgo ---
    with col_risk:
        st.subheader("Medidas de riesgo")
        st.write(
            f"Periodo: **{metrics['start'].date()}** a **{metrics['end'].date()}**"
        )
        st.metric("Precio actual", f"{metrics['price_last']:,.2f}")
        st.metric("Volatilidad diaria", f"{metrics['vol_daily']:.2%}")
        st.metric("Volatilidad anualizada", f"{metrics['vol_annual']:.2%}")
        st.write(
            f"- Kurtosis: **{metrics['kurtosis']:.2f}**  \n"
            f"- Skewness: **{metrics['skew']:.2f}**"
        )

    # --- 3) Escenarios e impacto en EERR ---
    st.subheader("Impacto de la volatilidad en el estado de resultados")

    col_inputs, col_table = st.columns([1.1, 2.0])

    with col_inputs:
        st.markdown("**Parámetros del escenario**")
        
        st.markdown(
            """
            **Escenarios sugeridos:**  
            - Pesimista → multiplicador **2.0**  
            - Base → multiplicador **1.0**  
            - Optimista → multiplicador **0.3**
            """
        )
        share_cogs_pct = st.number_input(
            f"% del COGS atribuible a {material}",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            key=f"share_{material}",
        )
        share_cogs = share_cogs_pct / 100

        k_sigma = st.slider(
            f"Multiplicador de σ anual para {material}",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key=f"k_{material}",
        )

        shock_pct = k_sigma * metrics["vol_annual"]

        st.markdown(
            f"""
            - Volatilidad anual estimada: **{metrics['vol_annual']:.2%}**  
            - Multiplicador de σ: **{k_sigma:.1f}**  
            - Shock aplicado al costo atribuible a {material}:  
              **{shock_pct:.2%}** sobre el {share_cogs_pct:.1f}% del COGS.
            """
        )

    scenario_is = simulate_scenario(base_is, share_cogs, shock_pct)
    summary_df = build_summary_table(base_is, scenario_is)

    with col_table:
        st.markdown("**Resumen numérico – Base vs Escenario**")
        st.dataframe(
            summary_df.style.format(
                {
                    "Base": "{:,.0f}",
                    "Escenario": "{:,.0f}",
                    "Δ absoluta": "{:,.0f}",
                    "Δ %": "{:.1%}",
                    "Margen base": "{:.1%}",
                    "Margen esc.": "{:.1%}",
                    "Δ margen (p.p.)": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    # Resumen rápido de ratios
    gp_row = summary_df.loc[summary_df["Métrica"] == "Gross Profit"].iloc[0]
    op_row = summary_df.loc[summary_df["Métrica"] == "Op Income"].iloc[0]
    net_row = summary_df.loc[summary_df["Métrica"] == "Net Income"].iloc[0]

    st.markdown(
        f"""
        **Resumen rápido del cambio en ratios – {material}**

        - Margen bruto: {gp_row["Margen base"]:.1%} → {gp_row["Margen esc."]:.1%}
          ({gp_row["Δ margen (p.p.)"]:+.2f} p.p.)  
        - Margen operativo: {op_row["Margen base"]:.1%} → {op_row["Margen esc."]:.1%}
          ({op_row["Δ margen (p.p.)"]:+.2f} p.p.)  
        - Margen neto: {net_row["Margen base"]:.1%} → {net_row["Margen esc."]:.1%}
          ({net_row["Δ margen (p.p.)"]:+.2f} p.p.)
        """
    )
