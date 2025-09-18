import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import scipy.stats as sps
from statsmodels.api import OLS, add_constant

# ======================================
# Helpers
# ======================================

def safe_std(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else np.nanstd(x, ddof=1)

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else np.nanmean(x)

def safe_div(a, b):
    if b is None or (isinstance(b, float) and np.isnan(b)) or b == 0:
        return np.nan
    return a / b

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna().head(5))
        return True
    except Exception:
        return False

def normalize_cols(cols):
    out = []
    for c in cols:
        s = "" if c is None else str(c)
        s = " ".join(s.strip().split())
        out.append(s)
    return out

def resolve_col(selection: str, columns: list[str]) -> str | None:
    if selection is None:
        return None
    columns_norm = normalize_cols(columns)
    sel_norm = " ".join(str(selection).strip().split())
    if sel_norm in columns_norm:
        return columns[columns_norm.index(sel_norm)]
    sel_cf = sel_norm.casefold()
    for i, c in enumerate(columns_norm):
        if c.casefold() == sel_cf:
            return columns[i]
    sel_compact = "".join(sel_norm.split()).casefold()
    for i, c in enumerate(columns_norm):
        if "".join(c.split()).casefold() == sel_compact:
            return columns[i]
    return None

def clamp_pct(x):
    if x is None or np.isnan(x):
        return np.nan
    return float(np.minimum(1.0, np.maximum(0.0, x)))

def detect_return_scale(r: pd.Series) -> bool:
    r = pd.to_numeric(pd.Series(r).dropna(), errors="coerce")
    if r.empty:
        return False
    med_abs = float(np.median(np.abs(r)))
    max_abs = float(np.max(np.abs(r)))
    if (0.5 <= med_abs <= 100) or (2 < max_abs < 200):
        return True
    return False

def reset_with_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly create a 'Date' column after reset_index().
    - Pick the first datetime-like column if present
    - Else coerce the first column to datetime
    """
    tmp = frame.reset_index()
    # Identify datetime-like columns
    dt_cols = [c for c in tmp.columns if pd.api.types.is_datetime64_any_dtype(tmp[c])]
    if dt_cols:
        dcol = dt_cols[0]
    else:
        # Use the first column as date carrier
        dcol = tmp.columns[0]
        tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
    # Rename selected date column to 'Date'
    if dcol != "Date":
        tmp = tmp.rename(columns={dcol: "Date"})
    return tmp

# ======================================
# Black–Scholes
# ======================================
def black_scholes(S, K, r, sigma, T, option_type="call"):
    try:
        S, K, r, sigma, T = map(float, [S, K, r, sigma, T])
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return (np.nan,)*6
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * sps.norm.cdf(d1) - K * np.exp(-r * T) * sps.norm.cdf(d2)
            rho =  K * T * np.exp(-r * T) * sps.norm.cdf(d2)
            delta = sps.norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * sps.norm.cdf(-d2) - S * sps.norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * sps.norm.cdf(-d2)
            delta = -sps.norm.cdf(-d1)
        gamma = sps.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * sps.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * (sps.norm.cdf(d2) if option_type=="call" else sps.norm.cdf(-d2)))
        vega = S * sps.norm.pdf(d1) * np.sqrt(T)
        return price, delta, gamma, theta, vega, rho
    except Exception:
        return (np.nan,)*6

# ======================================
# VaR helpers (percent; horizon-aware)
# ======================================
def _rolling_compounded(returns: pd.Series, window_days: int) -> np.ndarray:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return np.array([])
    if window_days <= 1:
        return r.values
    comp = (1.0 + r).rolling(window_days).apply(np.prod, raw=True) - 1.0
    return comp.dropna().values

def hist_var_pct(returns, conf=0.95, horizon_days=1):
    horizon_rets = _rolling_compounded(pd.Series(returns), max(1, int(horizon_days)))
    if horizon_rets.size == 0:
        return np.nan
    q_left = np.percentile(horizon_rets, (1 - conf) * 100.0)
    return max(0.0, -q_left)

def parametric_var_pct_lognormal(returns, conf=0.95, horizon_days=1):
    r = pd.Series(returns).dropna().astype(float).values
    if r.size == 0:
        return np.nan
    log_r = np.log1p(r)
    mu_log_d = np.nanmean(log_r)
    sig_log_d = np.nanstd(log_r, ddof=1)
    H = max(1, int(horizon_days))
    z = sps.norm.ppf(1 - conf)  # left tail
    q = np.exp(mu_log_d * H + sig_log_d * np.sqrt(H) * z) - 1.0
    return max(0.0, -q)

def mc_var_pct_from_returns(returns, conf=0.95, horizon_days=1, n=10000, seed=None):
    r = pd.Series(returns).dropna().astype(float).values
    if r.size == 0 or n < 100:
        return np.nan
    if seed is not None:
        np.random.seed(int(seed))
    log_r = np.log1p(r)
    mu_log_d = np.nanmean(log_r)
    sig_log_d = np.nanstd(log_r, ddof=1)
    H = max(1, int(horizon_days))
    sims = np.exp(mu_log_d * H + sig_log_d * np.sqrt(H) * np.random.randn(int(n))) - 1.0
    q_left = np.percentile(sims, (1 - conf) * 100.0)
    return max(0.0, -q_left)

# ======================================
# Streamlit App
# ======================================
st.set_page_config(page_title="Risk & Portfolio Dashboard", layout="wide")
st.title("📊 Risk & Portfolio Dashboard")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Upload Excel (any stock + optional benchmark)", type=["xlsx"])
if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.sidebar.selectbox("Worksheet", options=xls.sheet_names, index=0)
        raw = pd.read_excel(xls, sheet_name=sheet)
    except Exception:
        raw = pd.read_excel(uploaded_file)  # fallback single-sheet
    default_name_asset = uploaded_file.name.split(".")[0]
else:
    raw = pd.read_excel("icici dashboard data.xlsx")
    default_name_asset = "ICICI"

# Normalize column names for consistent matching
raw.columns = normalize_cols(raw.columns)
raw_cols = list(raw.columns)

# ---------- Column mapping ----------
date_candidates = [c for c in raw_cols if "date" in c.lower()]
date_col_guess = date_candidates[0] if date_candidates else raw_cols[0]
numeric_cols = [c for c in raw_cols if is_numeric_series(raw[c])]

if not numeric_cols:
    st.error("No numeric columns detected. Please upload a sheet with price/return numerics.")
    st.stop()

st.sidebar.header("🔧 Column Mapping")
date_sel = st.sidebar.selectbox("Date column", options=raw_cols,
                                index=raw_cols.index(date_col_guess) if date_col_guess in raw_cols else 0)
asset_price_sel = st.sidebar.selectbox("Asset Price column", options=numeric_cols, index=0)
bench_price_sel = st.sidebar.selectbox("Benchmark Price column (optional)", options=["<None>"] + numeric_cols, index=0)
asset_ret_sel = st.sidebar.selectbox("Asset Return column (optional)", options=["<None>"] + numeric_cols, index=0)
bench_ret_sel = st.sidebar.selectbox("Benchmark Return column (optional)", options=["<None>"] + numeric_cols, index=0)

asset_name = st.sidebar.text_input("Display name: Asset", value=default_name_asset)
bench_name_default = "Benchmark" if bench_price_sel != "<None>" else ""
bench_name = st.sidebar.text_input("Display name: Benchmark", value=bench_name_default)

# Resolve selections to actual columns
date_col = resolve_col(date_sel, raw_cols)
asset_price_col = resolve_col(asset_price_sel, raw_cols)
benchmark_price_col = None if bench_price_sel == "<None>" else resolve_col(bench_price_sel, raw_cols)
asset_ret_col = None if asset_ret_sel == "<None>" else resolve_col(asset_ret_sel, raw_cols)
benchmark_ret_col = None if bench_ret_sel == "<None>" else resolve_col(bench_ret_sel, raw_cols)

missing = []
if date_col is None: missing.append("Date")
if asset_price_col is None: missing.append("Asset Price")
if missing:
    st.error(f"Could not resolve required column(s): {', '.join(missing)}. Please re-map in the sidebar.")
    st.write("Columns available:", raw_cols)
    st.stop()
if bench_price_sel != "<None>" and benchmark_price_col is None:
    st.error("Benchmark Price column could not be resolved. Please re-map.")
    st.write("Columns available:", raw_cols)
    st.stop()
if (asset_ret_sel not in (None, "<None>")) and asset_ret_col is None:
    st.error("Asset Return column could not be resolved. Please re-map.")
    st.write("Columns available:", raw_cols)
    st.stop()
if (bench_ret_sel not in (None, "<None>")) and benchmark_ret_col is None:
    st.error("Benchmark Return column could not be resolved. Please re-map.")
    st.write("Columns available:", raw_cols)
    st.stop()

# ---------- Build indexed frame ----------
df = raw.copy()
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True).set_index(date_col)

# ---------- Filters & Params ----------
st.sidebar.header("Filters")
min_d, max_d = df.index.min(), df.index.max()
date_range = st.sidebar.date_input("Select Date Range", [min_d, max_d])
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df.loc[start_d:end_d]
else:
    st.warning("Invalid date range; using full data.")
if df.empty:
    st.warning("No data in the selected range. Adjust filters.")
    st.stop()

st.sidebar.subheader("Parameters")
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100.0
volatility_ui = st.sidebar.slider("Volatility (%)", 0.0, 100.0, 30.0) / 100.0
time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0)
conf = st.sidebar.select_slider("VaR Confidence Level", options=[0.90, 0.95, 0.99], value=0.95,
                                format_func=lambda x: f"{int(x*100)}%")

# ---------- Normalized working frame ----------
ndf = pd.DataFrame(index=df.index)

# Asset price/returns
ndf["Asset_Price"] = pd.to_numeric(df[asset_price_col], errors="coerce")
if asset_ret_col is not None:
    ndf["Asset_Return"] = pd.to_numeric(df[asset_ret_col], errors="coerce")
else:
    ndf["Asset_Return"] = ndf["Asset_Price"].pct_change()

# Auto-scale returns and override
auto_pct = detect_return_scale(ndf["Asset_Return"])
with st.sidebar.expander("Return Scaling (if uploaded returns are in % units)"):
    st.write("If your return column is e.g., **1.2 = 1.2%**, enable scaling to convert to decimals.")
    force_pct = st.checkbox("Treat Asset Return column as % (divide by 100)", value=auto_pct, key="asset_pct_flag")
if st.session_state.get("asset_pct_flag", False):
    ndf["Asset_Return"] = ndf["Asset_Return"] / 100.0

# Benchmark optional
if benchmark_price_col is not None:
    ndf["Bench_Price"] = pd.to_numeric(df[benchmark_price_col], errors="coerce")
    if benchmark_ret_col is not None:
        ndf["Bench_Return"] = pd.to_numeric(df[benchmark_ret_col], errors="coerce")
    else:
        ndf["Bench_Return"] = ndf["Bench_Price"].pct_change()
    auto_pct_b = detect_return_scale(ndf["Bench_Return"])
    with st.sidebar.expander("Benchmark Return Scaling"):
        force_pct_b = st.checkbox("Treat Benchmark Return column as % (divide by 100)", value=auto_pct_b, key="bench_pct_flag")
    if st.session_state.get("bench_pct_flag", False):
        ndf["Bench_Return"] = ndf["Bench_Return"] / 100.0
else:
    ndf["Bench_Price"] = np.nan
    ndf["Bench_Return"] = np.nan

# ======================================
# 1) Performance
# ======================================
st.header("1️⃣ Performance Analysis")

df_reset = reset_with_date_index(ndf)

price_cols = ["Asset_Price"] + (["Bench_Price"] if ndf["Bench_Price"].notna().any() else [])
rename_map = {"Asset_Price": asset_name}
if "Bench_Price" in price_cols:
    rename_map["Bench_Price"] = bench_name or "Benchmark"

fig1 = px.line(
    df_reset,
    x="Date",
    y=price_cols,
    title=f"{asset_name}" + (f" vs {bench_name}" if "Bench_Price" in price_cols else "") + " — Prices"
)
# Apply display names
for tr in fig1.data:
    tr.name = rename_map.get(tr.name, tr.name)
st.plotly_chart(fig1, width='stretch')

cum_df = pd.DataFrame({"Date": df_reset["Date"]})
cum_df[f"{asset_name}_CumRet"] = (1 + ndf["Asset_Return"].fillna(0)).cumprod().values - 1
if ndf["Bench_Return"].notna().any():
    cum_df[f"{bench_name or 'Benchmark'}_CumRet"] = (1 + ndf["Bench_Return"].fillna(0)).cumprod().values - 1

fig_cum = px.line(
    cum_df,
    x="Date",
    y=[c for c in cum_df.columns if c.endswith("_CumRet")],
    title="Cumulative Returns"
)
st.plotly_chart(fig_cum, width='stretch')

stats_cols = ["Asset_Return"] + (["Bench_Return"] if ndf["Bench_Return"].notna().any() else [])
stats_tbl = ndf[stats_cols].agg(["mean", "var", "std"]).rename(
    columns={"Asset_Return": asset_name, "Bench_Return": bench_name or "Benchmark"}
)
st.write("**Return Stats (daily):**")
st.dataframe(stats_tbl, use_container_width=True)  # dataframe supports use_container_width

diag = pd.DataFrame({
    "Metric": ["Mean (daily)", "Std (daily)", "Min", "Max"],
    "Value (%)": [
        safe_mean(ndf["Asset_Return"]) * 100.0,
        safe_std(ndf["Asset_Return"]) * 100.0,
        pd.Series(ndf["Asset_Return"]).min() * 100.0,
        pd.Series(ndf["Asset_Return"]).max() * 100.0,
    ]
})
st.caption("Return diagnostics (after scaling):")
st.dataframe(diag.style.format({"Value (%)": "{:.4f}"}), use_container_width=True)

# ======================================
# 2) Risk–Return
# ======================================
st.header("2️⃣ Risk-Return Analysis")

rf_daily = risk_free / 252.0
excess = (ndf["Asset_Return"] - rf_daily).dropna().values
downside = (ndf["Asset_Return"][ndf["Asset_Return"] < 0] - rf_daily).dropna().values
sharpe = np.sqrt(252.0) * safe_div(safe_mean(excess), safe_std(excess))
sortino = np.sqrt(252.0) * safe_div(safe_mean(excess), safe_std(downside))
st.write(f"**Sharpe Ratio ({asset_name}):** {sharpe:.3f} | **Sortino Ratio:** {sortino:.3f}")

if ndf["Bench_Return"].notna().sum() > 5:
    reg_df = ndf[["Asset_Return", "Bench_Return"]].dropna()
    if not reg_df.empty and reg_df["Bench_Return"].nunique() > 1:
        X = add_constant(reg_df["Bench_Return"].values.astype(float))
        y = reg_df["Asset_Return"].values.astype(float)
        try:
            model = OLS(y, X).fit()
            alpha, beta = float(model.params[0]), float(model.params[1])
            st.write(f"**Alpha ({asset_name} vs {bench_name or 'Benchmark'}):** {alpha:.6f} | **Beta:** {beta:.4f}")
            fig2 = px.scatter(
                reg_df.reset_index(),
                x="Bench_Return",
                y="Asset_Return",
                trendline="ols",
                title=f"Regression: {asset_name} vs {bench_name or 'Benchmark'} (daily returns)"
            )
            st.plotly_chart(fig2, width='stretch')
        except Exception as e:
            st.warning(f"Regression failed: {e}")
    else:
        st.info("Insufficient variability in benchmark returns for regression.")
else:
    st.info("No benchmark selected/provided → skipping Alpha/Beta.")

# ======================================
# 3) VaR — Percent Loss (no amounts)
# ======================================
st.header("3️⃣ Value at Risk (VaR) — Percent Loss")

h_days = max(1, int(round(252 * float(time_horizon))))
hist_var_percent = hist_var_pct(ndf["Asset_Return"], conf=conf, horizon_days=h_days)
para_var_percent = parametric_var_pct_lognormal(ndf["Asset_Return"], conf=conf, horizon_days=h_days)
mc_var_percent   = mc_var_pct_from_returns(ndf["Asset_Return"], conf=conf, horizon_days=h_days, n=10000)

hist_var_percent = clamp_pct(hist_var_percent)
para_var_percent = clamp_pct(para_var_percent)
mc_var_percent   = clamp_pct(mc_var_percent)

st.caption(f"Left-tail quantile = (1 − Confidence). Holding period: {h_days} day(s). Confidence: {int(conf*100)}%")
st.write(f"**Historical VaR (loss):** {hist_var_percent:.2%}")
st.write(f"**Parametric VaR (loss) [lognormal]:** {para_var_percent:.2%}")
st.write(f"**Monte Carlo VaR (loss):** {mc_var_percent:.2%}")

# ======================================
# 4) Black–Scholes Calculation
# ======================================
st.header("4️⃣ Black–Scholes Calculation")
col1, col2 = st.columns(2)
with col1:
    last_price = ndf["Asset_Price"].dropna().iloc[-1] if ndf["Asset_Price"].dropna().size else 100.0
    S = st.number_input("Spot Price", min_value=0.0, value=float(last_price))
    K = st.number_input("Strike Price", min_value=0.0, value=float(last_price))
with col2:
    sigma_ui = st.number_input("Volatility (σ, decimal)", min_value=0.0, value=0.2)
    T = st.number_input("Time to Maturity (Years)", min_value=0.0, value=1.0)
opt_type = st.selectbox("Option Type", ["call", "put"])
price, delta, gamma, theta, vega, rho = black_scholes(S, K, risk_free, sigma_ui, T, option_type=opt_type)
st.write(f"**Price:** {price:.4f} | **Delta:** {delta:.4f} | **Gamma:** {gamma:.6f} | "
         f"**Theta:** {theta:.4f} | **Vega:** {vega:.4f} | **Rho:** {rho:.4f}")

# ======================================
# 5) Asset Liability Management (ALM)
# ======================================
st.header("5️⃣ Asset Liability Management (ALM)")

tab_direct, tab_csv = st.tabs(["🧮 Direct Input (Equity Shock)", "📤 Upload CSV (RSA/RSL + Auto-Durations)"])

with tab_direct:
    colA, colB, colC = st.columns(3)
    with colA:
        A = st.number_input("Total Rate-Sensitive Assets A", min_value=0.0,
                            value=st.session_state.get("alm_A", 1_000_000_000.0),
                            step=1_000_000.0, format="%.2f", key="alm_A_input")
        DA = st.number_input("Duration of Assets DA (years)", min_value=0.0,
                             value=st.session_state.get("alm_DA", 2.0),
                             step=0.1, format="%.2f", key="alm_DA_input")
        CA = st.number_input("Convexity of Assets CA (optional)", min_value=0.0,
                             value=st.session_state.get("alm_CA", 0.0),
                             step=0.1, format="%.4f", key="alm_CA_input")
    with colB:
        L = st.number_input("Total Rate-Sensitive Liabilities L", min_value=0.0,
                            value=st.session_state.get("alm_L", 900_000_000.0),
                            step=1_000_000.0, format="%.2f", key="alm_L_input")
        DL = st.number_input("Duration of Liabilities DL (years)", min_value=0.0,
                             value=st.session_state.get("alm_DL", 1.5),
                             step=0.1, format="%.2f", key="alm_DL_input")
        CL = st.number_input("Convexity of Liabilities CL (optional)", min_value=0.0,
                             value=st.session_state.get("alm_CL", 0.0),
                             step=0.1, format="%.4f", key="alm_CL_input")
    with colC:
        shock_mode = st.radio("Shock Input", ["Basis Points (bps)", "Percent (%)"], horizontal=True, key="alm_mode")
        shock_val = st.number_input("Parallel Yield Shock", value=100.0, step=25.0, format="%.2f", key="alm_shock")
        dy = shock_val/10000.0 if shock_mode == "Basis Points (bps)" else shock_val/100.0

    E = A - L
    A_safe = A if A > 0 else np.nan
    L_over_A = (L / A_safe) if (A_safe and not np.isnan(A_safe)) else np.nan
    DG = DA - (DL * L_over_A) if not np.isnan(L_over_A) else np.nan

    dE_linear = - DG * A * dy if not (np.isnan(DG) or np.isnan(dy)) else np.nan
    dE_conv = 0.5 * ((CA * A) - (CL * L)) * (dy**2) if (CA > 0 or CL > 0) else 0.0
    dE_total = (0.0 if np.isnan(dE_linear) else dE_linear) + dE_conv
    shock_pct = (dE_total / E * 100.0) if E != 0 else np.nan

    st.markdown("DG = DA − DL × (L/A);  ΔE ≈ − DG × A × Δy  (+ convexity: 0.5 × (CA×A − CL×L) × Δy²)")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Equity E = A − L", f"{E:,.2f}"); st.metric("DA (years)", f"{DA:.2f}")
    with c2: st.metric("DL (years)", f"{DL:.2f}");       st.metric("Duration Gap (DG)", f"{DG:.4f}" if not np.isnan(DG) else "—")
    with c3: st.metric("Δy", f"{dy:.4%}");               st.metric("Equity Shock (%)", f"{(shock_pct if not np.isnan(shock_pct) else 0):.2f}%")
    st.write(f"**ΔE (amount):** {dE_total:,.2f} | **Linear:** {0.0 if np.isnan(dE_linear) else dE_linear:,.2f} | **Convexity:** {dE_conv:,.2f}")

    st.subheader("ΔEVE Sensitivity: Equity Shock vs Yield Shift")
    step_bps = st.select_slider("Shock grid resolution (bps)", options=[10, 25, 50, 100], value=25)
    shocks_bps = np.arange(-300, 300 + step_bps, step_bps, dtype=int)
    dy_vec = shocks_bps / 10000.0
    if not np.isnan(DG) and E != 0 and not np.isnan(E):
        dE_linear_vec = -DG * A * dy_vec
        dE_conv_vec = 0.5 * ((CA * A) - (CL * L)) * (dy_vec ** 2) if (CA > 0 or CL > 0) else 0.0
        dE_total_vec = dE_linear_vec + dE_conv_vec
        eq_shock_pct_vec = (dE_total_vec / E) * 100.0
        sens_df = pd.DataFrame({
            "Shock_bps": shocks_bps,
            "Delta_y": dy_vec,
            "DeltaE_amount": dE_total_vec,
            "EquityShock_pct": eq_shock_pct_vec
        })
        st.dataframe(sens_df.style.format({"Delta_y": "{:.4%}", "DeltaE_amount": "{:,.2f}", "EquityShock_pct": "{:.2f}"}),
                     use_container_width=True)
        fig_sens = px.line(sens_df, x="Shock_bps", y="EquityShock_pct",
                           title="Equity Shock % vs Parallel Yield Shift (bps)")
        fig_sens.update_traces(mode="lines+markers")
        st.plotly_chart(fig_sens, width='stretch')
    else:
        st.info("Provide valid A, L, DA, DL (and ensure A ≠ 0) to run the sensitivity sweep.")

with tab_csv:
    st.caption("Minimum columns: **Type** ∈ {Asset, Liability}, **Amount** (numeric). Optional: **Midpoint_Years** for duration.")
    alm_file = st.file_uploader("Upload CSV", type=["csv"], key="alm_csv")
    if alm_file:
        alm_df = pd.read_csv(alm_file)
        st.dataframe(alm_df, use_container_width=True)
        cols_norm = {c.strip().lower(): c for c in alm_df.columns}
        if "type" in cols_norm and "amount" in cols_norm:
            type_col = cols_norm["type"]; amt_col = cols_norm["amount"]
            assets = pd.to_numeric(alm_df.loc[alm_df[type_col].str.lower()=="asset", amt_col], errors="coerce")
            liabs  = pd.to_numeric(alm_df.loc[alm_df[type_col].str.lower()=="liability", amt_col], errors="coerce")
            A_csv = float(assets.sum(skipna=True)) if not assets.empty else np.nan
            L_csv = float(liabs.sum(skipna=True)) if not liabs.empty else np.nan
            DA_csv = DL_csv = np.nan
            if "midpoint_years" in cols_norm:
                t_col = cols_norm["midpoint_years"]
                a_df = alm_df.loc[alm_df[type_col].str.lower()=="asset", [t_col, amt_col]].copy()
                a_df[amt_col] = pd.to_numeric(a_df[amt_col], errors="coerce")
                a_df[t_col]   = pd.to_numeric(a_df[t_col], errors="coerce")
                if a_df[amt_col].sum(skipna=True) > 0:
                    DA_csv = float((a_df[t_col] * a_df[amt_col]).sum(skipna=True) / a_df[amt_col].sum(skipna=True))
                l_df = alm_df.loc[alm_df[type_col].str.lower()=="liability", [t_col, amt_col]].copy()
                l_df[amt_col] = pd.to_numeric(l_df[amt_col], errors="coerce")
                l_df[t_col]   = pd.to_numeric(l_df[t_col], errors="coerce")
                if l_df[amt_col].sum(skipna=True) > 0:
                    DL_csv = float((l_df[t_col] * l_df[amt_col]).sum(skipna=True) / l_df[amt_col].sum(skipna=True))
            st.write(f"**RSA (A):** {A_csv:,.2f} | **RSL (L):** {L_csv:,.2f}")
            st.write(f"**Estimated DA (years):** {DA_csv:.4f}" if not np.isnan(DA_csv) else "**Estimated DA:** —")
            st.write(f"**Estimated DL (years):** {DL_csv:.4f}" if not np.isnan(DL_csv) else "**Estimated DL:** —")
            if st.button("Use these in Direct Input", type="primary"):
                if not np.isnan(A_csv): st.session_state["alm_A"] = A_csv
                if not np.isnan(L_csv): st.session_state["alm_L"] = L_csv
                if not np.isnan(DA_csv): st.session_state["alm_DA"] = DA_csv
                if not np.isnan(DL_csv): st.session_state["alm_DL"] = DL_csv
                st.success("Loaded into Direct Input. Go to the '🧮 Direct Input (Equity Shock)' tab.")
        else:
            st.info("CSV must include columns: Type, Amount. (Optional: Midpoint_Years).")

# ======================================
# 6) Portfolio Simulation (MC, consistent with VaR)
# ======================================
st.header("6️⃣ Portfolio Simulation")

log_r = np.log1p(ndf["Asset_Return"].dropna().values.astype(float))
if log_r.size < 5 or np.isnan(log_r).all():
    st.info("Insufficient data to simulate returns.")
else:
    n_sims  = st.slider("Number of simulations", 1000, 100_000, 10_000, step=1000)
    steps_y = int(np.ceil(252 * time_horizon))
    seed_on = st.checkbox("Set random seed (reproducible)", value=False)
    if seed_on:
        seed_val = st.number_input("Seed", min_value=0, value=42, step=1)
        np.random.seed(int(seed_val))

    mu_log_d = float(np.nanmean(log_r))
    sig_log_d = float(np.nanstd(log_r, ddof=1))
    term_returns = np.exp(mu_log_d * steps_y + sig_log_d * np.sqrt(steps_y) * np.random.randn(n_sims)) - 1.0

    var_left = np.percentile(term_returns, (1 - conf) * 100.0)
    prob_loss = float((term_returns < 0).mean())
    mean_ret  = float(np.mean(term_returns))
    p5, p50, p95 = np.percentile(term_returns, [5, 50, 95])

    sim_df = pd.DataFrame({"Terminal Return": term_returns})
    fig3 = px.histogram(
        sim_df, x="Terminal Return", nbins=60,
        title=f"Monte Carlo Terminal Return Distribution (Horizon = {time_horizon:.2f} yrs, {n_sims} paths)"
    )
    fig3.add_vline(x=float(var_left), line_dash="dash", line_color="red",
                   annotation_text=f"Left-tail {(1 - conf):.0%}", annotation_position="top right")
    st.plotly_chart(fig3, width='stretch')

    st.caption(f"Monte Carlo left-tail uses (1 − Confidence). Confidence: {int(conf*100)}% → Left tail: {(1-conf):.0%}")
    st.write(f"**Probability of Loss over horizon:** {prob_loss:.2%}")
    st.write(f"**Mean Terminal Return:** {mean_ret:.2%} | **Median:** {p50:.2%} | **5th pct:** {p5:.2%} | **95th pct:** {p95:.2%}")

# ======================================
# 7) Download
# ======================================
st.header("7️⃣ Download Options")
output = io.BytesIO()
export_df = ndf.copy()
export_df.reset_index().rename(columns={ndf.index.name: "Date"}).to_excel(output, index=False)
st.download_button(
    "Download Processed Data (Excel)",
    data=output.getvalue(),
    file_name="processed_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
