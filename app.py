import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import norm
from statsmodels.api import OLS, add_constant

# ----------------------------
# Utils: safe stats
# ----------------------------
def safe_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return np.nanstd(x, ddof=1)

def safe_mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return np.nanmean(x)

def safe_div(a, b):
    if b is None or np.isnan(b) or b == 0:
        return np.nan
    return a / b

# ----------------------------
# Black-Scholes Option Pricing
# ----------------------------
def black_scholes(S, K, r, sigma, T, option_type="call"):
    try:
        S, K, r, sigma, T = map(float, [S, K, r, sigma, T])
        if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
            return (np.nan,)*6
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            rho =  K * T * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            delta = -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type=="call" else norm.cdf(-d2)))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return price, delta, gamma, theta, vega, rho
    except Exception:
        return (np.nan,)*6

# ----------------------------
# VaR Computations
# ----------------------------
def historical_var(returns, conf=0.95):
    r = pd.Series(returns).dropna().values
    if r.size == 0:
        return np.nan
    return np.percentile(r, (1 - conf) * 100)

def parametric_var(returns, conf=0.95):
    r = pd.Series(returns).dropna().values
    if r.size == 0:
        return np.nan
    mu, sigma = r.mean(), r.std(ddof=1)
    if np.isnan(mu) or np.isnan(sigma):
        return np.nan
    return mu - sigma * norm.ppf(conf)

def monte_carlo_var(S0, mu, sigma, T, n=10000, conf=0.95):
    """One-step lognormal approximation for VaR section (kept).
       Simulation section uses full GBM paths with compounding."""
    try:
        S0 = float(S0); mu = float(mu); sigma = float(sigma); T = float(T)
        if S0 <= 0 or sigma < 0 or T <= 0 or n <= 10:
            return np.nan
        sims = S0 * np.exp((mu - 0.5 * sigma**2) * T +
                           sigma * np.sqrt(T) * np.random.randn(int(n)))
        rets = (sims - S0) / S0
        if rets.size == 0:
            return np.nan
        return np.percentile(rets, (1 - conf) * 100)
    except Exception:
        return np.nan

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="ICICI Risk Dashboard", layout="wide")
st.title("📊 ICICI Bank Risk & Portfolio Dashboard")

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload ICICI Dashboard Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_excel("icici dashboard data.xlsx")

# Expecting 5 columns: Date, ICICI_Price, ICICI_Return, Nifty_Price, Nifty_Return
if df.shape[1] < 5:
    st.error("Input file must have 5 columns: Date, ICICI_Price, ICICI_Return, Nifty_Price, Nifty_Return")
    st.stop()

df.columns = ["Date", "ICICI_Price", "ICICI_Return", "Nifty_Price", "Nifty_Return"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")

# ---- Sidebar Filters & Parameters ----
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
notional = st.sidebar.number_input("Notional Value", min_value=0.0, value=10000.0, step=100.0)
risk_free = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100.0
volatility_ui = st.sidebar.slider("Volatility (%)", 0.0, 100.0, 30.0) / 100.0
time_horizon = st.sidebar.slider("Time Horizon (Years)", 0.1, 5.0, 1.0)

# Global VaR confidence
conf = st.sidebar.select_slider(
    "VaR Confidence Level",
    options=[0.90, 0.95, 0.99],
    value=0.95,
    format_func=lambda x: f"{int(x*100)}%"
)

# ==========================
# 1. Performance Analysis
# ==========================
st.header("1️⃣ Performance Analysis")

icici_ret = pd.to_numeric(df["ICICI_Return"], errors="coerce") if df["ICICI_Return"].notna().any() \
            else pd.to_numeric(df["ICICI_Price"], errors="coerce").pct_change()
nifty_ret  = pd.to_numeric(df["Nifty_Return"], errors="coerce") if df["Nifty_Return"].notna().any() \
            else pd.to_numeric(df["Nifty_Price"], errors="coerce").pct_change()

df["ICICI_%Change"] = icici_ret
df["Nifty_%Change"] = nifty_ret

df_reset = df.reset_index()
fig1 = px.line(df_reset, x="Date", y=["ICICI_Price", "Nifty_Price"], title="ICICI vs Nifty Prices")
st.plotly_chart(fig1, use_container_width=True)

cum_df = pd.DataFrame({
    "Date": df_reset["Date"],
    "ICICI_CumRet": (1 + df["ICICI_%Change"].fillna(0)).cumprod().values - 1,
    "Nifty_CumRet":  (1 + df["Nifty_%Change"].fillna(0)).cumprod().values - 1
})
fig_cum = px.line(cum_df, x="Date", y=["ICICI_CumRet", "Nifty_CumRet"], title="Cumulative Returns")
st.plotly_chart(fig_cum, use_container_width=True)

stats_tbl = df[["ICICI_%Change", "Nifty_%Change"]].agg(["mean", "var", "std"])
st.write("**Return Stats (daily):**")
st.dataframe(stats_tbl)

# ==========================
# 2. Risk-Return Analysis
# ==========================
st.header("2️⃣ Risk-Return Analysis")

rf_daily = risk_free / 252.0
excess = (df["ICICI_%Change"] - rf_daily).dropna().values
downside = (df["ICICI_%Change"][df["ICICI_%Change"] < 0] - rf_daily).dropna().values

excess_mean = safe_mean(excess)
excess_std = safe_std(excess)
down_std = safe_std(downside)

sharpe = np.sqrt(252.0) * safe_div(excess_mean, excess_std)
sortino = np.sqrt(252.0) * safe_div(excess_mean, down_std)

reg_df = df[["ICICI_%Change", "Nifty_%Change"]].dropna()
alpha = beta = np.nan
if not reg_df.empty and reg_df["Nifty_%Change"].nunique() > 1:
    X = add_constant(reg_df["Nifty_%Change"].values.astype(float))
    y = reg_df["ICICI_%Change"].values.astype(float)
    try:
        model = OLS(y, X).fit()
        params = model.params
        alpha = float(params[0]); beta = float(params[1])
    except Exception as e:
        st.warning(f"Regression failed safely: {e}")

st.write(f"**Sharpe Ratio:** {sharpe:.3f} | **Sortino Ratio:** {sortino:.3f}")
st.write(f"**Alpha:** {alpha:.6f} | **Beta:** {beta:.4f}")

if not reg_df.empty:
    fig2 = px.scatter(reg_df.reset_index(), x="Nifty_%Change", y="ICICI_%Change",
                      trendline="ols", title="Regression: ICICI vs Nifty (daily returns)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Insufficient data for regression plot.")

# ==========================
# 3. Value at Risk
# ==========================
st.header("3️⃣ Value at Risk (VaR)")

hist_var_val = historical_var(df["ICICI_%Change"], conf)
param_var_val = parametric_var(df["ICICI_%Change"], conf)
mc_var_val = monte_carlo_var(
    S0=df["ICICI_Price"].dropna().iloc[-1] if df["ICICI_Price"].dropna().size else np.nan,
    mu=safe_mean(df["ICICI_%Change"]),
    sigma=safe_std(df["ICICI_%Change"]),
    T=time_horizon,
    n=10000,
    conf=conf
)

st.write(f"**Confidence:** {int(conf*100)}%")
st.write(f"**Historical VaR (return):** {hist_var_val:.2%}")
st.write(f"**Parametric VaR (return):** {param_var_val:.2%}")
st.write(f"**Monte Carlo VaR (return over {time_horizon:.2f}y):** {mc_var_val:.2%}")

if notional and not np.isnan(notional):
    st.write(f"**Historical VaR (amt):** {notional * hist_var_val:,.2f}")
    st.write(f"**Parametric VaR (amt):** {notional * param_var_val:,.2f}")
    st.write(f"**Monte Carlo VaR (amt):** {notional * mc_var_val:,.2f}")

# ==========================
# 4. Options & Greeks
# ==========================
st.header("4️⃣ Options & Greeks")
col1, col2 = st.columns(2)
with col1:
    S = st.number_input("Spot Price", min_value=0.0, value=100.0)
    K = st.number_input("Strike Price", min_value=0.0, value=100.0)
with col2:
    sigma_ui = st.number_input("Volatility (σ, decimal)", min_value=0.0, value=0.2)
    T = st.number_input("Time to Maturity (Years)", min_value=0.0, value=1.0)
opt_type = st.selectbox("Option Type", ["call", "put"])

price, delta, gamma, theta, vega, rho = black_scholes(S, K, risk_free, sigma_ui, T, option_type=opt_type)
st.write(
    f"**Price:** {price:.4f} | **Delta:** {delta:.4f} | **Gamma:** {gamma:.6f} | "
    f"**Theta:** {theta:.4f} | **Vega:** {vega:.4f} | **Rho:** {rho:.4f}"
)

# ==========================
# 5. Asset Liability Management (ALM)
# ==========================
st.header("5️⃣ Asset Liability Management (ALM)")

tab_direct, tab_csv = st.tabs(["🧮 Direct Input (Equity Shock)", "📤 Upload CSV (RSA/RSL + Auto-Durations)"])

with tab_direct:
    colA, colB, colC = st.columns(3)

    with colA:
        A = st.number_input("Total Rate-Sensitive Assets A",
                            min_value=0.0, value=st.session_state.get("alm_A", 1_000_000_000.0),
                            step=1_000_000.0, format="%.2f", key="alm_A_input")
        DA = st.number_input("Duration of Assets DA (years)",
                             min_value=0.0, value=st.session_state.get("alm_DA", 2.0),
                             step=0.1, format="%.2f", key="alm_DA_input")
        CA = st.number_input("Convexity of Assets CA (optional)",
                             min_value=0.0, value=st.session_state.get("alm_CA", 0.0),
                             step=0.1, format="%.4f", key="alm_CA_input")

    with colB:
        L = st.number_input("Total Rate-Sensitive Liabilities L",
                            min_value=0.0, value=st.session_state.get("alm_L", 900_000_000.0),
                            step=1_000_000.0, format="%.2f", key="alm_L_input")
        DL = st.number_input("Duration of Liabilities DL (years)",
                             min_value=0.0, value=st.session_state.get("alm_DL", 1.5),
                             step=0.1, format="%.2f", key="alm_DL_input")
        CL = st.number_input("Convexity of Liabilities CL (optional)",
                             min_value=0.0, value=st.session_state.get("alm_CL", 0.0),
                             step=0.1, format="%.4f", key="alm_CL_input")

    with colC:
        shock_mode = st.radio("Shock Input", ["Basis Points (bps)", "Percent (%)"],
                              horizontal=True, key="alm_mode")
        shock_val = st.number_input("Parallel Yield Shock", value=100.0, step=25.0,
                                    format="%.2f", key="alm_shock")
        dy = shock_val/10000.0 if shock_mode == "Basis Points (bps)" else shock_val/100.0

    # Core ALM math
    E = A - L
    A_safe = A if A > 0 else np.nan
    L_over_A = (L / A_safe) if (A_safe and not np.isnan(A_safe)) else np.nan
    DG = DA - (DL * L_over_A) if not np.isnan(L_over_A) else np.nan

    dE_linear = - DG * A * dy if not (np.isnan(DG) or np.isnan(dy)) else np.nan
    dE_conv = 0.5 * ((CA * A) - (CL * L)) * (dy**2) if (CA > 0 or CL > 0) else 0.0
    dE_total = (0.0 if np.isnan(dE_linear) else dE_linear) + dE_conv
    shock_pct = (dE_total / E * 100.0) if E != 0 else np.nan

    st.markdown(
        "**Formulas**  \n"
        "Duration Gap: **DG = DA − DL × (L/A)**  \n"
        "ΔE (linear): **− DG × A × Δy**  \n"
        "ΔE (with convexity): **− DG × A × Δy + 0.5 × (CA×A − CL×L) × Δy²**"
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Equity E = A − L", f"{E:,.2f}")
        st.metric("Duration of Assets (DA)", f"{DA:.2f} years")
    with m2:
        st.metric("Duration of Liabs (DL)", f"{DL:.2f} years")
        st.metric("Duration Gap (DG)", f"{DG:.4f}" if not np.isnan(DG) else "—")
    with m3:
        st.metric("Yield Shock Δy", f"{dy:.4%}")
        st.metric("Equity Shock (%)", f"{shock_pct:.2f}%" if not np.isnan(shock_pct) else "—")

    st.write(
        f"**ΔE (amount):** {dE_total:,.2f}  |  "
        f"**Linear component:** {0.0 if np.isnan(dE_linear) else dE_linear:,.2f}  |  "
        f"**Convexity add-on:** {dE_conv:,.2f}"
    )

    # -------- ΔEVE sweep: Equity Shock % vs Δy ----------
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

        st.dataframe(sens_df.style.format({
            "Delta_y": "{:.4%}",
            "DeltaE_amount": "{:,.2f}",
            "EquityShock_pct": "{:.2f}"
        }), use_container_width=True)

        fig_sens = px.line(
            sens_df, x="Shock_bps", y="EquityShock_pct",
            title="Equity Shock % vs Parallel Yield Shift (bps)"
        )
        fig_sens.update_traces(mode="lines+markers")
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("Provide valid A, L, DA, DL (and ensure A ≠ 0) to run the sensitivity sweep.")

with tab_csv:
    st.caption("Minimum columns: **Type** ∈ {Asset, Liability}, **Amount** (numeric). Optional: **Midpoint_Years** for duration.")
    alm_file = st.file_uploader("Upload CSV", type=["csv"], key="alm_csv")
    if alm_file:
        alm_df = pd.read_csv(alm_file)
        st.dataframe(alm_df)
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
                if not a_df[amt_col].dropna().empty and a_df[amt_col].sum(skipna=True) > 0:
                    DA_csv = float((a_df[t_col] * a_df[amt_col]).sum(skipna=True) / a_df[amt_col].sum(skipna=True))

                l_df = alm_df.loc[alm_df[type_col].str.lower()=="liability", [t_col, amt_col]].copy()
                l_df[amt_col] = pd.to_numeric(l_df[amt_col], errors="coerce")
                l_df[t_col]   = pd.to_numeric(l_df[t_col], errors="coerce")
                if not l_df[amt_col].dropna().empty and l_df[amt_col].sum(skipna=True) > 0:
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

# ==========================
# 6. Portfolio Simulation (GBM, horizon-aware)
# ==========================
st.header("6️⃣ Portfolio Simulation")

mu_daily  = safe_mean(df["ICICI_%Change"])
sig_daily = safe_std(df["ICICI_%Change"])

if np.isnan(mu_daily) or np.isnan(sig_daily) or sig_daily < 0:
    st.info("Insufficient data to simulate returns.")
else:
    n_sims  = st.slider("Number of simulations", 1000, 100_000, 10_000, step=1000)
    steps_y = int(np.ceil(252 * time_horizon))
    seed_on = st.checkbox("Set random seed (reproducible)", value=False)
    if seed_on:
        seed_val = st.number_input("Seed", min_value=0, value=42, step=1)
        np.random.seed(int(seed_val))

    dt = 1.0 / 252.0
    drift = (mu_daily - 0.5 * (sig_daily ** 2)) * dt
    diff  = sig_daily * np.sqrt(dt)

    Z = np.random.randn(n_sims, steps_y)
    log_growth = drift * steps_y + diff * Z.sum(axis=1)
    term_returns = np.exp(log_growth) - 1.0

    prob_loss = float((term_returns < 0).mean())
    mean_ret  = float(np.mean(term_returns))
    p5, p50, p95 = np.percentile(term_returns, [5, 50, 95])
    var_horizon = np.percentile(term_returns, (1 - conf) * 100)

    sim_df = pd.DataFrame({"Terminal Return": term_returns})
    fig3 = px.histogram(sim_df, x="Terminal Return", nbins=60,
                        title=f"Monte Carlo Terminal Return Distribution (Horizon = {time_horizon:.2f} years, {n_sims} paths)")
    fig3.add_vline(x=float(var_horizon), line_dash="dash", line_color="red",
                   annotation_text=f"VaR {int(conf*100)}%", annotation_position="top right")
    st.plotly_chart(fig3, use_container_width=True)

    st.write(f"**Confidence:** {int(conf*100)}%")
    st.write(f"**Probability of Loss over horizon:** {prob_loss:.2%}")
    st.write(f"**Mean Terminal Return:** {mean_ret:.2%} | **Median:** {p50:.2%} | **5th pct:** {p5:.2%} | **95th pct:** {p95:.2%}")
    if notional and not np.isnan(notional):
        st.write(f"**VaR (amount, horizon):** {notional * var_horizon:,.2f}")
        st.write(f"**Mean P&L (amount, horizon):** {notional * mean_ret:,.2f}")

# ==========================
# 7. Download Results
# ==========================
st.header("7️⃣ Download Options")
output = io.BytesIO()
df_export = df.copy()
df_export.reset_index().to_excel(output, index=False)
st.download_button(
    "Download Processed Data (Excel)",
    data=output.getvalue(),
    file_name="processed_icici.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
