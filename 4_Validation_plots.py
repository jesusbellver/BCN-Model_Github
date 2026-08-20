"""
Statistical validation plots for mosquito model - ZONE-MONTH aggregation.

Runs REALISTIC scenario (ASPB actual interventions + actual reform dates)
Supports two modes:
- "cases": Compare real_cases vs pred_cases (what is optimized in fitting)
- "prevalence": Compare real_prevalence vs pred_prevalence

"""

import numpy as np
import pandas as pd
import xarray as xr
import pickle
import matplotlib
from collections import OrderedDict
from numba import jit
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy import odr
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Dades_netes"
PARAM_DIR = SCRIPT_DIR / "Parameters"
FIGURES_DIR = SCRIPT_DIR / "Figures" / "Validation_Plots"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

d0 = datetime(2019, 1, 1)

# ============================================================================
# MODE SELECTION
# ============================================================================

MSE_MODE = "prevalence"  # "cases" or "prevalence"
WEIGHTING_MODE = "weighted"  # "weighted" or "unweighted"
SAVE_FIGURES = False  # True to save figures, False to only show them

# ============================================================================
# ZONE COLORS
# ============================================================================

zone_colors = OrderedDict([
    ("Parc de la Ciutadella", "#FDB462"),
    ("Parc del Turó de la Peira", "#FCCDE5"),
    ("Jardins Mossèn Cinto Verdaguer", "#80B1D3"),
    ("Jardins del Teatre Grec", "#BEBADA"),
    ("Jardins del Turó del Putxet", "#FB8072"),
    ("Jardins de Vil·la Amèlia - Cecília", "#8DD3C7"),
    ("Parc de la Guineueta", "#B3DE69")
])

# ============================================================================
# MODEL PARAMETERS
# ============================================================================



ALPHAS = np.array([0.088, 0.097, 0.069, 0.14, 0.017, 0.19, 0.086], dtype=np.float64)
W_MIN = 11.4
W_FLUSH = 9.6
B = 1.9e-4
D_PARAM = 1e-5



DT = 0.15 # RK4 integration step
BASELINE = 0.001  # Per-drain baseline probability




# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

rainfall_data = xr.open_dataset(PARAM_DIR / "Rainfall.nc")['data']
rainfall_21_data = xr.open_dataset(PARAM_DIR / "Rainfall_21.nc")['data']
r0_data = xr.open_dataset(PARAM_DIR / "R0_both.nc")['data']

with open(PARAM_DIR / "D_Matrix", 'rb') as f:
    D_global = pickle.load(f)
with open(PARAM_DIR / "X_coords", 'rb') as f:
    X_coords_dict = pickle.load(f)
with open(PARAM_DIR / "Y_coords", 'rb') as f:
    Y_coords_dict = pickle.load(f)
with open(PARAM_DIR / "ASPB_Interventions", 'rb') as f:
    interventions_dict = pickle.load(f)

unique_bs = pd.read_csv(DATA_DIR / "Unique_Items.csv")
revisions = pd.read_csv(DATA_DIR / "Revisions.csv")

n_drains = D_global.shape[0]

# Hardcoded zone ordering (consistent across all pipeline files)
zone_names = [
    "Parc de la Ciutadella",
    "Parc del Turó de la Peira",
    "Jardins Mossèn Cinto Verdaguer",
    "Jardins del Teatre Grec",
    "Jardins del Turó del Putxet",
    "Jardins de Vil·la Amèlia - Cecília",
    "Parc de la Guineueta"
]
n_zones = len(zone_names)
zone_name_to_idx = {name: i for i, name in enumerate(zone_names)}
drain_to_zone = np.array([zone_name_to_idx[z] for z in unique_bs['nom_zr']], dtype=np.int32)

X_emb = np.array([X_coords_dict[i] for i in range(n_drains)])
Y_emb = np.array([Y_coords_dict[i] for i in range(n_drains)])

unique_bs['data_reforma'] = pd.to_datetime(unique_bs['data_reforma'], errors='coerce')
unique_bs['data_reforma'] = unique_bs['data_reforma'].fillna(pd.Timestamp("2025-01-01"))
t_reformas_actual = np.array([(dt - d0).days for dt in unique_bs['data_reforma']], dtype=np.float64)

# Precompute zone membership arrays for fast zone-restricted M matrix
_zone_lists = [[] for _ in range(n_zones)]
for i in range(n_drains):
    _zone_lists[drain_to_zone[i]].append(i)
zone_members = np.array([idx for zl in _zone_lists for idx in zl], dtype=np.int32)
zone_starts = np.zeros(n_zones + 1, dtype=np.int32)
for z in range(n_zones):
    zone_starts[z + 1] = zone_starts[z] + len(_zone_lists[z])
del _zone_lists

print(f"Loaded {n_drains} drains, {n_zones} zones")

# ============================================================================
# CLIMATE INTERPOLATION
# ============================================================================

def interpolate_all_locations(data_array, lons, lats):
    lons_da = xr.DataArray(lons, dims='points')
    lats_da = xr.DataArray(lats, dims='points')
    result = data_array.interp(longitude=lons_da, latitude=lats_da)
    return result.values

# ============================================================================
# NUMBA FUNCTIONS
# ============================================================================

@jit(nopython=True)
def generate_M(D, alphas, drain_to_zone, zone_members, zone_starts, n):
    """Generate zone-restricted M matrix. Only same-zone entries are nonzero."""
    M = np.zeros((n, n))
    for i in range(n):
        z = drain_to_zone[i]
        alpha_i = alphas[z]
        for k in range(zone_starts[z], zone_starts[z + 1]):
            j = zone_members[k]
            M[i, j] = np.exp(-alpha_i * D[i, j])
    return M


@jit(nopython=True)
def compute_du(u, M, r0_day, B, D_param, n_drains, can_evolve,
               drain_to_zone, zone_members, zone_starts):
    """Compute du/dt. Inner loop only over same-zone drains for speed."""
    du = np.zeros(n_drains)
    for i in range(n_drains):
        if not can_evolve[i]:
            continue
        z = drain_to_zone[i]
        m_times_u = 0.0
        for k in range(zone_starts[z], zone_starts[z + 1]):
            j = zone_members[k]
            m_times_u += M[i, j] * u[j]
        du[i] = B * r0_day[i] * m_times_u * (1.0 - u[i]) - D_param * u[i]
    return du


@jit(nopython=True)
def compute_weighted_active_drains(drain_to_zone, t_reformas, zone_idx, day_start, day_end):
    """
    Count average active drains in a zone over [day_start, day_end],
    weighted by the fraction of time each drain is active (not reformed).
    
    If a drain gets reformed during the month, it contributes proportionally
    to the time it was active.
    """
    n_days_month = day_end - day_start + 1
    weighted_count = 0.0
    
    for i in range(len(drain_to_zone)):
        if drain_to_zone[i] == zone_idx:
            reform_day = t_reformas[i]
            
            if reform_day < day_start:
                # Reformed before month started: contributes 0
                continue
            elif reform_day >= day_end:
                # Active throughout the month: contributes 1
                weighted_count += 1.0
            else:
                # Reformed during the month: contributes fraction of days active
                days_active = reform_day - day_start + 1
                weighted_count += days_active / n_days_month
    
    return weighted_count


@jit(nopython=True)
def compute_zone_day_sums(solution, drain_to_zone, n_zones):
    """Aggregate drain-level solution to zone-level sums."""
    n_days = solution.shape[0]
    zone_sums = np.zeros((n_days, n_zones))
    for day in range(n_days):
        for i in range(solution.shape[1]):
            zone_sums[day, drain_to_zone[i]] += solution[day, i]
    return zone_sums


@jit(nopython=True)
def compute_zone_month_integral(zone_sums, zone_idx, day_start, day_end):
    """Integrate zone sum over [day_start, day_end] using trapezoidal rule."""
    n_days_sol = zone_sums.shape[0]
    d0_clamped = max(0, day_start)
    d1_clamped = min(n_days_sol - 1, day_end)
    
    if d0_clamped >= d1_clamped:
        return 0.0
    
    integral = 0.0
    for d in range(d0_clamped, d1_clamped):
        integral += 0.5 * (zone_sums[d, zone_idx] + zone_sums[d + 1, zone_idx])
    return integral


@jit(nopython=True)
def simulate_full(u0, n_drains, n_days, M, t_reformas, 
                  w_min, w_flush, B, D_param, dt,
                  rainfall_24h, rainfall_21d, r0_values,
                  interventions_list, intervention_starts,
                  drain_to_zone, zone_members, zone_starts, baseline):
    """Full simulation with intervention and environmental logic."""
    u = u0.copy()
    solution = np.zeros((n_days, n_drains))
    solution[0] = u.copy()
    
    n_substeps = int(1.0 / dt)
    can_evolve = np.zeros(n_drains, dtype=np.bool_)
    intervention_cancelled_day = np.full(n_drains, -1, dtype=np.int64)
    
    for day in range(1, n_days):
        for i in range(n_drains):
            # 1. Reformed -> u=0, permanent
            if day > t_reformas[i]:
                u[i] = 0.0
                can_evolve[i] = False
                continue
            
            is_flushed = rainfall_24h[day, i] > w_flush
            
            # 2. Flushed -> cancels intervention, u=baseline
            if is_flushed:
                intervention_cancelled_day[i] = day
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            # 3. Intervention (if not cancelled by flush)
            is_under_intervention = False
            start = intervention_starts[i]
            end = intervention_starts[i + 1]
            for k in range(start, end):
                inter_day = interventions_list[k]
                if inter_day <= day < inter_day + 49:
                    cancelled = intervention_cancelled_day[i]
                    if cancelled < 0 or cancelled < inter_day:
                        is_under_intervention = True
                        break
            
            if is_under_intervention:
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            # 4. Dry -> u=baseline
            if rainfall_21d[day, i] < w_min:
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            # 5. Can evolve
            can_evolve[i] = True
        
        # RK4 integration
        r0_day = r0_values[day]
        for _ in range(n_substeps):
            k1 = compute_du(u, M, r0_day, B, D_param, n_drains, can_evolve,
                            drain_to_zone, zone_members, zone_starts)
            u_temp = u + 0.5 * dt * k1
            k2 = compute_du(u_temp, M, r0_day, B, D_param, n_drains, can_evolve,
                            drain_to_zone, zone_members, zone_starts)
            u_temp = u + 0.5 * dt * k2
            k3 = compute_du(u_temp, M, r0_day, B, D_param, n_drains, can_evolve,
                            drain_to_zone, zone_members, zone_starts)
            u_temp = u + dt * k3
            k4 = compute_du(u_temp, M, r0_day, B, D_param, n_drains, can_evolve,
                            drain_to_zone, zone_members, zone_starts)
            
            u = u + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        solution[day] = u.copy()
    
    return solution

# ============================================================================
# PREPARE ASPB DATA - ZONE-MONTH AGGREGATION
# ============================================================================

def prepare_aspb_zone_month():
    """Aggregate ASPB data to zone-month level."""
    rev = revisions.copy()
    rev['Fecha'] = pd.to_datetime(rev['Fecha'])
    rev['year'] = rev['Fecha'].dt.year.astype(int)
    rev['month'] = rev['Fecha'].dt.month.astype(int)
    
    agg = rev.groupby(['nom_zr', 'year', 'month']).agg(
        n_positives=('activitat_both', 'sum'),
        n_visits=('Fecha', 'count')
    ).reset_index()
    
    agg['observed_rate'] = agg['n_positives'] / agg['n_visits']
    
    return agg

# ============================================================================
# PREPARE INTERVENTIONS (REALISTIC)
# ============================================================================

def prepare_interventions_realistic(n_drains):
    """Prepare ASPB interventions in array format for Numba."""
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        if i in interventions_dict:
            for day in interventions_dict[i]:
                interventions_list.append(day)
    
    intervention_starts[n_drains] = len(interventions_list)
    return np.array(interventions_list, dtype=np.int64), intervention_starts

# ============================================================================
# RUN SIMULATION (REALISTIC - ASPB INTERVENTIONS)
# ============================================================================

def run_simulation_realistic():
    """Run simulation with actual ASPB interventions and actual reform dates."""
    print("Running realistic simulation...")
    
    tf_date = datetime(2024, 12, 31)
    n_days = (tf_date - d0).days + 1
    
    # Climate data
    rainfall_24h = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    r0_vals = interpolate_all_locations(r0_data, X_emb, Y_emb)
    
    rainfall_24h = np.nan_to_num(rainfall_24h, nan=0.0)[:n_days]
    rainfall_21d = np.nan_to_num(rainfall_21d, nan=0.0)[:n_days]
    r0_vals = np.nan_to_num(r0_vals, nan=0.00001)[:n_days]
    
    # Realistic interventions (ASPB actual treatment dates)
    interventions_arr, intervention_starts = prepare_interventions_realistic(n_drains)
    
    # Actual t_reformas (not pushed to year start)
    t_reformas = t_reformas_actual
    
    # Per-drain initial conditions: BASELINE for active drains, 0 for reformed
    u0 = np.zeros(n_drains)
    for i in range(n_drains):
        if 0 <= t_reformas[i]:  # Active at day 0
            u0[i] = BASELINE
    
    M = generate_M(D_global, ALPHAS, drain_to_zone, zone_members, zone_starts, n_drains)
    
    solution = simulate_full(
        u0, n_drains, n_days, M, t_reformas,
        W_MIN, W_FLUSH, B, D_PARAM, DT,
        rainfall_24h, rainfall_21d, r0_vals,
        interventions_arr, intervention_starts,
        drain_to_zone, zone_members, zone_starts, BASELINE
    )
    
    return solution, t_reformas

# ============================================================================
# COMPUTE NORMALIZED ZONE-MONTH INTEGRALS
# ============================================================================

def compute_zone_month_integrals_normalized(solution, aspb_zm, t_reformas, mse_mode):
    """
    Compute zone-month predictions accounting for reformed drains.
    
    If mse_mode == "cases":
        - pred_cases = zone_integral / n_days_month
        - real_cases = (n_positives / n_visits) × n_active_weighted
    If mse_mode == "prevalence":
        - pred_prevalence = zone_integral / (n_days_month × n_active_weighted)
        - real_prevalence = n_positives / n_visits
    """
    zone_sums = compute_zone_day_sums(solution, drain_to_zone, n_zones)
    
    results = []
    
    for _, row in aspb_zm.iterrows():
        zone_name = row['nom_zr']
        year = int(row['year'])
        month = int(row['month'])
        n_visits = row['n_visits']
        n_positives = row['n_positives']
        observed_rate = row['observed_rate']
        
        # Month boundaries
        month_start_date = pd.Timestamp(year=year, month=month, day=1)
        month_end_date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        
        day_start = (month_start_date - d0).days
        day_end = (month_end_date - d0).days
        
        day_start = max(0, day_start)
        day_end = min(solution.shape[0] - 1, day_end)
        
        if day_start > day_end or day_start >= solution.shape[0]:
            continue
        
        zone_idx = zone_name_to_idx.get(zone_name, -1)
        if zone_idx < 0:
            continue
        
        n_days_month = day_end - day_start + 1
        
        integral = compute_zone_month_integral(zone_sums, zone_idx, day_start, day_end)
        n_active_weighted = compute_weighted_active_drains(
            drain_to_zone, t_reformas, zone_idx, day_start, day_end
        )
        
        if n_active_weighted < 0.01:
            continue
        
        # Compute quantities depending on mode
        if mse_mode == "prevalence":
            pred_value = integral / (n_days_month * n_active_weighted)
            real_value = n_positives / n_visits if n_visits > 0 else 0.0
        else:  # "cases"
            pred_value = integral / n_days_month
            prevalence = n_positives / n_visits if n_visits > 0 else 0.0
            real_value = prevalence * n_active_weighted
        
        results.append({
            'nom_zr': zone_name,
            'year': year,
            'month': month,
            'day_start': day_start,
            'day_end': day_end,
            'n_days_month': n_days_month,
            'n_visits': n_visits,
            'n_positives': n_positives,
            'n_active_weighted': n_active_weighted,
            'observed_rate': observed_rate,
            'pred_value': pred_value,
            'real_value': real_value,
        })
    
    return pd.DataFrame(results)

# ============================================================================
# STATISTICS
# ============================================================================

def weighted_pearson(x, y, w):
    """
    Weighted Pearson correlation coefficient.
    """
    w_sum = np.sum(w)
    mean_x = np.sum(w * x) / w_sum
    mean_y = np.sum(w * y) / w_sum
    
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y)) / w_sum
    var_x = np.sum(w * (x - mean_x)**2) / w_sum
    var_y = np.sum(w * (y - mean_y)**2) / w_sum
    
    if var_x == 0 or var_y == 0:
        return 0.0, 1.0
    
    r = cov_xy / np.sqrt(var_x * var_y)
    
    # Approximate p-value using Fisher transformation
    n_eff = w_sum**2 / np.sum(w**2)  # effective sample size
    if n_eff > 3:
        t_stat = r * np.sqrt((n_eff - 2) / (1 - r**2 + 1e-10))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_eff - 2))
    else:
        p_val = 1.0
    
    return r, p_val


def weighted_spearman(x, y, w):
    """
    Weighted Spearman correlation (rank-based, then weighted Pearson on ranks).
    """
    # Convert to ranks
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    # Apply weighted Pearson to the ranks
    return weighted_pearson(rank_x, rank_y, w)


def weighted_orthogonal_regression(x, y, w):
    """
    Weighted orthogonal distance regression (Total Least Squares).
    Returns slope, intercept.
    """
    # Normalize weights
    w_norm = w / np.sum(w)
    
    # Linear model for ODR
    def linear_func(p, x):
        return p[0] * x + p[1]
    
    # Create ODR model
    linear_model = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=1.0/np.sqrt(w_norm), sy=1.0/np.sqrt(w_norm))
    
    # Initial guess from weighted least squares
    w_sum = np.sum(w)
    mean_x = np.sum(w * x) / w_sum
    mean_y = np.sum(w * y) / w_sum
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y)) / w_sum
    var_x = np.sum(w * (x - mean_x)**2) / w_sum
    m_init = cov_xy / (var_x + 1e-10)
    b_init = mean_y - m_init * mean_x
    
    odr_obj = odr.ODR(data, linear_model, beta0=[m_init, b_init])
    output = odr_obj.run()
    
    slope = output.beta[0]
    intercept = output.beta[1]
    
    return slope, intercept


def compute_ccc(real, pred, weights=None):
    """
    Concordance Correlation Coefficient (CCC).
    Measures agreement between real and predicted values.
    CCC ranges from -1 to 1:
      - 1 = perfect concordance
      - 0 = no concordance
      - Values > 0.90 = excellent
      - Values 0.75-0.90 = good
      - Values < 0.75 = poor
    """
    if weights is not None:
        w_sum = np.sum(weights)
        mean_real = np.sum(weights * real) / w_sum
        mean_pred = np.sum(weights * pred) / w_sum
        var_real = np.sum(weights * (real - mean_real)**2) / w_sum
        var_pred = np.sum(weights * (pred - mean_pred)**2) / w_sum
        covar = np.sum(weights * (real - mean_real) * (pred - mean_pred)) / w_sum
    else:
        mean_real = np.mean(real)
        mean_pred = np.mean(pred)
        var_real = np.var(real)
        var_pred = np.var(pred)
        covar = np.cov(real, pred)[0, 1]
    
    ccc = (2 * covar) / (var_real + var_pred + (mean_real - mean_pred)**2)
    return ccc


def compute_statistics(data, mse_mode, weighting_mode):
    """
    Calculates the 3 main metrics we need to validate the model.
    weighting_mode: "weighted" uses n_visits as weights, "unweighted" treats all equally.
    """
    # Get the numpy arrays to make math easier
    real = data['real_value'].values
    pred = data['pred_value'].values
    weights = data['n_visits'].values.astype(float)
    
    use_weights = (weighting_mode == "weighted")
    
    # ---------------------------------------------------------
    # 1. RMSE (Root Mean Squared Error)
    # ---------------------------------------------------------
    errors = real - pred
    if use_weights:
        # Weighted MSE: sum(w * error^2) / sum(w)
        mse = np.sum(weights * errors**2) / np.sum(weights)
    else:
        mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    
    # ---------------------------------------------------------
    # 2. Correlations (Pearson & Spearman)
    # ---------------------------------------------------------
    if np.std(real) == 0 or np.std(pred) == 0:
        pearson_r, pearson_p = 0.0, 1.0
        spearman_r, spearman_p = 0.0, 1.0
    else:
        if use_weights:
            pearson_r, pearson_p = weighted_pearson(real, pred, weights)
            spearman_r, spearman_p = weighted_spearman(real, pred, weights)
        else:
            pearson_r, pearson_p = stats.pearsonr(real, pred)
            spearman_r, spearman_p = stats.spearmanr(real, pred)
    
    # ---------------------------------------------------------
    # 3. Wilcoxon Signed-Rank Test
    # ---------------------------------------------------------
    # Note: Wilcoxon doesn't have a straightforward weighted version.
    # We use a weighted sign test approximation for bias detection.
    if np.all(errors == 0):
        wilcoxon_p = 1.0
    else:
        try:
            if use_weights:
                # Weighted sign test: check if weighted sum of signs differs from zero
                signs = np.sign(errors)
                weighted_sum = np.sum(weights * signs)
                # Under H0, E[weighted_sum] = 0, Var = sum(w^2)
                var_ws = np.sum(weights**2)
                z_stat = weighted_sum / np.sqrt(var_ws + 1e-10)
                wilcoxon_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                w_stat, wilcoxon_p = stats.wilcoxon(real, pred)
        except:
            wilcoxon_p = 1.0
    
    # ---------------------------------------------------------
    # 4. Concordance Correlation Coefficient (CCC)
    # ---------------------------------------------------------
    if use_weights:
        ccc = compute_ccc(real, pred, weights)
    else:
        ccc = compute_ccc(real, pred)

    return {
        'RMSE': rmse,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Wilcoxon_p': wilcoxon_p,
        'CCC': ccc
    }

# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================

def plot_rainfall_diagnostics():
    """Plot rainfall and 21-day rainfall patterns over the years."""
    print("\nGenerating rainfall diagnostic plots...")
    
    tf_date = datetime(2024, 12, 31)
    n_days = (tf_date - d0).days + 1
    
    rainfall_24h = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    
    rainfall_24h = np.nan_to_num(rainfall_24h, nan=0.0)[:n_days]
    rainfall_21d = np.nan_to_num(rainfall_21d, nan=0.0)[:n_days]
    
    # Daily averages across all drains
    rainfall_24h_avg = np.mean(rainfall_24h, axis=1)
    rainfall_21d_avg = np.mean(rainfall_21d, axis=1)
    
    dates = pd.date_range(start=d0, periods=n_days, freq='D')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Daily rainfall
    ax = axes[0]
    ax.plot(dates, rainfall_24h_avg, linewidth=0.5, alpha=0.7, label='24h Rainfall')
    ax.axhline(y=W_FLUSH, color='red', linestyle='--', linewidth=2, label=f'Flush threshold ({W_FLUSH}mm)')
    ax.set_ylabel('Rainfall (mm)', fontsize=11)
    ax.set_title('Daily Rainfall (24h) - Average across all drains', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Plot 2: 21-day accumulated rainfall
    ax = axes[1]
    ax.plot(dates, rainfall_21d_avg, linewidth=1, alpha=0.8, label='21-day Accumulated')
    ax.axhline(y=W_MIN, color='orange', linestyle='--', linewidth=2, label=f'Dry threshold (W_MIN={W_MIN}mm)')
    ax.fill_between(dates, 0, W_MIN, where=(rainfall_21d_avg < W_MIN), alpha=0.2, color='orange', label='Dry periods')
    ax.set_ylabel('Accumulated Rainfall (mm)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_title('21-day Accumulated Rainfall - Average across all drains', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / "Rainfall_Diagnostics.png", dpi=150)
        print("  Saved: Rainfall_Diagnostics.png")
    
    # Summary statistics
    print(f"\n  Rainfall Statistics (2019-2024):")
    print(f"    24h rainfall: min={rainfall_24h_avg.min():.2f}, mean={rainfall_24h_avg.mean():.2f}, max={rainfall_24h_avg.max():.2f}")
    print(f"    21d rainfall: min={rainfall_21d_avg.min():.2f}, mean={rainfall_21d_avg.mean():.2f}, max={rainfall_21d_avg.max():.2f}")
    dry_days = np.sum(rainfall_21d_avg < W_MIN)
    dry_pct = 100 * dry_days / n_days
    print(f"    Dry days (<{W_MIN}mm): {dry_days}/{n_days} ({dry_pct:.1f}%)")
    flushed_days = np.sum(rainfall_24h_avg > W_FLUSH)
    flushed_pct = 100 * flushed_days / n_days
    print(f"    Flush events (>{W_FLUSH}mm): {flushed_days}/{n_days} ({flushed_pct:.1f}%)\n")

# ============================================================================
# PLOTTING
# ============================================================================

def weighted_kde(data, weights, grid, bandwidth=None):
    """
    Compute weighted KDE using Gaussian kernels.
    Each data point contributes proportionally to its weight.
    """
    if bandwidth is None:
        # Silverman's rule of thumb, adjusted for weighted data
        n_eff = np.sum(weights)**2 / np.sum(weights**2)
        std = np.sqrt(np.sum(weights * (data - np.sum(weights * data) / np.sum(weights))**2) / np.sum(weights))
        bandwidth = 1.06 * std * n_eff**(-1/5)
        bandwidth = max(bandwidth, 0.01)  # minimum bandwidth to avoid division issues
    
    # Normalize weights to sum to 1
    w_norm = weights / np.sum(weights)
    
    # Compute KDE at each grid point
    kde_values = np.zeros_like(grid)
    for i, g in enumerate(grid):
        kde_values[i] = np.sum(w_norm * np.exp(-0.5 * ((g - data) / bandwidth)**2)) / (bandwidth * np.sqrt(2 * np.pi))
    
    return kde_values


def plot_scatter(data, mse_mode, weighting_mode, filter_percentile=None):
    """Scatter plot with marginal KDEs and inset zoom."""
    
    # Filter out bottom percentile by n_visits if requested
    if filter_percentile is not None:
        visits_threshold = np.percentile(data['n_visits'], filter_percentile)
        data = data[data['n_visits'] >= visits_threshold].copy()
        print(f"  Filtered: keeping points with n_visits >= {visits_threshold:.0f} (top {100-filter_percentile:.0f}%)")
    
    # Create figure with gridspec for main plot + marginals
    fig = plt.figure(figsize=(10, 10))
    
    # Grid: main scatter takes most space, marginals on top and right
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])  # Main scatter
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)  # Top marginal (x distribution)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)  # Right marginal (y distribution)
    
    # Data
    x = data['real_value'].values
    y = data['pred_value'].values
    weights = data['n_visits'].values.astype(float)
    max_visits = data['n_visits'].max()
    min_size, max_size = 20, 200
    
    use_weights = (weighting_mode == "weighted")
    
    # ---------------------------------------------------------
    # MAIN SCATTER PLOT
    # ---------------------------------------------------------
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        sizes = min_size + (zone_data['n_visits'] / max_visits) * (max_size - min_size)
        ax_main.scatter(zone_data['real_value'], zone_data['pred_value'], 
                        alpha=0.6, s=sizes, color=zone_colors[zone])
        ax_main.scatter([], [], s=80, alpha=0.6, color=zone_colors[zone], label=zone)
    
    # Orthogonal regression
    if use_weights:
        m_odr, b_odr = weighted_orthogonal_regression(x, y, weights)
    else:
        m_odr, b_odr = weighted_orthogonal_regression(x, y, np.ones_like(x))
    
    x_fit = np.array([x.min(), x.max()])
    y_fit_odr = m_odr * x_fit + b_odr
    
    # ODR statistics
    ortho_residuals = np.abs(y - m_odr * x - b_odr) / np.sqrt(1 + m_odr**2)
    if use_weights:
        w_sum = np.sum(weights)
        mean_x = np.sum(weights * x) / w_sum
        mean_y = np.sum(weights * y) / w_sum
        total_var = np.sum(weights * ((x - mean_x)**2 + (y - mean_y)**2)) / w_sum
        resid_var = np.sum(weights * ortho_residuals**2) / w_sum
        odr_rmse = np.sqrt(resid_var)
        rmse_y = np.sqrt(np.sum(weights * (y - x)**2) / w_sum)
    else:
        mean_x, mean_y = np.mean(x), np.mean(y)
        total_var = np.mean((x - mean_x)**2 + (y - mean_y)**2)
        resid_var = np.mean(ortho_residuals**2)
        odr_rmse = np.sqrt(resid_var)
        rmse_y = np.sqrt(np.mean((y - x)**2))
    
    odr_r_squared = 1 - (resid_var / total_var) if total_var > 0 else 0
    
    # Plot regression and identity lines
    ax_main.plot(x_fit, y_fit_odr, 'r-', linewidth=2)
    max_val = max(x.max(), y.max())
    ax_main.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    # Stats box
    text_str = f"y = {m_odr:.3f}x + {b_odr:.4f}\nR² = {odr_r_squared:.4f}"
    ax_main.text(0.98, 0.15, text_str, transform=ax_main.transAxes, fontsize=10, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    if mse_mode == "prevalence":
        ax_main.set_xlabel('Real Prevalence', fontsize=12)
        ax_main.set_ylabel('Predicted Prevalence', fontsize=12)
    else:
        ax_main.set_xlabel('Real Cases', fontsize=12)
        ax_main.set_ylabel('Predicted Cases', fontsize=12)
    
    ax_main.legend(fontsize=8, loc='upper left', ncol=2)
    ax_main.grid(alpha=0.3)
    
    # ---------------------------------------------------------
    # MARGINAL KDEs (weighted)
    # ---------------------------------------------------------
    # Grid for KDE evaluation
    grid_x = np.linspace(0, max_val * 1.05, 200)
    grid_y = np.linspace(0, max_val * 1.05, 200)
    
    # Compute weighted KDEs
    kde_x = weighted_kde(x, weights, grid_x)
    kde_y = weighted_kde(y, weights, grid_y)
    
    # Top marginal (x = real)
    ax_top.fill_between(grid_x, kde_x, alpha=0.4, color='steelblue')
    ax_top.plot(grid_x, kde_x, color='steelblue', linewidth=1.5)
    ax_top.set_ylabel('Density', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(ax_main.get_xlim())
    
    # Right marginal (y = predicted)
    ax_right.fill_betweenx(grid_y, kde_y, alpha=0.4, color='coral')
    ax_right.plot(kde_y, grid_y, color='coral', linewidth=1.5)
    ax_right.set_xlabel('Density', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.set_ylim(ax_main.get_ylim())
    
    # ---------------------------------------------------------
    # INSET ZOOM (0 to 0.2 region)
    # ---------------------------------------------------------
    inset_lim = 0.2
    
    # Position inset in upper-right area of main plot (avoiding the marginals)
    ax_inset = ax_main.inset_axes([0.55, 0.55, 0.4, 0.4])  # [x, y, width, height] in axes coords
    
    # Plot same data in inset
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        sizes = min_size + (zone_data['n_visits'] / max_visits) * (max_size - min_size)
        ax_inset.scatter(zone_data['real_value'], zone_data['pred_value'], 
                         alpha=0.6, s=sizes * 0.5, color=zone_colors[zone])
    
    # Regression and identity in inset
    x_fit_inset = np.array([0, inset_lim])
    y_fit_inset = m_odr * x_fit_inset + b_odr
    ax_inset.plot(x_fit_inset, y_fit_inset, 'r-', linewidth=1.5)
    ax_inset.plot([0, inset_lim], [0, inset_lim], 'k--', alpha=0.3)
    
    ax_inset.set_xlim(0, inset_lim)
    ax_inset.set_ylim(0, inset_lim)
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.tick_params(labelsize=8)
    ax_inset.grid(alpha=0.3)
    ax_inset.set_title(f'Zoom: 0-{inset_lim}', fontsize=9)
    
    # Draw rectangle on main plot showing inset region
    rect = plt.Rectangle((0, 0), inset_lim, inset_lim, fill=False, 
                          edgecolor='gray', linestyle='--', linewidth=1.5)
    ax_main.add_patch(rect)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_ZoneMonth_Scatter_{mse_mode}.png", dpi=150)
        print(f"  Saved scatter: Validation_ZoneMonth_Scatter_{mse_mode}.png")
    
    return {
        'odr_slope': m_odr, 
        'odr_intercept': b_odr,
        'odr_r_squared': odr_r_squared,
        'odr_rmse': odr_rmse,
        'rmse_y': rmse_y
    }


def plot_scatter_2(data, mse_mode, weighting_mode):
    """
    Panel plot:
    - Left: Zone centroids with error crosses (weighted by n_visits)
    - Right: Overlayed KDE distributions for real vs predicted
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get all data for ODR fit
    x_all = data['real_value'].values
    y_all = data['pred_value'].values
    weights_all = data['n_visits'].values.astype(float)
    
    # Compute ODR fit on all data (same as plot_scatter)
    use_weights = (weighting_mode == "weighted")
    if use_weights:
        m_odr, b_odr = weighted_orthogonal_regression(x_all, y_all, weights_all)
    else:
        m_odr, b_odr = weighted_orthogonal_regression(x_all, y_all, np.ones_like(x_all))
    
    # ---------------------------------------------------------
    # LEFT PANEL: Zone centroids with crosses
    # ---------------------------------------------------------
    ax = axes[0]
    
    # Compute weighted centroids and std for each zone
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        if len(zone_data) == 0:
            continue
        
        x = zone_data['real_value'].values
        y = zone_data['pred_value'].values
        w = zone_data['n_visits'].values.astype(float)
        w_sum = np.sum(w)
        
        # Weighted mean (centroid)
        mean_x = np.sum(w * x) / w_sum
        mean_y = np.sum(w * y) / w_sum
        
        # Weighted std (for error bars)
        std_x = np.sqrt(np.sum(w * (x - mean_x)**2) / w_sum)
        std_y = np.sqrt(np.sum(w * (y - mean_y)**2) / w_sum)
        
        # Plot centroid with error bars (cross)
        ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, 
                    fmt='o', markersize=10, capsize=5, capthick=2,
                    color=zone_colors[zone], ecolor=zone_colors[zone],
                    label=zone, alpha=0.8, linewidth=2)
    
    # ODR regression line (red)
    plot_lim = 0.5
    x_fit = np.array([0, plot_lim])
    y_fit = m_odr * x_fit + b_odr
    ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'ODR fit (y={m_odr:.2f}x+{b_odr:.3f})')
    
    # Identity line
    ax.plot([0, plot_lim], [0, plot_lim], 'k--', alpha=0.3, label='Identity')
    
    # Labels
    if mse_mode == "prevalence":
        ax.set_xlabel('Real Prevalence (weighted mean)', fontsize=11)
        ax.set_ylabel('Predicted Prevalence (weighted mean)', fontsize=11)
    else:
        ax.set_xlabel('Real Cases (weighted mean)', fontsize=11)
        ax.set_ylabel('Predicted Cases (weighted mean)', fontsize=11)
    
    ax.set_title('Zone Centroids (error bars = weighted std)', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, plot_lim)
    ax.set_ylim(0, plot_lim)
    
    # ---------------------------------------------------------
    # RIGHT PANEL: Overlayed KDE distributions
    # ---------------------------------------------------------
    ax = axes[1]
    
    # Grid for KDE (full range)
    max_val = 1.0
    grid = np.linspace(0, max_val * 1.05, 200)
    
    # Compute weighted KDEs
    kde_real = weighted_kde(x_all, weights_all, grid)
    kde_pred = weighted_kde(y_all, weights_all, grid)
    
    # Plot overlayed
    ax.fill_between(grid, kde_real, alpha=0.3, color='steelblue', label='Real')
    ax.plot(grid, kde_real, color='steelblue', linewidth=2)
    
    ax.fill_between(grid, kde_pred, alpha=0.3, color='coral', label='Predicted')
    ax.plot(grid, kde_pred, color='coral', linewidth=2)
    
    # Labels
    label = 'Prevalence' if mse_mode == 'prevalence' else 'Cases'
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Density (weighted)', fontsize=11)
    ax.set_title('Distribution Comparison (weighted by n_visits)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_ZoneMonth_Scatter2_{mse_mode}.png", dpi=150)
        print(f"  Saved scatter2: Validation_ZoneMonth_Scatter2_{mse_mode}.png")


def plot_time_series_by_zone(data, mse_mode):
    """Plot real vs predicted values over time for each zone."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, zone in enumerate(zone_names):
        if i >= len(axes):
            break
        ax = axes[i]
        zone_data = data[data['nom_zr'] == zone].sort_values(['year', 'month'])
        
        time_idx = zone_data['year'].values + zone_data['month'].values / 12.0
        real = zone_data['real_value'].values
        pred = zone_data['pred_value'].values
        
        ax.plot(time_idx, real, 'o-', color='blue', label='Real', markersize=4, alpha=0.7)
        ax.plot(time_idx, pred, 's--', color='red', label='Pred', markersize=4, alpha=0.7)
        
        ax.set_title(zone, fontsize=10)
        ax.set_xlabel('Time (Year)')
        ylabel = 'Prevalence' if mse_mode == 'prevalence' else 'Cases'
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    for j in range(len(zone_names), len(axes)):
        axes[j].set_visible(False)
    
    title = 'Prevalence' if mse_mode == 'prevalence' else 'Cases'
    plt.suptitle(f'Zone-Month Time Series ({title})', fontsize=14)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_ZoneMonth_TimeSeries_{mse_mode}.png", dpi=150)
        print(f"  Saved time series: Validation_ZoneMonth_TimeSeries_{mse_mode}.png")


def aggregate_by_zone(data):
    """Aggregate zone-month data by zone (sum all months for each zone)."""
    agg = data.groupby('nom_zr').agg({
        'real_value': 'sum',
        'pred_value': 'sum'
    }).reset_index()
    return agg


def aggregate_by_year(data):
    """Aggregate zone-month data by year (sum all zones-months for each year)."""
    agg = data.groupby('year').agg({
        'real_value': 'sum',
        'pred_value': 'sum'
    }).reset_index()
    return agg


def plot_aggregated_scatter_panel(data, mse_mode):
    """Panel plot with two subplots: aggregated by zone (7 dots) and by year (5 dots)."""
    by_zone = aggregate_by_zone(data)
    by_year = aggregate_by_year(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== LEFT: By Zone ==========
    ax = axes[0]
    # SWAPPED: real on x, pred on y
    x = by_zone['real_value'].values
    y = by_zone['pred_value'].values
    
    ax.scatter(x, y, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    for idx, row in by_zone.iterrows():
        ax.annotate(row['nom_zr'][:15], (row['real_value'], row['pred_value']), 
                   fontsize=8, ha='center', va='bottom')
    
    # Use ODR (unweighted for aggregated data)
    m_odr, b_odr = weighted_orthogonal_regression(x, y, np.ones_like(x))
    x_fit = np.array([x.min(), x.max()])
    y_fit_odr = m_odr * x_fit + b_odr
    
    mse = np.mean((y - x) ** 2)
    
    # Plot lines without legend labels
    ax.plot(x_fit, y_fit_odr, 'r-', linewidth=2)
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    text_str = f"MSE = {mse:.6f}\nODR: y = {m_odr:.3f}x + {b_odr:.4f}\nn = 7 zones"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    label = 'Prevalence' if mse_mode == 'prevalence' else 'Cases'
    ax.set_xlabel(f'Real {label} (sum)', fontsize=11)
    ax.set_ylabel(f'Predicted {label} (sum)', fontsize=11)
    ax.set_title('Aggregated by Zone', fontsize=12)
    ax.grid(alpha=0.3)
    
    # ========== RIGHT: By Year ==========
    ax = axes[1]
    # SWAPPED: real on x, pred on y
    x = by_year['real_value'].values
    y = by_year['pred_value'].values
    
    ax.scatter(x, y, s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='orange')
    
    for idx, row in by_year.iterrows():
        ax.annotate(str(int(row['year'])), (row['real_value'], row['pred_value']), 
                   fontsize=9, ha='center', va='bottom')
    
    # Use ODR (unweighted for aggregated data)
    m_odr, b_odr = weighted_orthogonal_regression(x, y, np.ones_like(x))
    x_fit = np.array([x.min(), x.max()])
    y_fit_odr = m_odr * x_fit + b_odr
    
    mse = np.mean((y - x) ** 2)
    
    # Plot lines without legend labels
    ax.plot(x_fit, y_fit_odr, 'r-', linewidth=2)
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    text_str = f"MSE = {mse:.6f}\nODR: y = {m_odr:.3f}x + {b_odr:.4f}\nn = 6 years"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'Real {label} (sum)', fontsize=11)
    ax.set_ylabel(f'Predicted {label} (sum)', fontsize=11)
    ax.set_title('Aggregated by Year', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Aggregated Scatter Plots ({label})', fontsize=14, y=1.00)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_Aggregated_Scatter_{mse_mode}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved aggregated scatter: Validation_Aggregated_Scatter_{mse_mode}.png")


def plot_signed_error_timeseries(data, mse_mode, weighting_mode):
    """
    Plot signed error (real - predicted) time series for all zones with weighted monthly mean.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    use_weights = (weighting_mode == "weighted")
    
    # Scale dot sizes based on n_visits (normalize to reasonable range)
    max_visits = data['n_visits'].max()
    min_size, max_size = 20, 200
    
    # Plot individual zone data points (no lines connecting them)
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone].sort_values(['year', 'month'])
        
        # Compute signed error (real - pred)
        signed_error = zone_data['real_value'].values - zone_data['pred_value'].values
        
        # Create proper datetime for x-axis
        dates = []
        for _, row in zone_data.iterrows():
            dates.append(datetime(int(row['year']), int(row['month']), 15))
        
        # Scale marker sizes proportionally to n_visits
        sizes = min_size + (zone_data['n_visits'].values / max_visits) * (max_size - min_size)
        
        # Plot only scatter points (no connecting lines)
        ax.scatter(dates, signed_error, s=sizes, color=zone_colors[zone], 
                   alpha=0.8, edgecolors='white', linewidths=0.5)
        # Add uniform-sized dummy point for legend only
        ax.scatter([], [], s=80, alpha=0.8, color=zone_colors[zone], label=zone)
    
    # Compute weighted moving average across all zones
    # Sort all data by time
    data_sorted = data.sort_values(['year', 'month']).copy()
    data_sorted['date'] = pd.to_datetime(data_sorted[['year', 'month']].assign(day=15))
    data_sorted['error'] = data_sorted['real_value'] - data_sorted['pred_value']
    
    # Group by month and compute weighted mean and standard error
    monthly_stats = []
    for date in data_sorted['date'].unique():
        month_data = data_sorted[data_sorted['date'] == date]
        weights = month_data['n_visits'].values.astype(float)
        errors = month_data['error'].values
        
        if use_weights:
            # Weighted mean
            w_mean = np.sum(weights * errors) / np.sum(weights)
            
            # Weighted standard error (SE = sqrt(weighted_variance / effective_n))
            # Effective sample size for weighted data: n_eff = (sum(w))^2 / sum(w^2)
            w_var = np.sum(weights * (errors - w_mean)**2) / np.sum(weights)
            n_eff = np.sum(weights)**2 / np.sum(weights**2)
            w_se = np.sqrt(w_var / n_eff)
        else:
            # Unweighted mean and SE
            w_mean = np.mean(errors)
            w_se = np.std(errors, ddof=1) / np.sqrt(len(errors)) if len(errors) > 1 else 0.0
        
        monthly_stats.append({
            'date': date,
            'mean': w_mean,
            'se': w_se
        })
    
    stats_df = pd.DataFrame(monthly_stats)
    
    # Plot confidence band (±1 SE, approximately 68% confidence) - no legend entry
    ax.fill_between(stats_df['date'], 
                    stats_df['mean'] - stats_df['se'],
                    stats_df['mean'] + stats_df['se'],
                    color='gray', alpha=0.2, edgecolor='none', zorder=50)
    
    # Plot weighted monthly mean line with legend label
    ax.plot(stats_df['date'], stats_df['mean'], 'k-', linewidth=2.5, zorder=100, label='Weighted average')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Format x-axis: show month name + year every 3 months
    from matplotlib.dates import MonthLocator, DateFormatter
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Date', fontsize=12)
    ylabel = 'Real prevalence - Predicted prevalence'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_SignedError_TimeSeries_{mse_mode}.png", dpi=150)
        print(f"  Saved signed error plot: Validation_SignedError_TimeSeries_{mse_mode}.png")


def plot_error_vs_visits(data, mse_mode):
    """
    Plot error vs number of visits.
    Shows if prediction error depends on sample size.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    n_visits = data['n_visits'].values
    errors = data['real_value'].values - data['pred_value'].values
    
    # Color by zone
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        zone_visits = zone_data['n_visits'].values
        zone_errors = zone_data['real_value'].values - zone_data['pred_value'].values
        
        ax.scatter(zone_visits, zone_errors, s=60, alpha=0.7, 
                   color=zone_colors[zone], edgecolors='white', linewidths=0.5, label=zone)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax.set_xlabel('Number of Visits', fontsize=12)
    ylabel = 'Error (Real - Predicted)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Prediction Error vs Sample Size', fontsize=12)
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_Error_vs_Visits_{mse_mode}.png", dpi=150)
        print(f"  Saved error vs visits plot: Validation_Error_vs_Visits_{mse_mode}.png")


# ============================================================================
# P-VALUE ANALYSIS (BINOMIAL TEST)
# ============================================================================

def compute_binomial_pvalues(data):
    """
    Compute p-values for each zone-month observation.
    
    The question: "How likely/unlikely is it to see k positives out of n visits
    if the true prevalence is pred_prevalence?"
    
    Uses a two-sided binomial test: P(X >= k) + P(X <= k) where X ~ Binomial(n, p_pred)
    A low p-value means the observation is surprising given the prediction.
    """
    pvalues = []
    
    for idx, row in data.iterrows():
        n_visits = int(row['n_visits'])
        k_positives = int(row['n_positives'])
        pred_prevalence = row['pred_value']  # This is the predicted prevalence
        
        # Clamp predicted prevalence to valid range [0, 1]
        pred_prevalence = max(0.0, min(1.0, pred_prevalence))
        
        # Two-sided binomial test
        # P-value = probability of seeing result as extreme or more extreme than observed
        if pred_prevalence <= 0:
            # If prediction is 0, any positive is "impossible"
            p_val = 0.0 if k_positives > 0 else 1.0
        elif pred_prevalence >= 1:
            # If prediction is 1, any non-positive is "impossible"
            p_val = 0.0 if k_positives < n_visits else 1.0
        else:
            # Two-sided binomial test using scipy
            result = stats.binomtest(k_positives, n_visits, pred_prevalence, alternative='two-sided')
            p_val = result.pvalue
        
        pvalues.append({
            'nom_zr': row['nom_zr'],
            'year': row['year'],
            'month': row['month'],
            'n_visits': n_visits,
            'n_positives': k_positives,
            'observed_rate': row['observed_rate'],
            'pred_prevalence': pred_prevalence,
            'pvalue': p_val
        })
    
    return pd.DataFrame(pvalues)


def plot_pvalue_histogram(pval_df, mse_mode):
    """
    Plot histogram of p-values with significance threshold.
    Under the null hypothesis (model is correct), p-values should be uniform [0,1].
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pvals = pval_df['pvalue'].values
    
    # ===== LEFT: Histogram =====
    ax = axes[0]
    bins = np.linspace(0, 1, 21)  # 20 bins
    ax.hist(pvals, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axhline(y=len(pvals)/20, color='red', linestyle='--', linewidth=2, label='Expected (uniform)')
    ax.axvline(x=0.05, color='orange', linestyle='-', linewidth=2, label='α=0.05')
    
    # Stats
    n_significant = np.sum(pvals < 0.05)
    pct_significant = 100 * n_significant / len(pvals)
    
    ax.set_xlabel('P-value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Binomial P-values', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    text_str = f"n = {len(pvals)}\nSignificant (p<0.05): {n_significant} ({pct_significant:.1f}%)\nExpected if model correct: ~5%"
    ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== RIGHT: Q-Q plot =====
    ax = axes[1]
    sorted_pvals = np.sort(pvals)
    n = len(sorted_pvals)
    expected = (np.arange(1, n+1) - 0.5) / n  # Expected uniform quantiles
    
    ax.scatter(expected, sorted_pvals, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect uniform')
    
    ax.set_xlabel('Expected (Uniform)', fontsize=12)
    ax.set_ylabel('Observed P-values', fontsize=12)
    ax.set_title('Q-Q Plot (P-values vs Uniform)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_Pvalue_Distribution_{mse_mode}.png", dpi=150)
        print(f"  Saved p-value histogram: Validation_Pvalue_Distribution_{mse_mode}.png")


def plot_pvalue_heatmap(pval_df, mse_mode):
    """
    Plot heatmap of p-values by zone and time.
    Low p-values (red) indicate observations that are unlikely given the prediction.
    """
    # Pivot to create zone x (year-month) matrix
    pval_df['ym'] = pval_df['year'].astype(str) + '-' + pval_df['month'].astype(str).str.zfill(2)
    
    pivot = pval_df.pivot_table(index='nom_zr', columns='ym', values='pvalue', aggfunc='first')
    
    # Sort columns chronologically
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    
    fig, ax = plt.subplots(figsize=(18, 6))
    
    # Use a diverging colormap centered at 0.5, but we want low p-values to be "bad" (red)
    # So reverse it: low p-value = red, high p-value = blue
    cmap = plt.cm.RdYlGn  # Red (bad) -> Yellow -> Green (good)
    
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    # Mark significant cells (p < 0.05) with a border
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val) and val < 0.05:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                           edgecolor='black', linewidth=2))
    
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('P-value', fontsize=11)
    
    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Zone', fontsize=12)
    ax.set_title('Binomial P-values by Zone and Time\n(Black border = p < 0.05, Red = unlikely observation)', fontsize=12)
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_Pvalue_Heatmap_{mse_mode}.png", dpi=150)
        print(f"  Saved p-value heatmap: Validation_Pvalue_Heatmap_{mse_mode}.png")


def plot_pvalue_scatter(pval_df, mse_mode):
    """
    Scatter plot: observed vs predicted, colored by p-value.
    This shows where the model struggles (low p-values = surprising observations).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = pval_df['pred_prevalence'].values
    y = pval_df['observed_rate'].values
    pvals = pval_df['pvalue'].values
    
    # Color by p-value: use log scale for better visibility of low p-values
    # Clip p-values to avoid log(0)
    pvals_clipped = np.clip(pvals, 1e-10, 1)
    
    sc = ax.scatter(x, y, c=pvals_clipped, cmap='RdYlGn', 
                    norm=plt.cm.colors.LogNorm(vmin=0.001, vmax=1),
                    s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Identity line
    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('P-value (log scale)', fontsize=11)
    
    ax.set_xlabel('Predicted Prevalence', fontsize=12)
    ax.set_ylabel('Observed Prevalence', fontsize=12)
    ax.set_title('Observed vs Predicted Prevalence\n(Color = binomial p-value)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Count extremes
    n_low = np.sum(pvals < 0.05)
    n_high = np.sum(pvals > 0.95)
    text_str = f"P < 0.05: {n_low} ({100*n_low/len(pvals):.1f}%)\nP > 0.95: {n_high} ({100*n_high/len(pvals):.1f}%)"
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(FIGURES_DIR / f"Validation_Pvalue_Scatter_{mse_mode}.png", dpi=150)
        print(f"  Saved p-value scatter: Validation_Pvalue_Scatter_{mse_mode}.png")


# ============================================================================
# RUN VALIDATION
# ============================================================================

print("\n" + "="*60)
print(f"VALIDATION - MODE: {MSE_MODE}, WEIGHTING: {WEIGHTING_MODE}")
print("="*60 + "\n")

# Rainfall diagnostics (save but don't show)
plot_rainfall_diagnostics()
plt.close('all')  # Close the rainfall plot without showing

# Prepare ASPB zone-month data
print("Preparing ASPB zone-month data...")
aspb_zm = prepare_aspb_zone_month()
print(f"  Zone-month counts: {len(aspb_zm)}")
n_positive = (aspb_zm['n_positives'] > 0).sum()
n_zero = (aspb_zm['n_positives'] == 0).sum()
print(f"  Positive months: {n_positive}, Zero months: {n_zero}, Imbalance: {n_zero/n_positive:.2f}\n")

# Run simulation
solution, t_reformas = run_simulation_realistic()

# Compute zone-month integrals
print("Computing zone-month predictions...")
data_zm = compute_zone_month_integrals_normalized(solution, aspb_zm, t_reformas, MSE_MODE)
print(f"  {len(data_zm)} zone-months matched\n")

# Compute MSE (weighted or unweighted based on WEIGHTING_MODE)
errors = data_zm['real_value'].values - data_zm['pred_value'].values
if WEIGHTING_MODE == "weighted":
    weights = data_zm['n_visits'].values.astype(float)
    mse = np.sum(weights * errors**2) / np.sum(weights)
else:
    mse = np.mean(errors ** 2)
print(f"MSE ({MSE_MODE}, {WEIGHTING_MODE}): {mse:.6f}\n")

# Compute Detailed Statistics
stats_results = compute_statistics(data_zm, MSE_MODE, WEIGHTING_MODE)

# Create plots (save all, but only show scatter and error)
print("Creating plots...")

# Create time series plot (keep for showing)
plot_time_series_by_zone(data_zm, MSE_MODE)

# Create and save aggregated scatter (close after saving)
plot_aggregated_scatter_panel(data_zm, MSE_MODE)
plt.close('all')

# Create and save error timeseries (keep for showing)
plot_signed_error_timeseries(data_zm, MSE_MODE, WEIGHTING_MODE)

# Create and save error vs visits plot
plot_error_vs_visits(data_zm, MSE_MODE)
plt.close('all')

# ============================================================================
# P-VALUE ANALYSIS
# ============================================================================
print("\nComputing binomial p-values...")
pval_df = compute_binomial_pvalues(data_zm)

# P-value plots
plot_pvalue_histogram(pval_df, MSE_MODE)
plt.close('all')

plot_pvalue_heatmap(pval_df, MSE_MODE)
plt.close('all')

plot_pvalue_scatter(pval_df, MSE_MODE)
plt.close('all')

# Re-create the main plots for showing
print("\nGenerating final plots...")
scatter_stats = plot_scatter(data_zm, MSE_MODE, WEIGHTING_MODE)
#plot_scatter(data_zm, MSE_MODE, WEIGHTING_MODE, filter_percentile=10)  # Filtered version
plot_scatter_2(data_zm, MSE_MODE, WEIGHTING_MODE)  # Centroids + KDE panel
plot_time_series_by_zone(data_zm, MSE_MODE)
plot_signed_error_timeseries(data_zm, MSE_MODE, WEIGHTING_MODE)
plot_error_vs_visits(data_zm, MSE_MODE)
plot_pvalue_scatter(pval_df, MSE_MODE)
print()

# Print summary table
print("\n" + "="*60)
print(f"ZONE-MONTH SUMMARY TABLE ({MSE_MODE.upper()})")
print("="*60)
print(f"\n{'Zone':<40} {'Year':>6} {'Month':>6} {'Real':>10} {'Pred':>10} {'Error':>10} {'n_visits':>10} {'n_active':>10}")
print("-"*120)
for _, row in data_zm.iterrows():
    error = row['real_value'] - row['pred_value']
    print(f"{row['nom_zr']:<40} {row['year']:>6} {row['month']:>6} {row['real_value']:>10.4f} {row['pred_value']:>10.4f} {error:>10.4f} {row['n_visits']:>10} {row['n_active_weighted']:>10.2f}")

print("\n" + "="*60)
print(f"VALIDATION COMPLETE (MSE_MODE: {MSE_MODE}, WEIGHTING: {WEIGHTING_MODE})")
print("="*60)
print(f"\n  Total MSE: {mse:.6f}")
print(f"  Total zone-months: {len(data_zm)}")

print("\n" + "-"*60)
print("GOODNESS OF FIT STATISTICS")
print("-"*60)
print(f"  RMSE (Error Magnitude):      {stats_results['RMSE']:.6f}")

# Pearson Interpretation
p_p = stats_results['Pearson_p']
p_msg = "Significant (GOOD)" if p_p < 0.05 else "Not Significant (BAD)"
print(f"  Pearson r (Linearity):       {stats_results['Pearson_r']:.4f} (p={p_p:.4f}) -> {p_msg}")

# Spearman Interpretation
s_p = stats_results['Spearman_p']
s_msg = "Significant (GOOD)" if s_p < 0.05 else "Not Significant (BAD)"
print(f"  Spearman r (Monotonicity):   {stats_results['Spearman_r']:.4f} (p={s_p:.4f}) -> {s_msg}")

# Wilcoxon Interpretation
w_p = stats_results['Wilcoxon_p']
w_msg = "No Bias (GOOD)" if w_p > 0.05 else "Significant Bias (BAD)"
print(f"  Wilcoxon Test (Bias Check):  p={w_p:.4f} -> {w_msg}")

# CCC Interpretation
ccc = stats_results['CCC']
if ccc > 0.90:
    ccc_msg = "Excellent (GOOD)"
elif ccc > 0.75:
    ccc_msg = "Good"
else:
    ccc_msg = "Poor (BAD)"
print(f"  CCC (Concordance):           {ccc:.4f} -> {ccc_msg}")
print("    (CCC: >0.90=excellent, 0.75-0.90=good, <0.75=poor)")

print("-"*60)

# Orthogonal Distance Regression statistics (shown in plots)
print("\nORTHOGONAL DISTANCE REGRESSION (ODR / Total Least Squares)")
print("-"*60)
print(f"  Slope (ODR):                         {scatter_stats['odr_slope']:.4f}")
print(f"  Intercept (ODR):                     {scatter_stats['odr_intercept']:.4f}")
print(f"  R² (ODR, orthogonal variance):       {scatter_stats['odr_r_squared']:.4f}")
print(f"  RMSE (orthogonal distance):          {scatter_stats['odr_rmse']:.6f}")
print(f"  RMSE (y-direction, pred vs real):    {scatter_stats['rmse_y']:.6f}")
print("  Note: ODR R² measures fraction of total 2D variance explained by the line")
print("        using perpendicular distances, not vertical residuals.")
print("-"*60 + "\n")

# P-value summary statistics
print("BINOMIAL P-VALUE ANALYSIS")
print("-"*60)
pvals = pval_df['pvalue'].values
n_sig_low = np.sum(pvals < 0.05)
n_sig_high = np.sum(pvals > 0.95)
n_total = len(pvals)
print(f"  Total observations: {n_total}")
print(f"  P < 0.05 (surprisingly few/many positives): {n_sig_low} ({100*n_sig_low/n_total:.1f}%)")
print(f"  P > 0.95 (very close to prediction): {n_sig_high} ({100*n_sig_high/n_total:.1f}%)")
print(f"  Expected under correct model: ~5% significant")
print(f"  Interpretation: If >5% have p<0.05, model may be miscalibrated")
print("-"*60 + "\n")

# Display plots explanation
print("\nConfidence Band Explanation:")
print("  The black line shows the weighted monthly mean error (averaged across all zones each month).")
print("  The shaded gray area represents ±1 Standard Error (SE) around this monthly mean.")
print("  SE is computed using weighted variance and effective sample size: SE = sqrt(weighted_var / n_eff)")
print("  where n_eff = (sum(w))^2 / sum(w^2) accounts for unequal weighting.")
print("  This gives approximately 68% confidence interval for the mean error.")
print("  If the band contains zero, errors are not systematically biased.\n")

# Show only scatter and error timeseries plots
if SAVE_FIGURES:
    print("\nAll plots saved to Figures/Validation_Plots/")
else:
    print("\nSAVE_FIGURES=False - plots not saved")
print("Showing: Scatter plot, Error timeseries, Error vs Visits, and P-value scatter...")
plt.show()
