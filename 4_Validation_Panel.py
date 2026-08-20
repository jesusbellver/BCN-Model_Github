"""
Validation Panel Plot - Combined visualization for model validation.

Layout:
- Top left (horizontal): Prediction error vs sample size
- Bottom left (horizontal): Overlapped KDE distributions (real vs predicted)
- Top right (square): Centroids + crosses grouped by ZONE with global fit
- Bottom right (square): Centroids + crosses grouped by MONTH with global fit

All metrics weighted by number of visits.
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

MSE_MODE = "prevalence"
WEIGHTING_MODE = "weighted"
SAVE_FIGURES = True
BOTTOM_RIGHT_MODE = "year"  # "month" or "year"
DIST_MODE = "histogram"  # "kde" or "histogram"
HIST_BINS = 30           # number of bins (only used when DIST_MODE = "histogram")
CLIP_THRESHOLD = 0.5     # threshold for outlier clipping in distribution plots
LOAD_PARAMS_FROM_FILE = False  # True: load from file, False: use hardcoded values

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

# Month colors (one per month, using a nice categorical palette)
month_colors = {
    1: "#1f77b4",   # January - blue
    2: "#aec7e8",   # February - light blue
    3: "#2ca02c",   # March - green
    4: "#98df8a",   # April - light green
    5: "#d62728",   # May - red
    6: "#ff9896",   # June - light red
    7: "#9467bd",   # July - purple
    8: "#c5b0d5",   # August - light purple
    9: "#8c564b",   # September - brown
    10: "#c49c94",  # October - light brown
    11: "#e377c2",  # November - pink
    12: "#f7b6d2"   # December - light pink
}

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# Year colors (for years 2019-2024)
year_colors = {
    2019: "#1f77b4",   # blue
    2020: "#ff7f0e",   # orange
    2021: "#2ca02c",   # green
    2022: "#d62728",   # red
    2023: "#9467bd",   # purple
    2024: "#8c564b"    # brown
}


# ============================================================================
# LOAD MODEL PARAMETERS 
# ============================================================================


RESULTS_DIR = SCRIPT_DIR / "Parameter_estimation"
RESULTS_FILE = RESULTS_DIR / f"Estimated_params_both_nll_fit_balanced_4"

if LOAD_PARAMS_FROM_FILE:
    print(f"Loading parameters from: {RESULTS_FILE}")
    with open(RESULTS_FILE, 'rb') as f:
        _fit_results = pickle.load(f)

    # best_params is the full expanded array:
    #   indices 0-6  → per-zone alphas (order matches zone_names)
    #   index   7    → w_min
    #   index   8    → w_flush
    #   index   9    → B
    #   index   10   → D
    _bp = np.array(_fit_results['best_params'], dtype=np.float64)
    ALPHAS  = _bp[0:7]
    W_MIN   = float(_bp[7])
    W_FLUSH = float(_bp[8])
    B       = float(_bp[9])
    D_PARAM = float(_bp[10])
else:
    ALPHAS = np.array([0.088, 0.097, 0.069, 0.14, 0.017, 0.19, 0.086], dtype=np.float64)
    W_MIN = 11.4
    W_FLUSH = 9.6
    B = 1.9e-4
    D_PARAM = 1e-5



DT = 0.15 # RK4 integration step
BASELINE = 0.001  # Per-drain baseline probability


print(f"  ALPHAS  = {ALPHAS}")
print(f"  W_MIN   = {W_MIN}")
print(f"  W_FLUSH = {W_FLUSH}")
print(f"  B       = {B}")
print(f"  D_PARAM = {D_PARAM}")

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
# NUMBA FUNCTIONS (copied from 4_Validation_plots.py)
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
    n_days_month = day_end - day_start + 1
    weighted_count = 0.0
    for i in range(len(drain_to_zone)):
        if drain_to_zone[i] == zone_idx:
            reform_day = t_reformas[i]
            if reform_day < day_start:
                continue
            elif reform_day >= day_end:
                weighted_count += 1.0
            else:
                days_active = reform_day - day_start + 1
                weighted_count += days_active / n_days_month
    return weighted_count


@jit(nopython=True)
def compute_zone_day_sums(solution, drain_to_zone, n_zones):
    n_days = solution.shape[0]
    zone_sums = np.zeros((n_days, n_zones))
    for day in range(n_days):
        for i in range(solution.shape[1]):
            zone_sums[day, drain_to_zone[i]] += solution[day, i]
    return zone_sums


@jit(nopython=True)
def compute_zone_month_integral(zone_sums, zone_idx, day_start, day_end):
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
    u = u0.copy()
    solution = np.zeros((n_days, n_drains))
    solution[0] = u.copy()
    
    n_substeps = int(1.0 / dt)
    can_evolve = np.zeros(n_drains, dtype=np.bool_)
    intervention_cancelled_day = np.full(n_drains, -1, dtype=np.int64)
    
    for day in range(1, n_days):
        for i in range(n_drains):
            if day > t_reformas[i]:
                u[i] = 0.0
                can_evolve[i] = False
                continue
            
            is_flushed = rainfall_24h[day, i] > w_flush
            
            if is_flushed:
                intervention_cancelled_day[i] = day
                u[i] = baseline
                can_evolve[i] = False
                continue
            
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
            
            if rainfall_21d[day, i] < w_min:
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            can_evolve[i] = True
        
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
# DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_aspb_zone_month():
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


def prepare_interventions_realistic(n_drains):
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        if i in interventions_dict:
            for day in interventions_dict[i]:
                interventions_list.append(day)
    
    intervention_starts[n_drains] = len(interventions_list)
    return np.array(interventions_list, dtype=np.int64), intervention_starts


def run_simulation_realistic():
    print("Running realistic simulation...")
    
    tf_date = datetime(2024, 12, 31)
    n_days = (tf_date - d0).days + 1
    
    rainfall_24h = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    r0_vals = interpolate_all_locations(r0_data, X_emb, Y_emb)
    
    rainfall_24h = np.nan_to_num(rainfall_24h, nan=0.0)[:n_days]
    rainfall_21d = np.nan_to_num(rainfall_21d, nan=0.0)[:n_days]
    r0_vals = np.nan_to_num(r0_vals, nan=0.00001)[:n_days]
    
    interventions_arr, intervention_starts = prepare_interventions_realistic(n_drains)
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


def compute_zone_month_integrals_normalized(solution, aspb_zm, t_reformas, mse_mode):
    zone_sums = compute_zone_day_sums(solution, drain_to_zone, n_zones)
    results = []
    
    for _, row in aspb_zm.iterrows():
        zone_name = row['nom_zr']
        year = int(row['year'])
        month = int(row['month'])
        n_visits = row['n_visits']
        n_positives = row['n_positives']
        observed_rate = row['observed_rate']
        
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
        
        if mse_mode == "prevalence":
            pred_value = integral / (n_days_month * n_active_weighted)
            real_value = n_positives / n_visits if n_visits > 0 else 0.0
        else:
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
# HELPER FUNCTIONS
# ============================================================================

def weighted_kde(data, weights, grid, bandwidth=None):
    """Weighted KDE using Gaussian kernels."""
    if bandwidth is None:
        n_eff = np.sum(weights)**2 / np.sum(weights**2)
        std = np.sqrt(np.sum(weights * (data - np.sum(weights * data) / np.sum(weights))**2) / np.sum(weights))
        bandwidth = 1.06 * std * n_eff**(-1/5)
        bandwidth = max(bandwidth, 0.01)
    
    w_norm = weights / np.sum(weights)
    kde_values = np.zeros_like(grid)
    for i, g in enumerate(grid):
        kde_values[i] = np.sum(w_norm * np.exp(-0.5 * ((g - data) / bandwidth)**2)) / (bandwidth * np.sqrt(2 * np.pi))
    
    return kde_values


def compute_distribution_overlap(kde1, kde2, grid):
    """
    Overlap Coefficient (OVL) between two probability density functions.
    
    OVL = integral(min(f1, f2)) / integral(max(f1, f2))
    
    Since f1 and f2 are proper densities (integrate to 1):
      - integral(max(f1, f2)) = integral(f1 + f2 - min(f1, f2)) = 2 - OVL_num
      - So OVL = OVL_num / (2 - OVL_num)
    
    Returns 0 if no overlap, 1 if distributions are identical.
    """
    dx = grid[1] - grid[0]
    
    # Normalize to proper densities (safety check)
    kde1_norm = kde1 / (np.sum(kde1) * dx)
    kde2_norm = kde2 / (np.sum(kde2) * dx)
    
    # Intersection area (numerator)
    intersection = np.sum(np.minimum(kde1_norm, kde2_norm)) * dx
    
    # Union area (denominator): f1 + f2 - min(f1,f2) = max(f1,f2)
    union = np.sum(np.maximum(kde1_norm, kde2_norm)) * dx
    
    overlap = intersection / union if union > 0 else 0.0
    
    return overlap


def weighted_orthogonal_regression(x, y, w):
    """
    Weighted orthogonal distance regression.
    Returns slope, intercept, and R² value.
    """
    w_norm = w / np.sum(w)
    
    def linear_func(p, x):
        return p[0] * x + p[1]
    
    linear_model = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=1.0/np.sqrt(w_norm), sy=1.0/np.sqrt(w_norm))
    
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
    
    # Compute R² using orthogonal residuals (same as 4_Validation_plots.py)
    # This is the proper ODR R²: 1 - (orthogonal_residual_variance / total_2D_variance)
    ortho_residuals = np.abs(y - slope * x - intercept) / np.sqrt(1 + slope**2)
    total_var = np.sum(w * ((x - mean_x)**2 + (y - mean_y)**2)) / w_sum
    resid_var = np.sum(w * ortho_residuals**2) / w_sum
    r_squared = 1 - (resid_var / total_var) if total_var > 0 else 0
    
    return slope, intercept, r_squared

# ============================================================================
# MAIN PANEL PLOT
# ============================================================================

def create_validation_panel(data, mse_mode):
    """
    Create the main validation panel with 4 subplots:
    - Top left: Error vs sample size (horizontal)
    - Bottom left: Overlapped KDE distributions (horizontal)
    - Top right: Centroids + crosses by ZONE (square)
    - Bottom right: Centroids + crosses by MONTH (square)
    """
    
    # Create figure with custom gridspec
    # Left side: ~2/3 width (horizontal plots), Right side: ~1/3 width (square plots)
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], height_ratios=[1, 1],
                          left=0.07, right=0.98, top=0.95, bottom=0.09,
                          hspace=0.32, wspace=0.24)
    
    ax_error = fig.add_subplot(gs[0, 0])      # Top left - Error vs visits
    ax_kde = fig.add_subplot(gs[1, 0])        # Bottom left - KDE distributions
    ax_zone = fig.add_subplot(gs[0, 1])       # Top right - By zone
    ax_bottom_right = fig.add_subplot(gs[1, 1])  # Bottom right - By month or year
    
    # Data arrays
    x_all = data['real_value'].values
    y_all = data['pred_value'].values
    weights_all = data['n_visits'].values.astype(float)
    
    # Global ODR fit (will be shown on both right plots)
    m_odr, b_odr, r_squared = weighted_orthogonal_regression(x_all, y_all, weights_all)
    
    # ---------------------------------------------------------
    # TOP LEFT: Prediction Error vs Sample Size
    # ---------------------------------------------------------
    n_visits = data['n_visits'].values
    errors = data['real_value'].values - data['pred_value'].values
    max_visits = n_visits.max()
    min_size, max_size = 20, 200
    
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        zone_visits = zone_data['n_visits'].values
        zone_errors = zone_data['real_value'].values - zone_data['pred_value'].values
        
        # Dot sizes proportional to number of visits
        sizes = min_size + (zone_visits / max_visits) * (max_size - min_size)
        
        ax_error.scatter(zone_visits, zone_errors, s=sizes, alpha=0.7, 
                         color=zone_colors[zone], edgecolors='white', linewidths=0.5)
        # Add legend entry with uniform size
        ax_error.scatter([], [], s=60, color=zone_colors[zone], alpha=0.7, label=zone)
    
    ax_error.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax_error.set_xlabel('Number of drains sampled', fontsize=13)
    ax_error.set_ylabel('Error', fontsize=13)
    ax_error.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.9)
    ax_error.grid(alpha=0.3)
    
    # ---------------------------------------------------------
    # BOTTOM LEFT: Overlapped KDE or Histogram Distributions
    # ---------------------------------------------------------
    n_outliers_real = int(np.sum(x_all > CLIP_THRESHOLD))
    n_outliers_pred = int(np.sum(y_all > CLIP_THRESHOLD))
    max_val = CLIP_THRESHOLD
    print(f"  Distribution plot clipped at {CLIP_THRESHOLD} for visualization:")
    print(f"    - {n_outliers_real} observed zone-months outside plot range")
    print(f"    - {n_outliers_pred} predicted zone-months outside plot range")
    print(f"    - (Overlap coefficient calculated using ALL data)")
    label = 'Prevalence' if mse_mode == 'prevalence' else 'Cases'
    
    if DIST_MODE == "kde":
        grid = np.linspace(0, max_val, 300)
        kde_real = weighted_kde(x_all, weights_all, grid)
        kde_pred = weighted_kde(y_all, weights_all, grid)
        
        # Compute overlap coefficient
        kde_overlap = compute_distribution_overlap(kde_real, kde_pred, grid)
        
        ax_kde.fill_between(grid, kde_real, alpha=0.1, color='#1a5276', label='Observed')
        ax_kde.plot(grid, kde_real, color='#1a5276', linewidth=2)
        ax_kde.fill_between(grid, kde_pred, alpha=0.1, color='#e67e22', label='Predicted')
        ax_kde.plot(grid, kde_pred, color='#e67e22', linewidth=2)
        ax_kde.set_ylabel('Density', fontsize=13)
    
    else:  # histogram
        # Weights = n_visits: each zone-month contributes proportionally to its sample size.
        # IMPORTANT: Calculate overlap over ALL data, then display clipped range for visualization
        
        n_bins = HIST_BINS
        
        # Step 1: Create histogram over ALL data (no upper limit) for overlap calculation
        max_all_data = np.max(np.concatenate([x_all, y_all]))
        bins_full = np.linspace(0, max_all_data + 1, n_bins + 1)
        
        counts_real_full, _ = np.histogram(x_all, bins=bins_full, weights=weights_all)
        counts_pred_full, _ = np.histogram(y_all, bins=bins_full, weights=weights_all)
        
        # Step 2: Normalize to proper density over ALL data (integral = 1 over full range)
        bin_width_full = bins_full[1] - bins_full[0]
        sum_real_full = counts_real_full.sum()
        sum_pred_full = counts_pred_full.sum()
        
        counts_real_normalized = counts_real_full / (sum_real_full * bin_width_full) if sum_real_full > 0 else counts_real_full
        counts_pred_normalized = counts_pred_full / (sum_pred_full * bin_width_full) if sum_pred_full > 0 else counts_pred_full
        
        # Step 3: Calculate overlap coefficient over ALL data
        # OVL = intersection / union = sum(min) / sum(max)
        intersection = np.sum(np.minimum(counts_real_normalized, counts_pred_normalized)) * bin_width_full
        union = np.sum(np.maximum(counts_real_normalized, counts_pred_normalized)) * bin_width_full
        kde_overlap = intersection / union if union > 0 else 0.0
        
        # Step 4: Create visualization bins (clipped range for display only)
        bins_visual = np.linspace(0, max_val, n_bins + 1)
        
        counts_real_visual, _ = np.histogram(x_all, bins=bins_visual, weights=weights_all)
        counts_pred_visual, _ = np.histogram(y_all, bins=bins_visual, weights=weights_all)
        
        # Normalize visual histograms to match the density from full data
        bin_width_visual = bins_visual[1] - bins_visual[0]
        counts_real_visual = counts_real_visual / (sum_real_full * bin_width_full)
        counts_pred_visual = counts_pred_visual / (sum_pred_full * bin_width_full)
        
        ax_kde.bar(bins_visual[:-1], counts_real_visual, width=bin_width_visual, align='edge',
                   alpha=0.4, color='#1a5276', label='Observed')
        ax_kde.bar(bins_visual[:-1], counts_pred_visual, width=bin_width_visual, align='edge',
                   alpha=0.4, color='#e67e22', label='Predicted')
        
        ax_kde.set_ylabel('Density', fontsize=13)
    
    ax_kde.set_xlabel(label, fontsize=13)
    ax_kde.legend(fontsize=10, loc='upper right')
    ax_kde.grid(alpha=0.3)
    ax_kde.set_xlim(0, max_val)
    
    # ---------------------------------------------------------
    # TOP RIGHT: Centroids + Crosses by ZONE
    # ---------------------------------------------------------
    plot_lim_zone = 0.37
    max_visits_zone = 0
    
    # First pass: find max visits per zone for sizing
    for zone in zone_names:
        zone_data = data[data['nom_zr'] == zone]
        if len(zone_data) > 0:
            total_visits = zone_data['n_visits'].sum()
            max_visits_zone = max(max_visits_zone, total_visits)
    
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
        
        # Dot size proportional to total visits
        size = 80 + 200 * (w_sum / max_visits_zone)
        
        # Plot the actual data point with variable size
        ax_zone.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, 
                         fmt='o', markersize=np.sqrt(size/np.pi), capsize=4, capthick=1.5,
                         color=zone_colors[zone], ecolor=zone_colors[zone],
                         alpha=0.85, linewidth=1.5)
        # Add legend entry with uniform size (no error bars)
        ax_zone.scatter([], [], s=60, color=zone_colors[zone], alpha=0.85, label=zone)
    
    # Global ODR line
    x_fit_zone = np.array([0, plot_lim_zone])
    y_fit_zone = m_odr * x_fit_zone + b_odr
    ax_zone.plot(x_fit_zone, y_fit_zone, 'r-', linewidth=2, label='Global fit')
    
    # Identity line
    ax_zone.plot([0, plot_lim_zone], [0, plot_lim_zone], 'k--', alpha=0.4, linewidth=1.5)
    
    ax_zone.set_xlabel('Observed Prevalence', fontsize=13)
    ax_zone.set_ylabel('Predicted Prevalence', fontsize=13)
    ax_zone.grid(alpha=0.3)
    ax_zone.set_xlim(0, plot_lim_zone)
    ax_zone.set_ylim(0, plot_lim_zone)
    ax_zone.set_aspect('equal', adjustable='box')
    
    # ---------------------------------------------------------
    # BOTTOM RIGHT: Centroids + Crosses by MONTH or YEAR
    # ---------------------------------------------------------
    plot_lim_br = 0.25
    max_visits_br = 0
    
    if BOTTOM_RIGHT_MODE == "month":
        # By MONTH
        # First pass: find max visits per month for sizing
        for month in range(1, 13):
            month_data = data[data['month'] == month]
            if len(month_data) > 0:
                total_visits = month_data['n_visits'].sum()
                max_visits_br = max(max_visits_br, total_visits)
        
        for month in range(1, 13):
            month_data = data[data['month'] == month]
            if len(month_data) == 0:
                continue
            
            x = month_data['real_value'].values
            y = month_data['pred_value'].values
            w = month_data['n_visits'].values.astype(float)
            w_sum = np.sum(w)
            
            # Weighted mean (centroid)
            mean_x = np.sum(w * x) / w_sum
            mean_y = np.sum(w * y) / w_sum
            
            # Weighted std (for error bars)
            std_x = np.sqrt(np.sum(w * (x - mean_x)**2) / w_sum)
            std_y = np.sqrt(np.sum(w * (y - mean_y)**2) / w_sum)
            
            # Dot size proportional to total visits
            size = 80 + 200 * (w_sum / max_visits_br)
            
            # Plot the actual data point with variable size
            ax_bottom_right.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, 
                              fmt='o', markersize=np.sqrt(size/np.pi), capsize=4, capthick=1.5,
                              color=month_colors[month], ecolor=month_colors[month],
                              alpha=0.85, linewidth=1.5)
            # Add legend entry with uniform size (no error bars)
            ax_bottom_right.scatter([], [], s=60, color=month_colors[month], alpha=0.85, label=month_names[month])
        
        ax_bottom_right.legend(fontsize=7, loc='upper left', ncol=3, framealpha=0.9)
    
    else:
        # By YEAR
        years = sorted(data['year'].unique())
        
        # First pass: find max visits per year for sizing
        for year in years:
            year_data = data[data['year'] == year]
            if len(year_data) > 0:
                total_visits = year_data['n_visits'].sum()
                max_visits_br = max(max_visits_br, total_visits)
        
        for year in years:
            year_data = data[data['year'] == year]
            if len(year_data) == 0:
                continue
            
            x = year_data['real_value'].values
            y = year_data['pred_value'].values
            w = year_data['n_visits'].values.astype(float)
            w_sum = np.sum(w)
            
            # Weighted mean (centroid)
            mean_x = np.sum(w * x) / w_sum
            mean_y = np.sum(w * y) / w_sum
            
            # Weighted std (for error bars)
            std_x = np.sqrt(np.sum(w * (x - mean_x)**2) / w_sum)
            std_y = np.sqrt(np.sum(w * (y - mean_y)**2) / w_sum)
            
            # Dot size proportional to total visits
            size = 80 + 200 * (w_sum / max_visits_br)
            
            color = year_colors.get(year, "#333333")
            
            # Plot the actual data point with variable size
            ax_bottom_right.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, 
                              fmt='o', markersize=np.sqrt(size/np.pi), capsize=4, capthick=1.5,
                              color=color, ecolor=color,
                              alpha=0.85, linewidth=1.5)
            # Add legend entry with uniform size (no error bars)
            ax_bottom_right.scatter([], [], s=60, color=color, alpha=0.85, label=str(year))
        
        ax_bottom_right.legend(fontsize=9, loc='upper left', ncol=2, framealpha=0.9)
    
    # Global ODR line
    x_fit_br = np.array([0, plot_lim_br])
    y_fit_br = m_odr * x_fit_br + b_odr
    ax_bottom_right.plot(x_fit_br, y_fit_br, 'r-', linewidth=2, label='Global fit')
    
    # Identity line
    ax_bottom_right.plot([0, plot_lim_br], [0, plot_lim_br], 'k--', alpha=0.4, linewidth=1.5)
    
    ax_bottom_right.set_xlabel('Observed Prevalence', fontsize=13)
    ax_bottom_right.set_ylabel('Predicted Prevalence', fontsize=13)
    ax_bottom_right.grid(alpha=0.3)
    ax_bottom_right.set_xlim(0, plot_lim_br)
    ax_bottom_right.set_ylim(0, plot_lim_br)
    ax_bottom_right.set_aspect('equal', adjustable='box')
    
    for ax in (ax_error, ax_kde, ax_zone, ax_bottom_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Place panel labels in figure coordinates so they are consistent across
    # axes of different widths (the right square axes are narrower than the left ones).
    fig.canvas.draw()  # needed to finalise axes positions
    for ax, label in [(ax_error, 'a)'), (ax_kde, 'b)'), (ax_zone, 'c)'), (ax_bottom_right, 'd)')]:
        bbox = ax.get_position()   # axes position in figure fraction
        fig.text(bbox.x0, bbox.y1 + 0.01, label, fontsize=13, fontweight='bold',
                 va='bottom', ha='left', transform=fig.transFigure)

    if SAVE_FIGURES:
        suffix = f"{mse_mode}_{BOTTOM_RIGHT_MODE}"
        plt.savefig(FIGURES_DIR / f"Validation_Panel_{suffix}.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURES_DIR / f"Validation_Panel_{suffix}.pdf", bbox_inches='tight')
        print(f"Saved: Validation_Panel_{suffix}.png and .pdf")
    
    # Return statistics for printing
    n_total = len(data)
    stats = {
        'slope': m_odr,
        'intercept': b_odr,
        'r_squared': r_squared,
        'kde_overlap': kde_overlap,
        'n_outliers_real': n_outliers_real,
        'n_outliers_pred': n_outliers_pred,
        'n_total': n_total,
    }
    
    return fig, stats

# ============================================================================
# MAIN
# ============================================================================

print("\n" + "="*60)
print("VALIDATION PANEL PLOT")
print("="*60 + "\n")

# Prepare data
print("Preparing ASPB zone-month data...")
aspb_zm = prepare_aspb_zone_month()
print(f"  Zone-month counts: {len(aspb_zm)}")

# Run simulation
solution, t_reformas = run_simulation_realistic()

# Compute zone-month integrals
print("Computing zone-month predictions...")
data_zm = compute_zone_month_integrals_normalized(solution, aspb_zm, t_reformas, MSE_MODE)
print(f"  {len(data_zm)} zone-months matched\n")

# Create panel
print("Creating validation panel...")
fig, stats = create_validation_panel(data_zm, MSE_MODE)

# Print statistics
print("\n" + "="*60)
print("VALIDATION STATISTICS")
print("="*60)
print(f"\nGlobal ODR Fit (weighted by n_visits):")
print(f"  Slope:     {stats['slope']:.4f}")
print(f"  Intercept: {stats['intercept']:.4f}")
print(f"  R²:        {stats['r_squared']:.4f}")
dist_label = "Histogram" if DIST_MODE == "histogram" else "KDE"
print(f"\n{dist_label} Distribution Overlap:")
print(f"  Overlap Coefficient: {stats['kde_overlap']:.2%}")
print(f"  Visualization clipped at {CLIP_THRESHOLD}:")
print(f"    - {stats['n_outliers_real']} observed zone-months outside plot range")
print(f"    - {stats['n_outliers_pred']} predicted zone-months outside plot range")
print("="*60)

print("\nDone!")
plt.show()
