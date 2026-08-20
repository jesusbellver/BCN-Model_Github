import os
import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
from pathlib import Path
import pickle
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import cma
from scipy.stats import qmc
import functools
print = functools.partial(print, flush=True)  # Flush for cluster

# ============================================================================
# SETUP & DIRECTORIES
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Dades_netes"
PARAM_DIR = SCRIPT_DIR / "Parameters"
RESULTS_DIR = SCRIPT_DIR / "Parameter_estimation"
RESULTS_DIR.mkdir(exist_ok=True)

d0 = datetime(2019, 1, 1)
tf_date = datetime(2024, 12, 31)
n_days = (tf_date - d0).days + 1  # 2192 days

# ============================================================================
# LOAD DATA
# ============================================================================

rainfall_data = xr.open_dataset(PARAM_DIR / "Rainfall.nc")['data']
rainfall_21_data = xr.open_dataset(PARAM_DIR / "Rainfall_21.nc")['data']
r0_data = xr.open_dataset(PARAM_DIR / "R0_both.nc")['data']

with open(PARAM_DIR / "D_Matrix", 'rb') as f:
    D = pickle.load(f)
with open(PARAM_DIR / "X_coords", 'rb') as f:
    X_coords_dict = pickle.load(f)
with open(PARAM_DIR / "Y_coords", 'rb') as f:
    Y_coords_dict = pickle.load(f)
with open(PARAM_DIR / "ASPB_Interventions", 'rb') as f:
    interventions_dict = pickle.load(f)

unique_bs = pd.read_csv(DATA_DIR / "Unique_Items.csv")
revisions = pd.read_csv(DATA_DIR / "Revisions.csv")

n_drains = D.shape[0]

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
print(f"Zones: {zone_names}")

X_emb = np.array([X_coords_dict[i] for i in range(n_drains)])
Y_emb = np.array([Y_coords_dict[i] for i in range(n_drains)])

unique_bs['data_reforma'] = pd.to_datetime(unique_bs['data_reforma'], errors='coerce')
unique_bs['data_reforma'] = unique_bs['data_reforma'].fillna(pd.Timestamp("2025-01-01"))
t_reformas = np.array([(dt - d0).days for dt in unique_bs['data_reforma']], dtype=np.float64)

# Precompute zone membership arrays for fast zone-restricted M matrix
# zone_members[zone_starts[z]:zone_starts[z+1]] gives drain indices in zone z
_zone_lists = [[] for _ in range(n_zones)]
for i in range(n_drains):
    _zone_lists[drain_to_zone[i]].append(i)
zone_members = np.array([idx for zl in _zone_lists for idx in zl], dtype=np.int32)
zone_starts = np.zeros(n_zones + 1, dtype=np.int32)
for z in range(n_zones):
    zone_starts[z + 1] = zone_starts[z] + len(_zone_lists[z])
del _zone_lists

# Per-drain baseline probability (0.1% per drain)
BASELINE = 0.001

print(f"Loaded {n_drains} drains, {n_days} days simulation")

# ============================================================================
# CLIMATE ACCESS
# ============================================================================

def interpolate_all_locations(data_array, lons, lats):
    lons_da = xr.DataArray(lons, dims='points')
    lats_da = xr.DataArray(lats, dims='points')
    result = data_array.interp(longitude=lons_da, latitude=lats_da)
    return result.values

# ============================================================================
# NUMBA-COMPILED CORE
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
def simulate_one_parameter_set(
    u0, n_drains, n_days, M, t_reformas, w_min, w_flush, B, D_param,
    dt, rainfall_24h, rainfall_21d, r0_start,
    interventions_list, intervention_starts,
    drain_to_zone, zone_members, zone_starts, baseline
):
    """
    RK4 simulation with ASPB interventions.
    
    Logic (priority order):
    1. Reformed drains (day > t_reformas): u=0, no dynamics
    2. Flushed (24h rain > w_flush): u=baseline, cancels intervention
    3. Under intervention (within 49 days, not cancelled): u=baseline
    4. Dry (21d rain < w_min): u=baseline
    5. Otherwise: can evolve
    """
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
        r0_day = r0_start[day]
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
# ZONE-MONTH PREPROCESSING
# ============================================================================

def preprocess_zone_months(cases_df, zone_name_to_idx, d0, n_days):
    """
    Group observations by (zone, year, month).
    Returns list of tuples: (zone_idx, day_start, day_end, n_days_month, n_visits, n_positives)
    """
    cases_df = cases_df.copy()
    cases_df['year'] = cases_df['Fecha'].dt.year.astype(int)
    cases_df['month'] = cases_df['Fecha'].dt.month.astype(int)
    
    zone_month_data = []
    for (zone, year, month), group in cases_df.groupby(['nom_zr', 'year', 'month']):
        n_visits = len(group)
        n_positives = int(group['activitat'].sum())
        zone_idx = zone_name_to_idx[zone]
        
        month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
        if month == 12:
            month_end = pd.Timestamp(year=int(year) + 1, month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = pd.Timestamp(year=int(year), month=int(month) + 1, day=1) - pd.Timedelta(days=1)
        
        day_start = max(0, (month_start - d0).days)
        day_end = min(n_days - 1, (month_end - d0).days)
        
        if day_start > day_end:
            continue
        
        n_days_month = day_end - day_start + 1
        zone_month_data.append((zone_idx, day_start, day_end, n_days_month, n_visits, n_positives))
    
    print(f"  Total zone-months: {len(zone_month_data)}")
    n_positive = sum(1 for _, _, _, _, _, n_pos in zone_month_data if n_pos > 0)
    n_zero = sum(1 for _, _, _, _, _, n_pos in zone_month_data if n_pos == 0)
    print(f"  Zone-months with activity > 0: {n_positive}")
    print(f"  Zone-months with activity = 0: {n_zero}")
    
    return zone_month_data


# ============================================================================
# CLIMATE & INTERVENTIONS PREP
# ============================================================================

def prepare_climate_arrays():
    print("  Interpolating climate data...")
    
    rainfall_24h = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    r0_start = interpolate_all_locations(r0_data, X_emb, Y_emb)
    
    rainfall_24h = np.nan_to_num(rainfall_24h, nan=0.0)
    rainfall_21d = np.nan_to_num(rainfall_21d, nan=0.0)
    r0_start = np.nan_to_num(r0_start, nan=0.00001)
    
    if rainfall_24h.shape[0] < n_days:
        pad_days = n_days - rainfall_24h.shape[0]
        rainfall_24h = np.vstack([rainfall_24h, np.zeros((pad_days, n_drains))])
        rainfall_21d = np.vstack([rainfall_21d, np.zeros((pad_days, n_drains))])
        r0_start = np.vstack([r0_start, np.full((pad_days, n_drains), 0.00001)])
    
    print(f"  Climate arrays ready: {rainfall_24h.shape}")
    return r0_start[:n_days], rainfall_24h[:n_days], rainfall_21d[:n_days]


def prepare_interventions():
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
# ZONE HELPERS
# ============================================================================

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
def compute_weighted_active_drains(drain_to_zone, t_reformas, zone_idx, day_start, day_end):
    """
    Average active drains in a zone over [day_start, day_end],
    weighted by the fraction of time each drain is active.
    """
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


# ============================================================================
# LOSS FUNCTION (weighted MSE on prevalence) - COMMENTED OUT, replaced by NLL
# ============================================================================

# @jit(nopython=True)
# def compute_weighted_mse_prevalence(zone_sums, zone_month_data, drain_to_zone, t_reformas):
#     """
#     Weighted MSE loss function for zone-month prevalence fitting.
#     
#     - Real prevalence: n_positives / n_visits
#     - Predicted prevalence: zone_integral / (n_days_month × n_active_weighted)
#     - Weighting: by n_visits (more samples = more reliable estimate)
#     """
#     total_loss = 0.0
#     total_weight = 0.0
#     
#     for i in range(len(zone_month_data)):
#         zone_idx = int(zone_month_data[i][0])
#         day_start = int(zone_month_data[i][1])
#         day_end = int(zone_month_data[i][2])
#         n_days_month = int(zone_month_data[i][3])
#         n_visits = int(zone_month_data[i][4])
#         n_positives = int(zone_month_data[i][5])
#         
#         if n_visits == 0:
#             continue
#         
#         integral = compute_zone_month_integral(zone_sums, zone_idx, day_start, day_end)
#         n_active = compute_weighted_active_drains(drain_to_zone, t_reformas, zone_idx, day_start, day_end)
#         
#         pred_prevalence = integral / (n_days_month * n_active)
#         real_prevalence = n_positives / n_visits
#         
#         error = real_prevalence - pred_prevalence
#         weight = float(n_visits)
#         total_loss += weight * error * error
#         total_weight += weight
#     
#     return total_loss / total_weight if total_weight > 0 else 0.0


# ============================================================================
# LOSS FUNCTION (Binomial Negative Log-Likelihood with class weighting)
# ============================================================================

@jit(nopython=True)
def compute_binomial_nll(zone_sums, zone_month_data, drain_to_zone, t_reformas, 
                         weight_positive, weight_zero):
    """
    Binomial Negative Log-Likelihood for zone-month prevalence fitting.
    
    NLL = -sum[ w_i * (n_positives * log(p_pred) + (n_visits - n_positives) * log(1 - p_pred)) ]
    
    With class weighting:
    - Zone-months with n_positives > 0 get weight = weight_positive
    - Zone-months with n_positives = 0 get weight = weight_zero
    - Weights are inversely proportional to class frequency (rarer class = higher weight)
    
    This helps the optimizer focus more on capturing the peaks (positive zone-months)
    which are rarer but more informative.
    
    Note: We clamp p_pred to [eps, 1-eps] to avoid log(0) = -inf
    """
    eps = 1e-7
    total_nll = 0.0
    total_weight = 0.0
    
    for i in range(len(zone_month_data)):
        zone_idx = int(zone_month_data[i][0])
        day_start = int(zone_month_data[i][1])
        day_end = int(zone_month_data[i][2])
        n_days_month = int(zone_month_data[i][3])
        n_visits = int(zone_month_data[i][4])
        n_positives = int(zone_month_data[i][5])
        
        if n_visits == 0:
            continue
        
        # Compute predicted prevalence from simulation
        integral = compute_zone_month_integral(zone_sums, zone_idx, day_start, day_end)
        n_active = compute_weighted_active_drains(drain_to_zone, t_reformas, zone_idx, day_start, day_end)
        
        if n_active < 0.01:
            continue
        
        p_pred = integral / (n_days_month * n_active)
        
        # Clamp to avoid log(0)
        if p_pred < eps:
            p_pred = eps
        elif p_pred > 1.0 - eps:
            p_pred = 1.0 - eps
        
        # Binomial log-likelihood (negative, to minimize)
        log_p = np.log(p_pred)
        log_1mp = np.log(1.0 - p_pred)
        nll_i = -(n_positives * log_p + (n_visits - n_positives) * log_1mp)
        
        # Apply class weight
        if n_positives > 0:
            w = weight_positive
        else:
            w = weight_zero
        
        total_nll += w * nll_i
        total_weight += w
    
    # Return weighted average NLL
    if total_weight > 0:
        return total_nll / total_weight
    else:
        return 0.0


# ============================================================================
# FIXED PARAMETER SUPPORT
# ============================================================================

def create_param_mapping(all_bounds):
    """
    Create mapping between free and fixed parameters.
    
    Use tuple (lower, upper) for free parameters, scalar for fixed.
    Example: (0.005, 0.15) -> free, 12.0 -> fixed at 12.0
    """
    free_indices = []
    fixed_indices = []
    fixed_values = {}
    free_bounds = []
    
    for i, b in enumerate(all_bounds):
        if isinstance(b, tuple) and len(b) == 2:
            free_indices.append(i)
            free_bounds.append(b)
        else:
            fixed_indices.append(i)
            fixed_values[i] = float(b)
    
    return free_indices, fixed_indices, fixed_values, free_bounds


def expand_params(reduced_params, free_indices, fixed_values, n_total):
    """Expand reduced parameters (free only) to full parameter vector."""
    full_params = np.zeros(n_total)
    for i, idx in enumerate(free_indices):
        full_params[idx] = reduced_params[i]
    for idx, val in fixed_values.items():
        full_params[idx] = val
    return full_params


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

_worker_data = {}

def _init_worker(r0_start, rainfall_24h, rainfall_21d, interventions_list,
                 intervention_starts, zone_month_data, dt, D_mat, drain_to_zone_arr,
                 zone_members_arr, zone_starts_arr,
                 n_drains_val, n_days_val, n_zones_val, t_reformas_arr, 
                 baseline_val, u0_arr, free_indices, fixed_values, n_total_params,
                 weight_positive, weight_zero):
    global _worker_data
    _worker_data['r0_start'] = r0_start
    _worker_data['rainfall_24h'] = rainfall_24h
    _worker_data['rainfall_21d'] = rainfall_21d
    _worker_data['interventions_list'] = interventions_list
    _worker_data['intervention_starts'] = intervention_starts
    _worker_data['zone_month_data'] = zone_month_data
    _worker_data['dt'] = dt
    _worker_data['D'] = D_mat
    _worker_data['drain_to_zone'] = drain_to_zone_arr
    _worker_data['zone_members'] = zone_members_arr
    _worker_data['zone_starts'] = zone_starts_arr
    _worker_data['n_drains'] = n_drains_val
    _worker_data['n_days'] = n_days_val
    _worker_data['n_zones'] = n_zones_val
    _worker_data['t_reformas'] = t_reformas_arr
    _worker_data['baseline'] = baseline_val
    _worker_data['u0'] = u0_arr
    _worker_data['free_indices'] = free_indices
    _worker_data['fixed_values'] = fixed_values
    _worker_data['n_total_params'] = n_total_params
    _worker_data['weight_positive'] = weight_positive
    _worker_data['weight_zero'] = weight_zero


def run_one_param_set(reduced_params):
    wd = _worker_data
    n_z = wd['n_zones']
    
    params = expand_params(reduced_params, wd['free_indices'], 
                           wd['fixed_values'], wd['n_total_params'])
    
    alphas = np.array(params[:n_z], dtype=np.float64)
    w_min = float(params[n_z])
    w_flush = float(params[n_z + 1])
    B = float(params[n_z + 2])
    D_param = float(params[n_z + 3])
    
    M = generate_M(wd['D'], alphas, wd['drain_to_zone'],
                    wd['zone_members'], wd['zone_starts'], wd['n_drains'])
    
    solution = simulate_one_parameter_set(
        wd['u0'], wd['n_drains'], wd['n_days'], M, wd['t_reformas'],
        w_min, w_flush, B, D_param, wd['dt'],
        wd['rainfall_24h'], wd['rainfall_21d'], wd['r0_start'],
        wd['interventions_list'], wd['intervention_starts'],
        wd['drain_to_zone'], wd['zone_members'], wd['zone_starts'],
        wd['baseline']
    )
    
    zone_sums = compute_zone_day_sums(solution, wd['drain_to_zone'], n_z)
    loss = compute_binomial_nll(zone_sums, wd['zone_month_data'], 
                                wd['drain_to_zone'], wd['t_reformas'],
                                wd['weight_positive'], wd['weight_zero'])
    
    # Penalty for invalid results (NLL scale is different from MSE)
    if np.isnan(loss) or np.isinf(loss) or loss > 100.0:
        loss = 100.0
    
    return loss


def prepare_worker_data(cases):
    """Prepare all data needed for optimization."""
    print(f"Preparing climate data...")
    r0_start, rainfall_24h, rainfall_21d = prepare_climate_arrays()
    
    print(f"Preparing interventions...")
    interventions_list, intervention_starts = prepare_interventions()
    
    print(f"Pre-processing zone-month data...")
    zone_month_data = preprocess_zone_months(cases, zone_name_to_idx, d0, n_days)
    zone_month_array = np.array(zone_month_data, dtype=np.float64)
    
    # Compute class weights
    n_positive_zm = sum(1 for row in zone_month_data if row[5] > 0)
    n_zero_zm = sum(1 for row in zone_month_data if row[5] == 0)
    n_total_zm = n_positive_zm + n_zero_zm
    
    if WEIGHTING == "balanced":
        # Weight = (total / class_count) / 2, so weights sum to ~total
        # This makes positive zone-months (rarer) have higher weight
        weight_positive = (n_total_zm / n_positive_zm) / 2.0 if n_positive_zm > 0 else 1.0
        weight_zero = (n_total_zm / n_zero_zm) / 2.0 if n_zero_zm > 0 else 1.0
    elif WEIGHTING == "uniform":
        weight_positive = 1.0
        weight_zero = 1.0
    else:
        raise ValueError(f"WEIGHTING must be 'balanced' or 'uniform', got '{WEIGHTING}'")
    
    print(f"  Class weighting for NLL (mode: {WEIGHTING}):")
    print(f"    Positive zone-months: {n_positive_zm} -> weight = {weight_positive:.3f}")
    print(f"    Zero zone-months: {n_zero_zm} -> weight = {weight_zero:.3f}")
    print(f"    Weight ratio (positive/zero): {weight_positive/weight_zero:.2f}x")
    
    print(f"Computing initial conditions...")
    
    # Per-drain baseline: 0.001 for active drains, 0 for reformed
    u0 = np.zeros(n_drains)
    for i in range(n_drains):
        if 0 <= t_reformas[i]:  # Active at day 0
            u0[i] = BASELINE
    
    print(f"  Per-drain baseline: {BASELINE}\n")
    
    return (r0_start, rainfall_24h, rainfall_21d, interventions_list,
            intervention_starts, zone_month_array, u0,
            weight_positive, weight_zero)


# ============================================================================
# BASELINE NLL COMPUTATION (for reference)
# ============================================================================

def compute_baseline_nll_values(zone_month_data, weight_positive, weight_zero):
    """
    Compute baseline NLL values for reference (using same class weighting as optimizer):
    1. Predicting all zeros (p_pred = epsilon)
    2. Predicting global mean prevalence everywhere
    3. Predicting per-zone mean prevalence
    
    This gives a sense of what NLL values mean and what "beating the baseline" looks like.
    """
    eps = 1e-7
    
    # Gather all data
    total_positives = 0
    total_visits = 0
    zone_positives = {}
    zone_visits = {}
    
    for row in zone_month_data:
        zone_idx = int(row[0])
        n_visits = int(row[4])
        n_positives = int(row[5])
        
        if n_visits == 0:
            continue
        
        total_positives += n_positives
        total_visits += n_visits
        
        if zone_idx not in zone_positives:
            zone_positives[zone_idx] = 0
            zone_visits[zone_idx] = 0
        zone_positives[zone_idx] += n_positives
        zone_visits[zone_idx] += n_visits
    
    global_mean_prev = total_positives / total_visits if total_visits > 0 else 0.0
    
    # Compute weighted NLL for each baseline
    nll_all_zeros = 0.0
    nll_global_mean = 0.0
    nll_zone_mean = 0.0
    total_weight = 0.0
    
    for row in zone_month_data:
        zone_idx = int(row[0])
        n_visits = int(row[4])
        n_positives = int(row[5])
        
        if n_visits == 0:
            continue
        
        n_negatives = n_visits - n_positives
        w = weight_positive if n_positives > 0 else weight_zero
        total_weight += w
        
        # Baseline 1: Predict p=eps (essentially 0)
        p_zero = eps
        nll_all_zeros += w * (-(n_positives * np.log(p_zero) + n_negatives * np.log(1 - p_zero)))
        
        # Baseline 2: Predict global mean prevalence
        p_global = max(eps, min(1 - eps, global_mean_prev))
        nll_global_mean += w * (-(n_positives * np.log(p_global) + n_negatives * np.log(1 - p_global)))
        
        # Baseline 3: Predict zone-specific mean prevalence
        if zone_visits[zone_idx] > 0:
            p_zone = zone_positives[zone_idx] / zone_visits[zone_idx]
        else:
            p_zone = global_mean_prev
        p_zone = max(eps, min(1 - eps, p_zone))
        nll_zone_mean += w * (-(n_positives * np.log(p_zone) + n_negatives * np.log(1 - p_zone)))
    
    # Weighted average
    nll_all_zeros /= total_weight
    nll_global_mean /= total_weight
    nll_zone_mean /= total_weight
    
    print(f"\n{'='*70}")
    print(f"BASELINE NLL VALUES (weighted, for reference)")
    print(f"{'='*70}")
    print(f"  Global mean prevalence: {global_mean_prev:.4f} ({total_positives}/{total_visits})")
    print(f"\n  Baseline weighted NLL (avg per zone-month):")
    print(f"    1. Predict all zeros (p={eps}):     NLL = {nll_all_zeros:.4f}")
    print(f"    2. Predict global mean ({global_mean_prev:.4f}): NLL = {nll_global_mean:.4f}")
    print(f"    3. Predict zone means:              NLL = {nll_zone_mean:.4f}")
    print(f"\n  Your optimizer should aim to BEAT these baselines!")
    print(f"  NLL < {nll_zone_mean:.4f} means you're doing better than zone means.")
    print(f"{'='*70}\n")
    
    return {
        'nll_all_zeros': nll_all_zeros,
        'nll_global_mean': nll_global_mean,
        'nll_zone_mean': nll_zone_mean,
        'global_mean_prev': global_mean_prev,
    }


# ============================================================================
# MAIN
# ============================================================================

species = "both"
MODE = "fit"  # "test" for quick debugging (~15-20 min) or "fit" for full optimization (~6h)
WEIGHTING = "balanced"  # "balanced" = inverse class-frequency weighting, "uniform" = all weights = 1

print(f"\n{'='*70}")
print(f"Simulating species: {species} (MODE: {MODE}, WEIGHTING: {WEIGHTING})")
print(f"{'='*70}\n")

# Prepare case data
col_name = f"activitat_{species}"
cases = revisions[['Fecha', 'nom_zr', 'id_item', col_name]].copy()
cases.rename(columns={col_name: 'activitat'}, inplace=True)
cases['Fecha'] = pd.to_datetime(cases['Fecha'])
cases['day_idx'] = (cases['Fecha'] - d0).dt.days

# === TUNABLE PARAMETERS ===
dt = 0.15

# Per-zone alpha bounds (7 zones) - order matches zone_names from Unique_Items.csv
alpha_bounds = [
    (0.09, 0.14),  # alpha_0: Parc de la Ciutadella       (n=50, mean dist=185m)
    (0.18, 0.25),  # alpha_1: Parc del Turó de la Peira   (n=61, mean dist=168m)
    (0.04, 0.7),  # alpha_2: Jardins Mossèn Cinto Verd.  (n=27, mean dist=91m)
    (0.17, 0.25),  # alpha_3: Jardins del Teatre Grec     (n=8,  mean dist=68m)
    (0.013, 0.03),  # alpha_4: Jardins del Turó del Putxet (n=34, mean dist=96m)
    (0.19, 0.25),  # alpha_5: Jardins de Vil·la Amèlia    (n=96, mean dist=119m)
    (0.13, 0.18),  # alpha_6: Parc de la Guineueta        (n=30, mean dist=198m)
]

if len(alpha_bounds) != n_zones:
    raise ValueError("alpha_bounds list must have one entry per zone")


global_bounds = [
    (8.5, 11.0),    # w_min  (21d drying threshold, mm)
    (15.0, 20.0),    # w_flush (24h flushing threshold, mm)
    (1e-4, 1.4e-4),   # B (birth rate, calibrated for r0~700)
    (5e-6, 4e-5),    # D (death rate)
]



all_bounds = alpha_bounds + global_bounds

free_indices, fixed_indices, fixed_values, free_bounds = create_param_mapping(all_bounds)
n_total_params = len(all_bounds)
n_free_params = len(free_bounds)

print(f"Total parameters: {n_total_params} ({n_zones} alphas + 4 global)")
print(f"  Free parameters: {n_free_params}")
print(f"  Fixed parameters: {len(fixed_indices)} at indices {fixed_indices}")
for idx in fixed_indices:
    print(f"    param[{idx}] = {fixed_values[idx]}")

# ============================================================================
# MODE-SPECIFIC HYPERPARAMETERS
# ============================================================================

if MODE == "test":
    print(f"\n{'*'*70}")
    print(f"TEST MODE - Sanity check (~20-30 minutes)")
    print(f"{'*'*70}\n")
    
    # 11D problem: CMA-ES default popsize ~ 4 + floor(3*ln(11)) = 11
    # Need at least ~100 generations to get meaningful convergence per run
    # 4 starts × 1500 evals = 6000 exploration + 1 refinement × 2000 = 8000 total
    n_starts = 4
    maxfevals_per_start = 1500  # ~136 generations at popsize=11
    popsize_start = 11          # matches CMA-ES default for 11D
    sigma0_factor = 0.3
    tolfun = 1e-6
    tolx = 1e-4
    n_refine_runs = 1
    maxfevals_refine = 2000
    popsize_refine = 20
    
    print(f"TEST CONFIGURATION:")
    print(f"  Exploration: {n_starts} starts × {maxfevals_per_start} evals = {n_starts*maxfevals_per_start} total")
    print(f"  Refinement: {n_refine_runs} × {maxfevals_refine} evals = {n_refine_runs*maxfevals_refine} total")
    print(f"  Total evaluations: {n_starts*maxfevals_per_start + n_refine_runs*maxfevals_refine}")
    print(f"  Expected time: ~20-30 minutes\n")

elif MODE == "fit":
    print(f"\n{'*'*70}")
    print(f"FIT MODE - Full optimization (~5-6 hours on cluster)")
    print(f"{'*'*70}\n")
    
    # 11D problem: 20 diverse starts × 3000 evals (~272 generations each) for thorough exploration
    # Top 3 refined with 6000 evals (~272 generations at popsize=22) for tight convergence
    #n_starts = 20
    n_starts = 15
    maxfevals_per_start = 3000  # ~272 generations at popsize=11
    popsize_start = 15          # slightly above default for better landscape coverage
    sigma0_factor = 0.35
    tolfun = 1e-8
    tolx = 1e-6
    # n_refine_runs = 3
    # maxfevals_refine = 6000
    # popsize_refine = 30
    n_refine_runs = 2
    maxfevals_refine = 3000
    popsize_refine = 15
    
    print(f"FIT CONFIGURATION:")
    print(f"  Exploration: {n_starts} starts × {maxfevals_per_start} evals = {n_starts*maxfevals_per_start} total")
    print(f"  Refinement: {n_refine_runs} × {maxfevals_refine} evals = {n_refine_runs*maxfevals_refine} total")
    print(f"  Total evaluations: {n_starts*maxfevals_per_start + n_refine_runs*maxfevals_refine}")
    print(f"  Expected time: ~5-6 hours on cluster\n")

else:
    raise ValueError(f"MODE must be 'test' or 'fit', got '{MODE}'")

# Prepare data
worker_data_tuple = prepare_worker_data(cases)

# Compute baseline NLL values for reference (using same weights)
baseline_nll = compute_baseline_nll_values(
    worker_data_tuple[5],   # zone_month_array
    worker_data_tuple[7],   # weight_positive
    worker_data_tuple[8]    # weight_zero
)

# ============================================================================
# MULTI-START CMA-ES OPTIMIZATION
# ============================================================================

# Read worker count from Slurm (or fall back to all available CPUs)
workers = int(os.environ.get("SLURM_NTASKS", os.cpu_count()))
print(f"Workers: {workers} (from SLURM_NTASKS or cpu_count)")
print(f"Starting CMA-ES optimization with dt={dt}...\n")

# Generate initial guesses using Latin Hypercube Sampling
print(f"Generating {n_starts} initial guesses using Latin Hypercube Sampling...")
sampler = qmc.LatinHypercube(d=n_free_params, seed=None, scramble=True)
initial_samples = sampler.random(n=n_starts)

initial_guesses = []
for sample in initial_samples:
    x0 = np.array([free_bounds[i][0] + sample[i] * (free_bounds[i][1] - free_bounds[i][0]) 
                   for i in range(n_free_params)])
    perturbation = np.random.uniform(-0.05, 0.05, n_free_params) * np.array([b[1] - b[0] for b in free_bounds])
    x0 = np.clip(x0 + perturbation, [b[0] for b in free_bounds], [b[1] for b in free_bounds])
    initial_guesses.append(x0)

print(f"Multi-start config: {n_starts} runs, {maxfevals_per_start} evals each")
print(f"Population size: {popsize_start}\n")

# Initialize worker pool
with ProcessPoolExecutor(
    max_workers=workers,
    initializer=_init_worker,
    initargs=(worker_data_tuple[0], worker_data_tuple[1], worker_data_tuple[2],
              worker_data_tuple[3], worker_data_tuple[4], worker_data_tuple[5],
              dt, D, drain_to_zone, zone_members, zone_starts,
              n_drains, n_days, n_zones, t_reformas,
              BASELINE, worker_data_tuple[6],
              free_indices, fixed_values, n_total_params,
              worker_data_tuple[7], worker_data_tuple[8])  # weight_positive, weight_zero
) as executor:
    
    all_results = []
    
    # Exploration phase
    for run_idx, x0 in enumerate(initial_guesses):
        print(f"\n{'='*70}")
        print(f"EXPLORATION RUN {run_idx + 1}/{n_starts}")
        print(f"{'='*70}")
        print(f"Initial guess: {x0}")
        
        sigma0_value = sigma0_factor * np.mean([(b[1] - b[0]) for b in free_bounds])
        
        opts = {
            'bounds': [list(b) for b in zip(*free_bounds)],
            'maxfevals': maxfevals_per_start,
            'popsize': popsize_start,
            'tolfun': tolfun,
            'tolx': tolx,
            'verb_disp': 0,
            'verb_log': 0,
            'verbose': -1,
        }
        
        es = cma.CMAEvolutionStrategy(x0, sigma0_value, opts)
        
        while not es.stop():
            solutions = es.ask()
            fitness = list(executor.map(run_one_param_set, solutions))
            es.tell(solutions, fitness)
        
        print(f"Run {run_idx + 1} complete: loss = {es.result.fbest:.9f}")
        print(f"Best params: {es.result.xbest}")
        
        all_results.append({
            'run_idx': run_idx,
            'x0': x0,
            'xbest': es.result.xbest.copy(),
            'fbest': es.result.fbest,
        })
    
    # Select top solutions for refinement
    all_results.sort(key=lambda r: r['fbest'])
    top_solutions = all_results[:n_refine_runs]
    
    print(f"\n\n{'='*70}")
    print(f"EXPLORATION PHASE COMPLETE")
    print(f"{'='*70}")
    print(f"\nTop {n_refine_runs} results selected for refinement:")
    for i, r in enumerate(top_solutions):
        print(f"  {i+1}. Run {r['run_idx']+1}: loss = {r['fbest']:.9f}")
    
    # Refinement phase
    refinement_results = []
    
    if n_refine_runs > 0:
        print(f"\n\n{'='*70}")
        print(f"REFINEMENT PHASE - Refining top {n_refine_runs} solutions")
        print(f"{'='*70}")
        
        for refine_idx, candidate in enumerate(top_solutions):
            print(f"\n{'-'*70}")
            print(f"Refining solution {refine_idx + 1}/{n_refine_runs}")
            print(f"Starting loss: {candidate['fbest']:.9f}")
            print(f"{'-'*70}\n")
            
            sigma0_refine = 0.03 * np.mean([(b[1] - b[0]) for b in free_bounds])
            
            opts_refine = {
                'bounds': [list(b) for b in zip(*free_bounds)],
                'maxfevals': maxfevals_refine,
                'popsize': popsize_refine,
                'tolfun': tolfun,
                'tolx': tolx,
                'verb_disp': 0,
                'verb_log': 0,
                'verbose': -1,
            }
            
            es_refine = cma.CMAEvolutionStrategy(candidate['xbest'], sigma0_refine, opts_refine)
            
            best_loss = candidate['fbest']
            best_params = candidate['xbest'].copy()
            
            generation = 0
            while not es_refine.stop():
                solutions = es_refine.ask()
                fitness = list(executor.map(run_one_param_set, solutions))
                es_refine.tell(solutions, fitness)
                
                if es_refine.result.fbest < best_loss:
                    best_loss = es_refine.result.fbest
                    best_params = es_refine.result.xbest.copy()
                
                generation += 1
                if generation % 20 == 0:
                    print(f"  Generation {generation}: best = {best_loss:.9f}")
            
            improvement = candidate['fbest'] - best_loss
            print(f"\nRefinement {refine_idx + 1} complete: loss = {best_loss:.9f}, improvement = {improvement:.9f}")
            
            refinement_results.append({
                'refine_idx': refine_idx,
                'initial_loss': candidate['fbest'],
                'final_loss': best_loss,
                'improvement': improvement,
                'xbest': best_params,
                'covariance': es_refine.C,
            })
    
    # Select overall best
    if refinement_results:
        refinement_results.sort(key=lambda r: r['final_loss'])
        best_refined = refinement_results[0]
        best_params = best_refined['xbest']
        best_loss = best_refined['final_loss']
        initial_loss = best_refined['initial_loss']
    else:
        best_refined = all_results[0]
        best_params = best_refined['xbest']
        best_loss = best_refined['fbest']
        initial_loss = best_loss

# Expand to full params
best_params_full = expand_params(best_params, free_indices, fixed_values, n_total_params)
 
# Save results
results_file = RESULTS_DIR / f"Estimated_params_{species}_nll_{MODE}_{WEIGHTING}_wflush_3"

results = {
    'best_params': best_params_full,
    'best_params_reduced': best_params,
    'best_nll': best_loss,
    'initial_nll': initial_loss,
    'improvement': initial_loss - best_loss,
    'exploration_results': all_results,
    'top_solutions': top_solutions,
    'refinement_results': refinement_results,
    'baseline_nll': baseline_nll,
    'free_indices': free_indices,
    'fixed_indices': fixed_indices,
    'fixed_values': fixed_values,
    'n_total_params': n_total_params,
    'loss_type': 'binomial_nll',
    'weighting': WEIGHTING,
}

with open(results_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\n{'='*70}")
print(f"OPTIMIZATION COMPLETE (using Binomial NLL loss)")
print(f"{'='*70}")
print(f"Exploration: {n_starts} runs, best NLL = {all_results[0]['fbest']:.4f}")
print(f"Refinement: {n_refine_runs} runs")
print(f"Final NLL: {best_loss:.4f}")
print(f"Improvement: {initial_loss - best_loss:.4f}")
print(f"\nBaseline comparison:")
print(f"  Predict all zeros: {baseline_nll['nll_all_zeros']:.4f}")
print(f"  Predict global mean: {baseline_nll['nll_global_mean']:.4f}")
print(f"  Predict zone means: {baseline_nll['nll_zone_mean']:.4f}")
print(f"  Your model: {best_loss:.4f}")
if best_loss < baseline_nll['nll_zone_mean']:
    print(f"  --> BETTER than zone-mean baseline by {baseline_nll['nll_zone_mean'] - best_loss:.4f}")
else:
    print(f"  --> WORSE than zone-mean baseline by {best_loss - baseline_nll['nll_zone_mean']:.4f}")
print(f"\nBest parameters ({n_total_params} total):")
print(f"  {best_params_full}")
if fixed_indices:
    print(f"  (Fixed at indices {fixed_indices})")
print(f"Results saved to: {results_file}")
print(f"{'='*70}\n")
