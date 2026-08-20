"""
Schedule Optimization using Simulated Annealing with Parallel Neighbor Evaluation

Special case for beta=0: minimizes number of treatments only.
Since all single-treatment schedules have equal J, we use INT as tiebreaker.
Periodicity is fixed large (365) so each zone gets exactly one treatment.

Usage:
    python 6_Schedule_Optimization_B0.py <year>
    
Example:
    python 6_Schedule_Optimization_B0.py 2020
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import sys
import functools
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from numba import jit
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)


# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

YEAR = 2022  # Change as needed
BETA_SIM = 0.0  # Fixed for this script

print(f"Year: {YEAR}, Beta: {BETA_SIM}")


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Dades_netes"
PARAM_DIR = SCRIPT_DIR / "Parameters"
RESULTS_DIR = SCRIPT_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# ZONE COLORS (same order as 4_Validation_plots.py)
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
# MODEL PARAMETERS (from 4_Validation_plots.py)
# ============================================================================

ALPHAS = np.array([0.088, 0.097, 0.069, 0.14, 0.017, 0.19, 0.086], dtype=np.float64)
W_MIN = 11.4
W_FLUSH = 9.6
B = 1.9e-4
D_PARAM = 1e-5

DT = 0.30  #RK4 integration step
BASELINE = 0.001  # Per-drain baseline probability



# ============================================================================
# ANNEALING PARAMETERS
# ============================================================================

# Beta=0 uses INT as Metropolis criterion (J=PER is constant).
# Temperature has little practical effect here: most neighbor moves produce
# dINT=0 (same integral), so exp(-0/T)=1 and acceptance is ~100% regardless of T.
# Convergence is driven by landscape discreteness, not cooling schedule.
# Number of distinct INT levels varies by year (2024: ~3, 2020: ~120),
# so runtime ranges from minutes to hours depending on the year.


ITERS_ANNEALING = 5000
INIT_T = 0.072
C_T = 0.99940


# Stepped eps schedule: [(eps_t, eps_p, unproductive_limit_to_next_phase)]
# eps_p not used (periodicity fixed) but kept for compatibility
EPS_SCHEDULE = [
    (4, 2, 50),   # Phase 0: wide exploration, reduce after 50 unproductive
    (2, 1, 50),   # Phase 1: medium, reduce after 50 more unproductive
    (1, 1, 100),   # Phase 2: fine, stop after 50 unproductive
]

# Fixed periodicity for beta=0 case (one treatment per zone)
FIXED_PERIODICITY = 365

# Constraints
MAX_ZONES_DAY = 3
MAX_DAYS_WEEK = 4

# Parallelization 
N_WORKERS = 6 

N_NEIGHBORS = 30  # Scale neighbors with workers
N_RUNS = 10  # Number of independent optimization runs

print(f"Detected {N_WORKERS} available CPU cores")


# ============================================================================
# NORMALIZATION DICT — ASPB_Visits_Zona_risc (zone-level, all visits, merged <=2d)
# ============================================================================


NORM_DICT = {
    2019: {"int_ASPB": 3845.58, "per_ASPB": 90},
    2020: {"int_ASPB": 3502.45, "per_ASPB": 75},
    2021: {"int_ASPB": 274.65, "per_ASPB": 59},
    2022: {"int_ASPB": 1010.74, "per_ASPB": 67},
    2023: {"int_ASPB": 1579.46, "per_ASPB": 46},
    2024: {"int_ASPB": 717.86, "per_ASPB": 48},
}


# ============================================================================
# LOAD DATA
# ============================================================================

print(f"Loading data for year {YEAR}...")

# Climate data (full dataset, will slice for year)
rainfall_data = xr.open_dataset(PARAM_DIR / "Rainfall.nc")['data']
rainfall_21_data = xr.open_dataset(PARAM_DIR / "Rainfall_21.nc")['data']
r0_data = xr.open_dataset(PARAM_DIR / "R0_both.nc")['data']

# Static data
with open(PARAM_DIR / "D_Matrix", 'rb') as f:
    D_global = pickle.load(f)
with open(PARAM_DIR / "X_coords", 'rb') as f:
    X_coords_dict = pickle.load(f)
with open(PARAM_DIR / "Y_coords", 'rb') as f:
    Y_coords_dict = pickle.load(f)

unique_bs = pd.read_csv(DATA_DIR / "Unique_Items.csv")

n_drains = D_global.shape[0]

# Use order from zone_colors (ground truth from 4_Validation)
zone_names = list(zone_colors.keys())
n_zones = len(zone_names)
zone_name_to_idx = {name: i for i, name in enumerate(zone_names)}
drain_to_zone = np.array([zone_name_to_idx[z] for z in unique_bs['nom_zr']], dtype=np.int32)

X_emb = np.array([X_coords_dict[i] for i in range(n_drains)])
Y_emb = np.array([Y_coords_dict[i] for i in range(n_drains)])

# Parse reform dates
unique_bs['data_reforma'] = pd.to_datetime(unique_bs['data_reforma'], errors='coerce')
unique_bs['data_reforma'] = unique_bs['data_reforma'].fillna(pd.Timestamp("2030-01-01"))

# Precompute zone membership arrays for fast zone-restricted M matrix
_zone_lists = [[] for _ in range(n_zones)]
for i in range(n_drains):
    _zone_lists[drain_to_zone[i]].append(i)
zone_members = np.array([idx for zl in _zone_lists for idx in zl], dtype=np.int32)
zone_starts_arr = np.zeros(n_zones + 1, dtype=np.int32)
for z in range(n_zones):
    zone_starts_arr[z + 1] = zone_starts_arr[z] + len(_zone_lists[z])
del _zone_lists

print(f"Loaded {n_drains} drains, {n_zones} zones")


# ============================================================================
# YEAR-SPECIFIC DATA PREPARATION
# ============================================================================
# NOTE: Years are counted as days from Jan 1
# Regular years: 365 days (0 to 364), leap years: 366 days (0 to 365)
# Mosquito season: April 1 to Nov 30 each year
# Climate data starts from Jan 1, 2019 as global reference (day 0)

# Simple season bounds calculation - April 1 to Nov 30
d0_year = datetime(YEAR, 1, 1)
init_date = datetime(YEAR, 4, 1)
init_mosq_season = (init_date - d0_year).days
end_date = datetime(YEAR, 11, 30)
end_mosq_season = (end_date - d0_year).days

print(f"Mosquito season: day {init_mosq_season} (April 1) to {end_mosq_season} (Nov 30)")

# Year setup
d0_global = datetime(2019, 1, 1)  # Reference date in data files
year_start = datetime(YEAR, 1, 1)
year_end = datetime(YEAR, 12, 31)
n_days = (year_end - year_start).days + 1  # 365 or 366 days
global_offset = (year_start - d0_global).days

print(f"Preparing climate data for year {YEAR}...")

# Interpolate climate data for drain locations
lons_da = xr.DataArray(X_emb, dims='points')
lats_da = xr.DataArray(Y_emb, dims='points')

rainfall_24h_full = rainfall_data.interp(longitude=lons_da, latitude=lats_da).values
rainfall_21d_full = rainfall_21_data.interp(longitude=lons_da, latitude=lats_da).values
r0_full = r0_data.interp(longitude=lons_da, latitude=lats_da).values

# Extract year data from global dataset
rainfall_24h = np.nan_to_num(rainfall_24h_full[global_offset:global_offset + n_days], nan=0.0)
rainfall_21d = np.nan_to_num(rainfall_21d_full[global_offset:global_offset + n_days], nan=0.0)
r0_values = np.nan_to_num(r0_full[global_offset:global_offset + n_days], nan=0.00001)

# Make arrays contiguous for numba
rainfall_24h = np.ascontiguousarray(rainfall_24h)
rainfall_21d = np.ascontiguousarray(rainfall_21d)
r0_values = np.ascontiguousarray(r0_values)

print(f"Climate data shape: {rainfall_24h.shape} - covers {n_days} days")

# Reform dates: when each drain was reformed (becomes inactive)
t_reformas = np.zeros(n_drains, dtype=np.float64)
for i in range(n_drains):
    reform_date = unique_bs['data_reforma'].iloc[i]
    if pd.notna(reform_date):
        reform_year = reform_date.year
        if reform_year < YEAR or reform_year == YEAR:
            t_reformas[i] = -1  # Already reformed from day 0
        else:
            t_reformas[i] = n_days + 1  # Active all year
    else:
        t_reformas[i] = n_days + 1

t_reformas = np.ascontiguousarray(t_reformas)
print(f"Total year length: {n_days} days (365 or 366 depending on leap year)")


# ============================================================================
# NUMBA FUNCTIONS (from 4_Validation_plots.py)
# ============================================================================

@jit(nopython=True, cache=True)
def compute_du(u, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve):
    """Compute du/dt using CSR M for sequential memory access."""
    du = np.zeros(n_drains)
    for i in range(n_drains):
        if not can_evolve[i]:
            continue
        m_times_u = 0.0
        for k in range(M_row_starts[i], M_row_starts[i + 1]):
            m_times_u += M_values[k] * u[M_col_indices[k]]
        du[i] = B * r0_day[i] * m_times_u * (1.0 - u[i]) - D_param * u[i]
    return du


@jit(nopython=True, cache=True)
def simulate_with_schedule(u0, n_drains, n_days, M_values, M_col_indices, M_row_starts,
                           t_reformas, w_min, w_flush, B, D_param, dt,
                           rainfall_24h, rainfall_21d, r0_values,
                           zone_intervention_days, zone_intervention_counts,
                           drain_to_zone, baseline, n_zones):
    """
    Full simulation with scheduled interventions.
    
    zone_intervention_days: 2D array [n_zones, max_interventions] of intervention days
    zone_intervention_counts: 1D array [n_zones] of how many interventions per zone
    
    Returns total integral: sum of zone integrals over the year (like 5_Simple_plot_ONE_year.py).
    """
    u = u0.copy()
    n_substeps = int(1.0 / dt)
    can_evolve = np.zeros(n_drains, dtype=np.bool_)
    intervention_cancelled_day = np.full(n_drains, -1, dtype=np.int64)
    
    # Store zone sums per day for proper integration (like in 5_)
    zone_sums = np.zeros((n_days, n_zones))
    
    # Day 0 - compute initial zone sums
    for i in range(n_drains):
        zone_idx = drain_to_zone[i]
        zone_sums[0, zone_idx] += u[i]
    
    for day in range(1, n_days):
        # Precompute zone sums at start of day (ASPB protocol: treat only if n_z > 1)
        current_zone_sums = np.zeros(n_zones)
        for i in range(n_drains):
            current_zone_sums[drain_to_zone[i]] += u[i]
        
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            
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
            
            # 3. Check if under intervention (zone-level schedule)
            # ASPB protocol: only treat if zone has n_z(t) > 1 expected positives
            is_under_intervention = False
            if current_zone_sums[zone_idx] >= 1.0:
                n_inter = zone_intervention_counts[zone_idx]
                for k in range(n_inter):
                    inter_day = zone_intervention_days[zone_idx, k]
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
            k1 = compute_du(u, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + 0.5 * dt * k1
            k2 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + 0.5 * dt * k2
            k3 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + dt * k3
            k4 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u = u + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        # Compute zone sums for this day
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            zone_sums[day, zone_idx] += u[i]
    
    # Compute zone integrals using trapezoidal rule (exactly like 5_Simple_plot_ONE_year.py)
    zone_integrals = np.zeros(n_zones)
    for z in range(n_zones):
        for day in range(n_days - 1):
            zone_integrals[z] += 0.5 * (zone_sums[day, z] + zone_sums[day + 1, z])
    
    # Return total integral (sum of zone integrals)
    total_integral = 0.0
    for z in range(n_zones):
        total_integral += zone_integrals[z]
    
    return total_integral


# ============================================================================
# SCHEDULE GENERATION
# ============================================================================
# For beta=0: periodicity is fixed, only init_time varies

def generate_schedule(init_times, periodicity, n_zones, end_season):
    """
    Generate intervention days for each zone.
    For beta=0: periodicity=365, so each zone gets exactly one treatment at init_time.
    """
    # Each zone gets exactly 1 treatment
    schedule = np.full((n_zones, 1), -1, dtype=np.int32)
    counts = np.ones(n_zones, dtype=np.int32)
    
    for z in range(n_zones):
        schedule[z, 0] = int(init_times[z])
    
    return schedule, counts


def check_schedule(schedule, counts, n_zones, init_season, end_season, max_zones_day, max_days_week):
    """Check if schedule satisfies constraints. For beta=0: one treatment per zone."""
    # Collect all intervention days (one per zone)
    all_days = [int(schedule[z, 0]) for z in range(n_zones)]
    
    # Check max zones per day
    day_counts = {}
    for d in all_days:
        day_counts[d] = day_counts.get(d, 0) + 1
    
    for count in day_counts.values():
        if count > max_zones_day:
            return False
    
    # Check max days per week (sliding window)
    unique_days = sorted(set(all_days))
    for i, d in enumerate(unique_days):
        days_in_week = 1
        for j in range(i + 1, len(unique_days)):
            if unique_days[j] - d < 7:
                days_in_week += 1
            else:
                break
        if days_in_week > max_days_week:
            return False
    
    return True


def random_schedule(n_zones, init_season, end_season, max_zones_day, max_days_week):
    """Generate a random valid initial schedule (Julia-style ranges)."""
    periodicity = np.full(n_zones, FIXED_PERIODICITY, dtype=np.float64)
    
    for _ in range(500):
        # Julia: init_times = init_mosq_season + 5 + 31*rand() + 0.5
        init_times = init_season + 5 + 31 * np.random.rand(n_zones)
        init_times = np.round(init_times) + 0.5
        
        schedule, counts = generate_schedule(init_times, periodicity, n_zones, end_season)
        
        if check_schedule(schedule, counts, n_zones, init_season, end_season, max_zones_day, max_days_week):
            return init_times, periodicity, schedule, counts
    
    # Fallback: spread zones over different days to satisfy constraints
    init_times = np.array([init_season + 10 + z * 2 for z in range(n_zones)], dtype=np.float64) + 0.5
    schedule, counts = generate_schedule(init_times, periodicity, n_zones, end_season)
    return init_times, periodicity, schedule, counts


def update_schedule(init_times_0, periodicity_0, eps_t, eps_p, n_zones, init_season, end_season,
                    max_zones_day, max_days_week):
    """
    Generate a neighbor by perturbing init_times.
    For beta=0: periodicity is fixed, only init_times change.
    """
    for _ in range(100):
        # +eps or -eps for each zone
        signs_t = 2 * np.round(np.random.rand(n_zones)) - 1
        new_init = np.clip(init_times_0 + eps_t * signs_t, init_season + 0.5, init_season + 61 + 0.5) # 1sr April to 31st May
        
        schedule, counts = generate_schedule(new_init, periodicity_0, n_zones, end_season)
        
        if check_schedule(schedule, counts, n_zones, init_season, end_season, max_zones_day, max_days_week):
            return new_init, periodicity_0, schedule, counts
     
    schedule, counts = generate_schedule(init_times_0, periodicity_0, n_zones, end_season)
    return init_times_0, periodicity_0, schedule, counts


# ============================================================================
# COST CALCULATION - Simple weighted sum
# ============================================================================
# J = beta*INT + (1-beta)*PER
# For beta=0: J = PER (constant for single-treatment schedules)
# We use INT as tiebreaker

def cost_function(integral, schedule, counts, n_zones, beta, norm_dict, init_season, end_season):
    """
    Compute cost: J = beta*INT + (1-beta)*PER
    
    For beta=0: J = PER = n_zones/per_ASPB (constant for all schedules).
    We return INT as the metric to minimize among equal-J schedules.
    """
    INT = integral / norm_dict["int_ASPB"]
    PER = n_zones / norm_dict["per_ASPB"]  # Always n_zones treatments for beta=0
    J = PER  # beta=0 means J = (1-0)*PER = PER
    return J, INT, PER


# ============================================================================
# PRECOMPUTE STATIC DATA
# ============================================================================

print("Computing static data...")

# Spatial coupling matrix M (zone-restricted dense)
print("Building spatial coupling matrix...")

@jit(nopython=True, cache=True)
def generate_M_csr(D, alphas, drain_to_zone, zone_members, zone_starts, n):
    """Generate zone-restricted M in CSR format. Only same-zone entries are stored."""
    # Count nonzeros per row
    row_counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        z = drain_to_zone[i]
        row_counts[i] = zone_starts[z + 1] - zone_starts[z]
    total_nnz = np.sum(row_counts)
    # Build CSR arrays
    M_row_starts = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        M_row_starts[i + 1] = M_row_starts[i] + row_counts[i]
    M_col_indices = np.empty(total_nnz, dtype=np.int64)
    M_values = np.empty(total_nnz, dtype=np.float64)
    for i in range(n):
        z = drain_to_zone[i]
        alpha_i = alphas[z]
        pos = M_row_starts[i]
        for k in range(zone_starts[z], zone_starts[z + 1]):
            j = zone_members[k]
            M_col_indices[pos] = j
            M_values[pos] = np.exp(-alpha_i * D[i, j])
            pos += 1
    return M_values, M_col_indices, M_row_starts

M_values, M_col_indices, M_row_starts = generate_M_csr(D_global, ALPHAS, drain_to_zone, zone_members, zone_starts_arr, n_drains)
print(f"  M CSR: {len(M_values)} nonzeros ({len(M_values)*8/1024:.0f} KB)")

# Initial conditions: per-drain baseline
print("Setting up initial conditions...")
u0 = np.zeros(n_drains, dtype=np.float64)
for i in range(n_drains):
    if 0 <= t_reformas[i]:  # Active drain at day 0
        u0[i] = BASELINE

# Make arrays contiguous for numba
drain_to_zone = np.ascontiguousarray(drain_to_zone)
u0 = np.ascontiguousarray(u0)

print(f"Setup complete for {n_drains} drains in {n_zones} zones")


# ============================================================================
# PARALLEL WORKER SETUP
# ============================================================================

_worker_data = {}

def _init_worker(M_values_arr, M_col_indices_arr, M_row_starts_arr,
                 u0_arr, n_drains_val, n_days_val, t_reformas_arr,
                 rainfall_24h_arr, rainfall_21d_arr, r0_arr,
                 drain_to_zone_arr, baseline_val, n_zones_val,
                 init_season_val, end_season_val, norm_dict_val, beta_val):
    """Initialize worker process with shared data."""
    global _worker_data
    _worker_data = {
        'M_values': M_values_arr, 'M_col_indices': M_col_indices_arr, 'M_row_starts': M_row_starts_arr,
        'u0': u0_arr, 'n_drains': n_drains_val, 'n_days': n_days_val,
        't_reformas': t_reformas_arr, 'rainfall_24h': rainfall_24h_arr,
        'rainfall_21d': rainfall_21d_arr, 'r0_values': r0_arr,
        'drain_to_zone': drain_to_zone_arr, 'baseline': baseline_val,
        'n_zones': n_zones_val, 'init_season': init_season_val,
        'end_season': end_season_val, 'norm_dict': norm_dict_val, 'beta': beta_val,
    }


def evaluate_neighbor(args):
    """Evaluate a neighbor schedule in worker process."""
    init_times, periodicity = args
    d = _worker_data
    
    try:
        schedule, counts = generate_schedule(
            init_times, periodicity, d['n_zones'], d['end_season']
        )
        
        integral = simulate_with_schedule(
            d['u0'], d['n_drains'], d['n_days'],
            d['M_values'], d['M_col_indices'], d['M_row_starts'],
            d['t_reformas'], W_MIN, W_FLUSH, B, D_PARAM, DT,
            d['rainfall_24h'], d['rainfall_21d'], d['r0_values'],
            schedule, counts,
            d['drain_to_zone'], d['baseline'], d['n_zones']
        )
        
        cost = cost_function(
            integral, schedule, counts, d['n_zones'], d['beta'],
            d['norm_dict'], d['init_season'], d['end_season']
        )
        
        if cost[1] < 0.001:
            return None
        
        return (cost[0], cost[1], cost[2], init_times, periodicity, schedule, counts)
    
    except Exception as e:
        print(f"Worker error: {e}")
        return None


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================

def run_simulated_annealing(beta, executor, run_id=1):
    """Run Simulated Annealing with parallel neighbor evaluation."""
    
    print(f"\n{'='*60}")
    print(f"SIMULATED ANNEALING - Year {YEAR}, beta={beta}, Run {run_id}/{N_RUNS}")
    print(f"  iters={ITERS_ANNEALING}, workers={N_WORKERS}")
    print(f"  (beta=0: using INT as selection metric, J=PER is constant)")
    print(f"{'='*60}\n")
    
    h_tag = datetime.now().strftime("%m_%d_%H%M%S")
    # Format beta as 2-digit string: 0->"00", 0.1->"01", ..., 1->"10"
    beta_str = f"{int(round(beta * 10)):02d}"
    
    THRESHOLD_J = 1.0  # Initial schedule cost threshold
    MAX_INIT_ATTEMPTS = 150

    # Find initial schedule
    print(f"Finding initial schedule with INT < {THRESHOLD_J}...")
    
    candidates = []
    best_candidate = None
    
    # First 50 attempts: collect all below threshold
    for attempt in range(1, 51):
        init_times_temp, periodicity_temp, schedule_temp, counts_temp = random_schedule(
            n_zones, init_mosq_season, end_mosq_season, MAX_ZONES_DAY, MAX_DAYS_WEEK
        )
        
        integral = simulate_with_schedule(
            u0, n_drains, n_days, M_values, M_col_indices, M_row_starts,
            t_reformas, W_MIN, W_FLUSH, B, D_PARAM, DT,
            rainfall_24h, rainfall_21d, r0_values,
            schedule_temp, counts_temp,
            drain_to_zone, BASELINE, n_zones
        )
        
        J_temp = cost_function(
            integral, schedule_temp, counts_temp, n_zones, beta,
            NORM_DICT[YEAR], init_mosq_season, end_mosq_season
        )
        
        # For beta=0, compare by INT (index 1) since J is constant
        if best_candidate is None or J_temp[1] < best_candidate[0][1]:
            best_candidate = (J_temp, init_times_temp, periodicity_temp, schedule_temp, counts_temp)
        
        if J_temp[1] < THRESHOLD_J:
            candidates.append((J_temp, init_times_temp, periodicity_temp, schedule_temp, counts_temp))
    
    # If any of first 50 below threshold: use best of those
    if len(candidates) > 0:
        candidates.sort(key=lambda x: x[0][1])  # Sort by INT for beta=0
        J0, init_times_0, periodicity_0, schedule_0, counts_0 = candidates[0]
        print(f"Using best from first 50: INT={J0[1]:.4f} (from {len(candidates)} below threshold)")
    else:
        # No candidates in first 50: continue 51-150 until first below threshold
        found = False
        for attempt in range(51, MAX_INIT_ATTEMPTS + 1):
            init_times_temp, periodicity_temp, schedule_temp, counts_temp = random_schedule(
                n_zones, init_mosq_season, end_mosq_season, MAX_ZONES_DAY, MAX_DAYS_WEEK
            )
            
            integral = simulate_with_schedule(
                u0, n_drains, n_days, M_values, M_col_indices, M_row_starts,
                t_reformas, W_MIN, W_FLUSH, B, D_PARAM, DT,
                rainfall_24h, rainfall_21d, r0_values,
                schedule_temp, counts_temp,
                drain_to_zone, BASELINE, n_zones
            )
            
            J_temp = cost_function(
                integral, schedule_temp, counts_temp, n_zones, beta,
                NORM_DICT[YEAR], init_mosq_season, end_mosq_season
            )
            
            if best_candidate is None or J_temp[1] < best_candidate[0][1]:
                best_candidate = (J_temp, init_times_temp, periodicity_temp, schedule_temp, counts_temp)
            
            if J_temp[1] < THRESHOLD_J:
                J0, init_times_0, periodicity_0, schedule_0, counts_0 = J_temp, init_times_temp, periodicity_temp, schedule_temp, counts_temp
                print(f"Found INT={J0[1]:.4f} < {THRESHOLD_J} at attempt {attempt}")
                found = True
                break
        
        # If still none below threshold: use overall best
        if not found:
            J0, init_times_0, periodicity_0, schedule_0, counts_0 = best_candidate
            print(f"No solution < {THRESHOLD_J} in {MAX_INIT_ATTEMPTS} attempts, using best: INT={J0[1]:.4f}")
    
    print(f"Initial: J={J0[0]:.4f}, INT={J0[1]:.4f}, PER={J0[2]:.4f}")
    
    # Storage: Results[step] = [J0, init_times, periodicity, Temp, accepted]
    # step counts ALL iterations (including failed), valid_iters counts only accepted
    Results = {}
    Results[0] = [J0, init_times_0.copy(), periodicity_0.copy(), INIT_T, True]
    
    Temp = INIT_T
    unproductive_iters = 0  # Count consecutive batches with no acceptance OR no INT change
    last_INT = J0[1]  # Track last accepted INT (for beta=0)
    
    # Stepped eps reduction
    eps_phase = 0
    eps_t = EPS_SCHEDULE[0][0]
    eps_p = EPS_SCHEDULE[0][1]  # Not used but kept for compatibility
    
    valid_iters = 0  # Only counts iterations where a neighbor was accepted
    step = 0  # Counts all iterations (for storage index)
    
    while valid_iters < ITERS_ANNEALING:
        step += 1
        Temp = Temp * C_T
        
        # Generate N_NEIGHBORS neighbors
        neighbor_args = []
        for _ in range(N_NEIGHBORS):
            new_init, new_period, _, _ = update_schedule(
                init_times_0, periodicity_0, eps_t, eps_p,
                n_zones, init_mosq_season, end_mosq_season, MAX_ZONES_DAY, MAX_DAYS_WEEK
            )
            neighbor_args.append((new_init.copy(), new_period.copy()))
        
        # Parallel evaluation
        neighbor_results = list(executor.map(evaluate_neighbor, neighbor_args))
        valid_neighbors = [(r, args) for r, args in zip(neighbor_results, neighbor_args) if r is not None]
        
        # Metropolis: for beta=0, use INT as the metric (since J is constant)
        accepted = None
        current_INT = J0[1]
        for result, args in valid_neighbors:
            new_INT = result[1]  # Use INT for comparison
            dE = new_INT - current_INT
            if dE < 0 or np.random.rand() < np.exp(-dE / Temp):
                if accepted is None or new_INT < accepted[0][1]:
                    accepted = (result, args)
        
        # Update state if any neighbor was accepted
        if accepted is not None:
            result, args = accepted
            J0 = (result[0], result[1], result[2])
            init_times_0 = args[0]
            periodicity_0 = args[1]
            valid_iters += 1  # Only increment valid_iters when accepted
            
            # Check if this acceptance was productive (INT changed)
            if J0[1] != last_INT:
                last_INT = J0[1]
                unproductive_iters = 0  # Reset: we made progress
            else:
                unproductive_iters += 1  # Accepted but INT didn't change
        else:
            # No neighbor accepted at all
            unproductive_iters += 1
        
        # Store every step (including failed) to see flat regions
        Results[step] = [J0, init_times_0.copy(), periodicity_0.copy(), Temp, accepted is not None]
        
        if step % 100 == 0:
            print(f"Step {step}, Valid {valid_iters}/{ITERS_ANNEALING}: J={J0[0]:.4f}, INT={J0[1]:.4f}, PER={J0[2]:.4f}, T={Temp:.6f}, unproductive={unproductive_iters}")
            
            save_path = RESULTS_DIR / f"Simulated_Annealing_{YEAR}_b{beta_str}_{h_tag}"
            with open(save_path, 'wb') as f:
                pickle.dump(Results, f)
            
            if unproductive_iters >= EPS_SCHEDULE[eps_phase][2]:
                if eps_phase < len(EPS_SCHEDULE) - 1:
                    eps_phase += 1
                    eps_t = EPS_SCHEDULE[eps_phase][0]
                    eps_p = EPS_SCHEDULE[eps_phase][1]
                    print(f"  -> Reducing eps to ({eps_t}, {eps_p}) [phase {eps_phase}]")
                    unproductive_iters = 0
                else:
                    print(f"No progress for {unproductive_iters} batches in final phase, stopping early")
                    break
    
    # Final save
    save_path = RESULTS_DIR / f"Simulated_Annealing_{YEAR}_b{beta_str}_{h_tag}"
    with open(save_path, 'wb') as f:
        pickle.dump(Results, f)
    print(f"\nSaved: {save_path}")
    print(f"Total steps: {step}, Valid iterations: {valid_iters}")
    
    # Best result (by INT for beta=0)
    best_step = min(Results.keys(), key=lambda k: Results[k][0][1])  # Min by INT
    best = Results[best_step]
    print(f"\nBest (step {best_step}):")
    print(f"  J={best[0][0]:.4f}, INT={best[0][1]:.4f}, PER={best[0][2]:.4f}")
    print(f"  Init times: {best[1]}")
    print(f"  Periodicity: {best[2]}")
    
    return Results


# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

print("Starting optimization...")
print(f"Year {YEAR}, {N_WORKERS} workers, {ITERS_ANNEALING} iterations, {N_RUNS} runs")

all_results = []
with ProcessPoolExecutor(
    max_workers=N_WORKERS,
    initializer=_init_worker,
    initargs=(M_values, M_col_indices, M_row_starts,
              u0, n_drains, n_days, t_reformas, rainfall_24h, rainfall_21d,
              r0_values, drain_to_zone, BASELINE, n_zones,
              init_mosq_season, end_mosq_season, NORM_DICT[YEAR], BETA_SIM)
) as executor:
    for run_id in range(1, N_RUNS + 1):
        print(f"\n*** Starting Run {run_id}/{N_RUNS} ***")
        Results = run_simulated_annealing(BETA_SIM, executor, run_id)
        all_results.append(Results)
        print(f"*** Completed Run {run_id}/{N_RUNS} ***")

print(f"\nAll {N_RUNS} optimization runs finished!")

# Print summary of all runs
print(f"\n{'='*60}")
print("SUMMARY OF ALL RUNS")
print(f"{'='*60}")
for i, results in enumerate(all_results, 1):
    best_step = min(results.keys(), key=lambda k: results[k][0][1])  # Min by INT for beta=0
    best = results[best_step]
    total_steps = max(results.keys())
    print(f"Run {i}: Best INT={best[0][1]:.4f} (J={best[0][0]:.4f}) at step {best_step}, total steps: {total_steps}")

# Find overall best
best_run = 0
best_int = float('inf')
for i, results in enumerate(all_results):
    best_step = min(results.keys(), key=lambda k: results[k][0][1])
    int_val = results[best_step][0][1]
    if int_val < best_int:
        best_int = int_val
        best_run = i + 1

print(f"\n*** Overall best: Run {best_run} with INT={best_int:.4f} ***")
