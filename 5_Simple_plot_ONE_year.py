"""Simple Plot Year: Simulate mosquito dynamics for a single year.

Intervention modes:
  1. uncontrolled - no treatment at all
  2. periodic - define init_times and periodicity by hand
  3. ASPB - load from ASPB_Interventions file (all treatments, drain-level)
  4. ASPB_Zona_risc - load from ASPB_Visits_Zona_risc (ALL zona-risc visits,
     zone-level, merged <=2d back-to-back). The n_z>=1 gate in simulate_year
     decides whether treatment fires on each visit day.
"""

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit
from pathlib import Path
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from collections import OrderedDict

# ============================================================================
# SETUP & PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Dades_netes"
PARAM_DIR = SCRIPT_DIR / "Parameters"
FIGURES_DIR = SCRIPT_DIR / "Figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Global reference date (climate data and ASPB interventions are relative to this)
d0_global = datetime(2019, 1, 1)

# ============================================================================
# USER SETTINGS - CHANGE THESE
# ============================================================================

# Year to simulate
YEAR = 2024

# Intervention mode: "uncontrolled", "periodic", "ASPB", or "ASPB_Zona_risc"
INTERVENTION_MODE = "ASPB_Zona_risc"

# periodic intervention settings (used only if INTERVENTION_MODE = "periodic")
# init_times[i] = first intervention day for zone i (relative to Jan 1 of YEAR)
# periodicity[i] = days between interventions for zone i
#CUSTOM_INIT_TIMES = [91, 91, 91, 91, 91, 91, 91]  # April 1 = day 91
#CUSTOM_PERIODICITY = [30, 30, 30, 30, 30, 30, 30]  # Every 30 days

# ============================================================================
# MODEL PARAMETERS 
# ============================================================================


ALPHAS = np.array([0.088, 0.097, 0.069, 0.14, 0.017, 0.19, 0.086], dtype=np.float64)
W_MIN = 11.4
W_FLUSH = 9.6
B = 1.9e-4
D_PARAM = 1e-5

DT = 0.30  #RK4 integration step
BASELINE = 0.001  # Per-drain baseline probability



# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# Climate NetCDF files
rainfall_data = xr.open_dataset(PARAM_DIR / "Rainfall.nc")['data']
rainfall_21_data = xr.open_dataset(PARAM_DIR / "Rainfall_21.nc")['data']
r0_data = xr.open_dataset(PARAM_DIR / "R0_both.nc")['data']

# Spatial data
with open(PARAM_DIR / "D_Matrix", 'rb') as f:
    D = pickle.load(f)
with open(PARAM_DIR / "X_coords", 'rb') as f:
    X_coords_dict = pickle.load(f)
with open(PARAM_DIR / "Y_coords", 'rb') as f:
    Y_coords_dict = pickle.load(f)

# Metadata
unique_bs = pd.read_csv(DATA_DIR / "Unique_Items.csv")

n_drains = D.shape[0]

# Build arrays
X_emb = np.array([X_coords_dict[i] for i in range(n_drains)])
Y_emb = np.array([Y_coords_dict[i] for i in range(n_drains)])

# ============================================================================
# ZONE COLORS (canonical zone order - same across all scripts)
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

# Zone info (use zone_colors as canonical order)
zone_names = list(zone_colors.keys())
n_zones = len(zone_names)
zone_name_to_idx = {name: i for i, name in enumerate(zone_names)}
drain_to_zone = np.array([zone_name_to_idx[z] for z in unique_bs['nom_zr']], dtype=np.int32)

# Precompute zone membership arrays for fast zone-restricted M matrix
_zone_lists = [[] for _ in range(n_zones)]
for i in range(n_drains):
    _zone_lists[drain_to_zone[i]].append(i)
zone_members = np.array([idx for zl in _zone_lists for idx in zl], dtype=np.int32)
zone_starts = np.zeros(n_zones + 1, dtype=np.int32)
for z in range(n_zones):
    zone_starts[z + 1] = zone_starts[z] + len(_zone_lists[z])
del _zone_lists

# Reform dates - parsed here, actual t_reformas computed per year in compute_t_reformas_for_year()
unique_bs['data_reforma'] = pd.to_datetime(unique_bs['data_reforma'], errors='coerce')
unique_bs['data_reforma'] = unique_bs['data_reforma'].fillna(pd.Timestamp("2030-01-01"))

print(f"Loaded {n_drains} drains, {n_zones} zones")
print(f"Zones: {zone_names}")

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
def simulate_year(
    u0, n_drains, n_days, M_values, M_col_indices, M_row_starts,
    t_reformas, w_min, w_flush, B, D_param, dt,
    rainfall_24h, rainfall_21d, r0_values,
    interventions_list, intervention_starts,
    drain_to_zone, baseline
):
    """
    RK4 simulation harmonized with 3_Param_fit_R0.py.
    
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
    n_zones = 0
    for i in range(n_drains):
        if drain_to_zone[i] + 1 > n_zones:
            n_zones = drain_to_zone[i] + 1
    
    for day in range(1, n_days):
        # Precompute zone sums at start of day (ASPB protocol: treat only if n_z >= 1)
        current_zone_sums = np.zeros(n_zones)
        for i in range(n_drains):
            current_zone_sums[drain_to_zone[i]] += u[i]
        
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
            # ASPB protocol: only treat if zone has n_z(t) >= 1 expected positives
            is_under_intervention = False
            zone_idx = drain_to_zone[i]
            if current_zone_sums[zone_idx] >= 1.0:
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
            k1 = compute_du(u, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + 0.5 * dt * k1
            k2 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + 0.5 * dt * k2
            k3 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            u_temp = u + dt * k3
            k4 = compute_du(u_temp, M_values, M_col_indices, M_row_starts, r0_day, B, D_param, n_drains, can_evolve)
            
            u = u + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
        solution[day] = u.copy()
    
    return solution


@jit(nopython=True, cache=True)
def compute_zone_integrals(solution, drain_to_zone, n_zones, n_days):
    """Compute integral over time for each zone (trapezoidal rule)."""
    zone_integrals = np.zeros(n_zones)
    
    # First compute zone sums per day
    zone_sums = np.zeros((n_days, n_zones))
    for day in range(n_days):
        for i in range(len(drain_to_zone)):
            zone_sums[day, drain_to_zone[i]] += solution[day, i]
    
    # Integrate using trapezoidal rule
    for z in range(n_zones):
        for day in range(n_days - 1):
            zone_integrals[z] += 0.5 * (zone_sums[day, z] + zone_sums[day + 1, z])
    
    return zone_integrals, zone_sums

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_climate_for_year(year):
    """Prepare climate data for a specific year (Jan 1 to Dec 31)."""
    print("  Interpolating climate data...")
    
    d0_year = datetime(year, 1, 1)
    df_year = datetime(year, 12, 31)
    day_start = (d0_year - d0_global).days
    day_end = (df_year - d0_global).days
    n_days = day_end - day_start + 1
    
    # Interpolate climate data
    rainfall_24h_full = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d_full = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    r0_full = interpolate_all_locations(r0_data, X_emb, Y_emb)
    
    rainfall_24h_full = np.nan_to_num(rainfall_24h_full, nan=0.0)
    rainfall_21d_full = np.nan_to_num(rainfall_21d_full, nan=0.0)
    r0_full = np.nan_to_num(r0_full, nan=0.00001)
    
    # Slice for this year
    rainfall_24h = rainfall_24h_full[day_start:day_end + 1].copy()
    rainfall_21d = rainfall_21d_full[day_start:day_end + 1].copy()
    r0_vals = r0_full[day_start:day_end + 1].copy()
    
    print(f"  Climate arrays shape: {rainfall_24h.shape}")
    return rainfall_24h, rainfall_21d, r0_vals, n_days


def compute_t_reformas_for_year(year):
    """
    Compute reform dates for a specific year.
    
    Logic (same as 6_Schedule_Optimization.py):
    - If reformed before or during this year: t_reformas = -1 (already reformed from day 0)
    - If reformed in a future year or never: t_reformas = n_days + 1 (active all year)
    
    This ensures year-by-year simulations are comparable with optimization results.
    """
    d0_year = datetime(year, 1, 1)
    df_year = datetime(year, 12, 31)
    n_days_year = (df_year - d0_year).days + 1
    
    t_reformas = np.zeros(n_drains, dtype=np.float64)
    
    for i in range(n_drains):
        reform_date = unique_bs['data_reforma'].iloc[i]
        if pd.notna(reform_date):
            reform_year = reform_date.year
            if reform_year < year or reform_year == year:
                t_reformas[i] = -1  # Already reformed from day 0
            else:
                t_reformas[i] = n_days_year + 1  # Active all year
        else:
            t_reformas[i] = n_days_year + 1  # Never reformed
    
    return t_reformas


def compute_initial_conditions(drain_to_zone, t_reformas, n_drains, baseline):
    """Initial condition per drain: BASELINE if active, 0 if reformed."""
    u0 = np.zeros(n_drains, dtype=np.float64)
    for i in range(n_drains):
        if 0 <= t_reformas[i]:  # Active at day 0
            u0[i] = baseline
    return u0


def generate_schedule_from_params(init_times, periodicity, init_mosq, end_mosq):
    """Generate intervention schedule from init_times and periodicity."""
    schedule = {}
    for z in range(n_zones):
        times = []
        t = init_times[z]
        while t <= end_mosq:
            times.append(int(t))
            t += periodicity[z]
        schedule[z] = times
    return schedule


def schedule_to_interventions(schedule, drain_to_zone, n_drains):
    """Convert zone-based schedule to drain-based intervention arrays."""
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        zone_idx = drain_to_zone[i]
        if zone_idx in schedule:
            for t in schedule[zone_idx]:
                interventions_list.append(t)
    
    intervention_starts[n_drains] = len(interventions_list)
    return np.array(interventions_list, dtype=np.int64), intervention_starts


def prepare_no_interventions(n_drains):
    """Prepare empty intervention arrays (no treatment)."""
    interventions_list = np.array([], dtype=np.int64)
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    return interventions_list, intervention_starts


def prepare_ASPB_interventions(year, intervention_file):
    """
    Load ASPB interventions for a specific year.
    
    ASPB intervention days are stored as global days (relative to 2019-01-01).
    We convert them to year-local days (relative to Jan 1 of target year).
    """
    with open(PARAM_DIR / intervention_file, 'rb') as f:
        interventions_dict = pickle.load(f)
    
    d0_year = datetime(year, 1, 1)
    df_year = datetime(year, 12, 31)
    day_start_global = (d0_year - d0_global).days
    day_end_global = (df_year - d0_global).days
    
    # Filter and convert to year-local days
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    total_interventions = 0
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        if i in interventions_dict:
            for global_day in interventions_dict[i]:
                if day_start_global <= global_day <= day_end_global:
                    local_day = global_day - day_start_global
                    interventions_list.append(local_day)
                    total_interventions += 1
    
    intervention_starts[n_drains] = len(interventions_list)
    
    print(f"  Loaded {total_interventions} interventions for year {year}")
    return np.array(interventions_list, dtype=np.int64), intervention_starts


def extract_zone_schedule_from_interventions(interventions_list, intervention_starts, drain_to_zone, n_drains, n_zones):
    """Extract zone-level schedule from drain-level interventions."""
    schedule = {z: set() for z in range(n_zones)}
    
    for i in range(n_drains):
        zone_idx = drain_to_zone[i]
        start = intervention_starts[i]
        end = intervention_starts[i + 1]
        for k in range(start, end):
            schedule[zone_idx].add(interventions_list[k])
    
    # Convert to sorted lists
    for z in range(n_zones):
        schedule[z] = sorted(list(schedule[z]))
    
    return schedule


def compute_PER(schedule, init_mosq, end_mosq, n_zones):
    """
    Compute PER (treatment frequency metric).
    
    PER = total_treatments / effective_season_length
    
    effective_season_length = max(end_mosq, last_treatment) - min(init_mosq, first_treatment) + 1
    
    Returns PER in [0, ~1]:
    - PER = 0 means no treatment (best, least cost)
    - PER ≈ 1 means ~1 treatment per day (worst, most cost)
    - For periodicity P: PER ≈ 1/P (e.g., P=30 days → PER ≈ 0.033)
    - Lower PER = fewer treatments = better (in minimization context)
    """
    # Collect all treatments across zones
    all_treatments = []
    for z in range(n_zones):
        all_treatments.extend(schedule.get(z, []))
    
    n_treatments = len(all_treatments)
    if n_treatments == 0:
        return 0.0
    
    # first_treatment = min(all_treatments)
    # last_treatment = max(all_treatments)
    
    # effective_start = min(init_mosq, first_treatment)
    # effective_end = max(end_mosq, last_treatment)
    # effective_length = effective_end - effective_start + 1
    
    return n_treatments #/ effective_length


def compute_INT(zone_integrals):
    """Compute total integral (sum across zones)."""
    return np.sum(zone_integrals)

# ============================================================================
# PLOTTING
# ============================================================================

def plot_stacked_zones_year(solution, year, n_days, title, filename):
    """Create stacked area plot by zone for a single year."""
    days = np.arange(n_days)
    
    zone_probs = []
    zone_labels = []
    colors = []
    
    for zone_name in zone_names:
        zone_idx = zone_name_to_idx[zone_name]
        zone_mask = drain_to_zone == zone_idx
        zone_sum = np.sum(solution[:, zone_mask], axis=1)
        zone_probs.append(zone_sum)
        zone_labels.append(zone_name)
        colors.append(zone_colors.get(zone_name, "#CCCCCC"))
    
    # Sort by total area descending (largest at bottom)
    zone_totals = [np.trapz(zp) for zp in zone_probs]
    order = np.argsort(zone_totals)[::-1]
    zone_probs  = [zone_probs[i]  for i in order]
    zone_labels = [zone_labels[i] for i in order]
    colors      = [colors[i]      for i in order]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(days, *zone_probs, labels=zone_labels, colors=colors, alpha=0.85)
    
    ax.set_xlabel("Day of Year", fontsize=12)
    ax.set_ylabel("Expected Positive Drains", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # X-axis: show months
    month_starts = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    ax.set_xlim(0, n_days)
    
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150)
    plt.show()
    plt.close()
    
    print(f"  Saved: {filename}")



# ==============
# SIMULATION
# ==============


print(f"\n{'='*70}")
print(f"Mosquito Dynamics Simulation - Year {YEAR}")
print(f"Intervention mode: {INTERVENTION_MODE}")
print(f"{'='*70}\n")

# Mosquito season bounds (April 1 to Nov 30), relative to Jan 1
init_mosq_season = (datetime(YEAR, 4, 1) - datetime(YEAR, 1, 1)).days
end_mosq_season = (datetime(YEAR, 11, 30) - datetime(YEAR, 1, 1)).days
print(f"Mosquito season: day {init_mosq_season} to {end_mosq_season}")

# Prepare climate data
print("\nPreparing data...")
rainfall_24h, rainfall_21d, r0_vals, n_days = prepare_climate_for_year(YEAR)
print(f"Simulation: {n_days} days (Jan 1 to Dec 31)")

# Compute reform dates and initial conditions
t_reformas = compute_t_reformas_for_year(YEAR)
u0 = compute_initial_conditions(drain_to_zone, t_reformas, n_drains, BASELINE)

# Prepare interventions based on mode
if INTERVENTION_MODE == "uncontrolled":
    print("  Using NO INTERVENTION")
    interventions_list, intervention_starts = prepare_no_interventions(n_drains)
    schedule = {z: [] for z in range(n_zones)}
    mode_tag = "uncontrolled"
    
elif INTERVENTION_MODE == "periodic":
    print(f"  Using periodic intervention schedule")
    print(f"    init_times: {CUSTOM_INIT_TIMES}")
    print(f"    periodicity: {CUSTOM_PERIODICITY}")
    
    init_times = np.array(CUSTOM_INIT_TIMES, dtype=np.float64)
    periodicity = np.array(CUSTOM_PERIODICITY, dtype=np.float64)
    
    schedule = generate_schedule_from_params(init_times, periodicity, 
                                                init_mosq_season, end_mosq_season)
    interventions_list, intervention_starts = schedule_to_interventions(
        schedule, drain_to_zone, n_drains)
    mode_tag = "periodic"
    
elif INTERVENTION_MODE == "ASPB":
    print("  Using ASPB interventions (all treatments)")
    interventions_list, intervention_starts = prepare_ASPB_interventions(
        YEAR, "ASPB_Interventions")
    schedule = extract_zone_schedule_from_interventions(
        interventions_list, intervention_starts, drain_to_zone, n_drains, n_zones)
    mode_tag = "ASPB"
    
elif INTERVENTION_MODE == "ASPB_Zona_risc":
    print("  Using ASPB zona-risc ALL visits (zone-level, merged <=2d)")
    interventions_list, intervention_starts = prepare_ASPB_interventions(
        YEAR, "ASPB_Visits_Zona_risc")
    schedule = extract_zone_schedule_from_interventions(
        interventions_list, intervention_starts, drain_to_zone, n_drains, n_zones)
    mode_tag = "ASPB_Zona_risc"
    
else:
    raise ValueError(f"Unknown INTERVENTION_MODE: {INTERVENTION_MODE}")

# Print schedule summary
print("\n  Schedule by zone:")
for z in range(n_zones):
    n_interv = len(schedule.get(z, []))
    print(f"    {zone_names[z]}: {n_interv} interventions")

# Build connectivity matrix (zone-restricted dense)
print("\nBuilding connectivity matrix...")
M_values, M_col_indices, M_row_starts = generate_M_csr(D, ALPHAS, drain_to_zone, zone_members, zone_starts, n_drains)
print(f"  M CSR: {len(M_values)} nonzeros ({len(M_values)*8/1024:.0f} KB)")

# Run simulation
print("\nRunning simulation...")
solution = simulate_year(
    u0, n_drains, n_days, M_values, M_col_indices, M_row_starts,
    t_reformas, W_MIN, W_FLUSH, B, D_PARAM, DT,
    rainfall_24h, rainfall_21d, r0_vals,
    interventions_list, intervention_starts,
    drain_to_zone, BASELINE
)
print(f"  Solution shape: {solution.shape}")

# Compute statistics
zone_integrals, zone_sums = compute_zone_integrals(solution, drain_to_zone, n_zones, n_days)
INT_total = compute_INT(zone_integrals)
PER = compute_PER(schedule, init_mosq_season, end_mosq_season, n_zones)

# Print summary
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print(f"\nZone integrals (INT):")
for z in range(n_zones):
    print(f"  {zone_names[z]}: {zone_integrals[z]:.2f}")
print(f"\n  TOTAL INT: {INT_total:.2f}")

print(f"\nTreatment frequency (PER):")
print(f"  PER: {PER:.4f} treatments")

# Max and mean
total = np.sum(solution, axis=1)
print(f"\nDynamics summary:")
print(f"  Max INT: {np.max(total):.2f}")
print(f"  Mean INT: {np.mean(total):.2f}")

# Plot
print("\nGenerating plot...")
title = f"Mosquito Dynamics {YEAR} - {INTERVENTION_MODE.replace('_', ' ')}"
filename = f"Dynamics_{YEAR}_{mode_tag}.png"

plot_stacked_zones_year(solution, YEAR, n_days, title, filename)

print("\nDone!")

