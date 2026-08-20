"""Simple Plot ALL Years: Simulate mosquito dynamics for all years (2019-2024).

Intervention modes:
  1. uncontrolled - no treatment at all
  2. ASPB - load from ASPB_Interventions file (all treatments, drain-level)
  3. ASPB_Zona_risc - load from ASPB_Visits_Zona_risc (ALL zona-risc visits,
     zone-level, merged <=2d back-to-back). The n_z>=1 gate in simulate_year
     decides whether treatment fires on each visit day.

Note: periodic mode is not available for multi-year simulation.
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
FIGURES_DIR = SCRIPT_DIR / "Figures/Validation_Plots"
FIGURES_DIR.mkdir(exist_ok=True)

# Global reference date (climate data and ASPB interventions are relative to this)
d0_global = datetime(2019, 1, 1)
tf_global = datetime(2024, 12, 31)

# ============================================================================
# USER SETTINGS - CHANGE THESE
# ============================================================================

# Intervention mode: "uncontrolled", "ASPB", or "ASPB_Zona_risc"
#INTERVENTION_MODE = "ASPB_Zona_risc"
INTERVENTION_MODE = "ASPB"

# Show observed ASPB data panel below the model plot
PANEL = True

# Aggregation mode for observed-data bars:
#   "fixed_weeks"    -> current behavior, anchored at 2019-01-01
#   "calendar_month" -> true month-by-month aggregation
OBS_AGG_MODE = "calendar_month"

# Aggregation window for the observed-data bars in weeks
# 4 = roughly monthly, 2 = biweekly, 1 = weekly
OBS_AGG_WEEKS = 3

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
n_days = (tf_global - d0_global).days + 1  # 2192 days

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

# Reform dates (relative to global d0)
unique_bs['data_reforma'] = pd.to_datetime(unique_bs['data_reforma'], errors='coerce')
unique_bs['data_reforma'] = unique_bs['data_reforma'].fillna(pd.Timestamp("2030-01-01"))
t_reformas = np.array([(dt - d0_global).days for dt in unique_bs['data_reforma']], dtype=np.float64)

# Precompute zone membership arrays for fast zone-restricted M matrix
_zone_lists = [[] for _ in range(n_zones)]
for i in range(n_drains):
    _zone_lists[drain_to_zone[i]].append(i)
zone_members = np.array([idx for zl in _zone_lists for idx in zl], dtype=np.int32)
zone_starts = np.zeros(n_zones + 1, dtype=np.int32)
for z in range(n_zones):
    zone_starts[z + 1] = zone_starts[z] + len(_zone_lists[z])
del _zone_lists


print(f"Loaded {n_drains} drains, {n_zones} zones, {n_days} days")
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
def simulate_all_years(
    u0, n_drains, n_days, M, t_reformas, w_min, w_flush, B, D_param, dt,
    rainfall_24h, rainfall_21d, r0_values,
    interventions_list, intervention_starts,
    drain_to_zone, zone_members, zone_starts, baseline
):
    """
    RK4 simulation for all years, harmonized with 3_Param_fit_R0.py.
    
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


@jit(nopython=True)
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

def prepare_climate_all_years():
    """Prepare climate data for all years (2019-2024)."""
    print("  Interpolating climate data...")
    
    rainfall_24h = interpolate_all_locations(rainfall_data, X_emb, Y_emb)
    rainfall_21d = interpolate_all_locations(rainfall_21_data, X_emb, Y_emb)
    r0_vals = interpolate_all_locations(r0_data, X_emb, Y_emb)
    
    rainfall_24h = np.nan_to_num(rainfall_24h, nan=0.0)
    rainfall_21d = np.nan_to_num(rainfall_21d, nan=0.0)
    r0_vals = np.nan_to_num(r0_vals, nan=0.00001)
    
    # Ensure we have enough days
    if rainfall_24h.shape[0] < n_days:
        pad_days = n_days - rainfall_24h.shape[0]
        rainfall_24h = np.vstack([rainfall_24h, np.zeros((pad_days, n_drains))])
        rainfall_21d = np.vstack([rainfall_21d, np.zeros((pad_days, n_drains))])
        r0_vals = np.vstack([r0_vals, np.full((pad_days, n_drains), 0.00001)])
    
    print(f"  Climate arrays shape: {rainfall_24h[:n_days].shape}")
    return rainfall_24h[:n_days], rainfall_21d[:n_days], r0_vals[:n_days]


def compute_initial_conditions_global():
    """Initial condition per drain: BASELINE if active at day 0, 0 if reformed."""
    u0 = np.zeros(n_drains, dtype=np.float64)
    for i in range(n_drains):
        if t_reformas[i] > 0:
            u0[i] = BASELINE
    return u0


def prepare_no_interventions():
    """Prepare empty intervention arrays (no treatment)."""
    interventions_list = np.array([], dtype=np.int64)
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    return interventions_list, intervention_starts


def prepare_ASPB_interventions_global(intervention_file):
    """
    Load ASPB interventions for all years.
    Intervention days are already stored as global days (relative to 2019-01-01).
    """
    with open(PARAM_DIR / intervention_file, 'rb') as f:
        interventions_dict = pickle.load(f)
    
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    total_interventions = 0
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        if i in interventions_dict:
            for day in interventions_dict[i]:
                interventions_list.append(day)
                total_interventions += 1
    
    intervention_starts[n_drains] = len(interventions_list)
    
    print(f"  Loaded {total_interventions} interventions (all years)")
    return np.array(interventions_list, dtype=np.int64), intervention_starts


def extract_zone_schedule_from_interventions_global(interventions_list, intervention_starts):
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


def compute_INT(zone_integrals):
    """Compute total integral (sum across zones)."""
    return np.sum(zone_integrals)

# ============================================================================
# PLOTTING
# ============================================================================

def load_observed_data():
    """Load visit-level observed data used to build the lower panel."""
    df = pd.read_csv(DATA_DIR / "Revisions.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df = df.dropna(subset=['Fecha', 'nom_zr'])
    df = df[df['visita'] == 'Zona risc'].copy()
    df['activitat_both'] = df['activitat_both'].fillna(False).astype(bool)
    return df


def compute_weighted_active_drains(zone_idx, day_start, day_end):
    """Average number of active drains in a zone over a time window."""
    n_days_window = day_end - day_start + 1
    weighted_count = 0.0

    for i in range(len(drain_to_zone)):
        if drain_to_zone[i] != zone_idx:
            continue

        reform_day = t_reformas[i]
        if reform_day < day_start:
            continue
        if reform_day >= day_end:
            weighted_count += 1.0
        else:
            days_active = reform_day - day_start + 1
            weighted_count += days_active / n_days_window

    return weighted_count


def aggregate_observed_cases(obs_raw, agg_weeks):
    """Aggregate observed cases by zone over fixed windows of `agg_weeks` weeks."""
    if agg_weeks <= 0:
        raise ValueError("OBS_AGG_WEEKS must be a positive integer.")

    window_days = int(agg_weeks * 7)
    obs = obs_raw.copy()
    obs['day_idx'] = (obs['Fecha'] - pd.Timestamp(d0_global)).dt.days.astype(int)
    obs = obs[(obs['day_idx'] >= 0) & (obs['day_idx'] < n_days)].copy()
    obs['window_idx'] = obs['day_idx'] // window_days

    results = []
    grouped = obs.groupby(['nom_zr', 'window_idx'], sort=True)

    for (zone_name, window_idx), group in grouped:
        zone_idx = zone_name_to_idx[zone_name]
        day_start = int(window_idx * window_days)
        day_end = min(day_start + window_days - 1, n_days - 1)
        n_visits = len(group)
        n_positives = int(group['activitat_both'].sum())

        n_active_weighted = compute_weighted_active_drains(zone_idx, day_start, day_end)
        if n_active_weighted < 0.01 or n_visits == 0:
            continue

        prevalence = n_positives / n_visits
        real_cases = prevalence * n_active_weighted
        period_mid = day_start + 0.5 * (day_end - day_start)
        date_mid = d0_global + pd.to_timedelta(period_mid, unit='D')

        results.append({
            'zone': zone_name,
            'zone_idx': zone_idx,
            'date': pd.Timestamp(date_mid),
            'real_cases': real_cases,
            'n_visits': n_visits,
            'n_positives': n_positives,
            'prevalence': prevalence,
            'n_active': n_active_weighted,
            'bar_width_days': window_days,
        })

    return pd.DataFrame(results)


def aggregate_observed_cases_monthly(obs_raw):
    """Aggregate observed cases by true calendar month."""
    obs = obs_raw.copy()
    obs['year'] = obs['Fecha'].dt.year.astype(int)
    obs['month'] = obs['Fecha'].dt.month.astype(int)

    results = []
    grouped = obs.groupby(['nom_zr', 'year', 'month'], sort=True)

    for (zone_name, year, month), group in grouped:
        zone_idx = zone_name_to_idx[zone_name]
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0)

        day_start = max(0, (month_start - pd.Timestamp(d0_global)).days)
        day_end = min((month_end - pd.Timestamp(d0_global)).days, n_days - 1)
        if day_start > day_end:
            continue

        n_visits = len(group)
        n_positives = int(group['activitat_both'].sum())
        n_active_weighted = compute_weighted_active_drains(zone_idx, day_start, day_end)
        if n_active_weighted < 0.01 or n_visits == 0:
            continue

        prevalence = n_positives / n_visits
        real_cases = prevalence * n_active_weighted
        bar_width_days = day_end - day_start + 1
        period_mid = day_start + 0.5 * (day_end - day_start)
        date_mid = d0_global + pd.to_timedelta(period_mid, unit='D')

        results.append({
            'zone': zone_name,
            'zone_idx': zone_idx,
            'date': pd.Timestamp(date_mid),
            'real_cases': real_cases,
            'n_visits': n_visits,
            'n_positives': n_positives,
            'prevalence': prevalence,
            'n_active': n_active_weighted,
            'bar_width_days': bar_width_days,
        })

    return pd.DataFrame(results)


def aggregate_observed_cases_for_plot(obs_raw, agg_mode, agg_weeks):
    """Dispatch observed-data aggregation according to the selected mode."""
    if agg_mode == "fixed_weeks":
        return aggregate_observed_cases(obs_raw, agg_weeks)
    if agg_mode == "calendar_month":
        return aggregate_observed_cases_monthly(obs_raw)
    raise ValueError(
        f"Unknown OBS_AGG_MODE: {agg_mode}. Use 'fixed_weeks' or 'calendar_month'."
    )


def _setup_xaxis(ax, date_min, date_max):
    """Shared x-axis: one tick every 3 months, labelled 'Mon YY'."""
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %y'))
    ax.set_xlim(date_min, date_max)
    ax.tick_params(axis='x', labelsize=14, rotation=30)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')


def plot_stacked_zones_all_years(solution, n_days, title, filename, show_panel=False):
    """Create stacked area plot by zone for all years, with optional observed data panel."""
    dates = pd.date_range(start=d0_global, periods=n_days, freq='D')
    date_min, date_max = dates[0], dates[-1]

    # --- build zone series ---
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
    zone_totals = [np.trapezoid(zp) for zp in zone_probs]
    order = np.argsort(zone_totals)[::-1]
    zone_probs  = [zone_probs[i]  for i in order]
    zone_labels = [zone_labels[i] for i in order]
    colors      = [colors[i]      for i in order]

    # --- figure layout ---
    if show_panel:
        fig, (ax_model, ax_obs) = plt.subplots(
            2, 1, figsize=(18, 8.8), sharex=True,
            gridspec_kw={'height_ratios': [3.1, 1.15], 'hspace': 0.04}
        )
    else:
        fig, ax_model = plt.subplots(figsize=(18, 7))

    # --- top panel: model ---
    ax_model.stackplot(dates, *zone_probs, labels=zone_labels, colors=colors, alpha=0.85)
    ax_model.set_ylabel("Expected Number of\nPositive Drains", fontsize=17)
    ax_model.set_ylim(bottom=0)
    ax_model.grid(True, alpha=0.25, axis='y')
    top_max = np.max(np.sum(np.vstack(zone_probs), axis=0))
    #ax_model.set_ylim(0, top_max * 1.01) #controlar espacio en blanco en el top plot
    ax_model.set_ylim(0, top_max * 0.8) #controlar truncado en el top plot
    ax_model.margins(x=0)
    ax_model.tick_params(axis='y', labelsize=14)
    ax_model.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_model.legend(
        loc='upper right',
        bbox_to_anchor=(1.0, 1.03),
        ncol=2,
        fontsize=18,
        frameon=False,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=1.9,
        columnspacing=1.0
    )
    ax_model.set_xlim(date_min, date_max)

    # --- bottom panel: observed stacked bars ---
    if show_panel:
        obs_raw = load_observed_data()
        obs = aggregate_observed_cases_for_plot(obs_raw, OBS_AGG_MODE, OBS_AGG_WEEKS)

        # Plot stacked bars in same zone order as top panel
        bottom_vals = np.zeros(len(obs['date'].unique()))
        all_dates_sorted = np.sort(obs['date'].unique())
        date_nums = plt.matplotlib.dates.date2num(pd.to_datetime(all_dates_sorted))
        width_by_date = (obs[['date', 'bar_width_days']]
                         .drop_duplicates('date')
                         .set_index('date')['bar_width_days'])
        bar_widths = np.array([width_by_date.loc[d] for d in pd.to_datetime(all_dates_sorted)])

        for label, color in zip(zone_labels, colors):
            zone_obs = obs[obs['zone'] == label].copy()
            zone_obs = zone_obs.set_index('date')['real_cases']
            vals = np.array([zone_obs.get(d, 0.0) for d in all_dates_sorted])
            ax_obs.bar(date_nums, vals, width=bar_widths.mean() - 1.5, bottom=bottom_vals,  #aqui para cambiar bar width
                       color=color, alpha=0.85, label=label)
            bottom_vals += vals

        # y-axis headroom: 15% above the tallest bar
        max_total = np.max(bottom_vals) if len(bottom_vals) > 0 else 1.0
        ax_obs.set_ylabel("Observed Number of\nPositive Drains", fontsize=17)
        ax_obs.set_ylim(bottom=0, top=max_total * 1.05) #controlar espacio en blanco en el bottom plot
        ax_obs.grid(True, alpha=0.25, axis='y')
        ax_obs.margins(x=0)
        ax_obs.tick_params(axis='y', labelsize=13)
        _setup_xaxis(ax_obs, date_min, date_max)

    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.align_ylabels()
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.1)
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.show()
    plt.close()

    print(f"  Saved: {filename}")

# ==============
# SIMULATION
# ==============


print(f"\n{'='*70}")
print(f"Mosquito Dynamics Simulation - All Years (2019-2024)")
print(f"Intervention mode: {INTERVENTION_MODE}")
print(f"{'='*70}\n")

# Prepare climate data
print("Preparing data...")
rainfall_24h, rainfall_21d, r0_vals = prepare_climate_all_years()
print(f"Simulation: {n_days} days (Jan 1, 2019 to Dec 31, 2024)")

# Compute initial conditions
u0 = compute_initial_conditions_global()

# Prepare interventions based on mode
if INTERVENTION_MODE == "uncontrolled":
    print("  Using NO INTERVENTION")
    interventions_list, intervention_starts = prepare_no_interventions()
    schedule = {z: [] for z in range(n_zones)}
    mode_tag = "uncontrolled"
    
elif INTERVENTION_MODE == "ASPB":
    print("  Using ASPB interventions (all treatments)")
    interventions_list, intervention_starts = prepare_ASPB_interventions_global(
        "ASPB_Interventions")
    schedule = extract_zone_schedule_from_interventions_global(
        interventions_list, intervention_starts)
    mode_tag = "ASPB"
    
elif INTERVENTION_MODE == "ASPB_Zona_risc":
    print("  Using ASPB zona-risc ALL visits (zone-level, merged <=2d)")
    interventions_list, intervention_starts = prepare_ASPB_interventions_global(
        "ASPB_Visits_Zona_risc")
    schedule = extract_zone_schedule_from_interventions_global(
        interventions_list, intervention_starts)
    mode_tag = "ASPB_Zona_risc"
    
else:
    raise ValueError(f"Unknown INTERVENTION_MODE: {INTERVENTION_MODE}. "
                    "For multi-year simulation, use 'uncontrolled', 'ASPB', or 'ASPB_Zona_risc'.")

# Print schedule summary
print("\n  Total interventions by zone:")
for z in range(n_zones):
    n_interv = len(schedule.get(z, []))
    print(f"    {zone_names[z]}: {n_interv} interventions")

# Build connectivity matrix (zone-restricted dense)
print("\nBuilding connectivity matrix...")
M = generate_M(D, ALPHAS, drain_to_zone, zone_members, zone_starts, n_drains)

# Run simulation
print("\nRunning simulation...")
solution = simulate_all_years(
    u0, n_drains, n_days, M, t_reformas,
    W_MIN, W_FLUSH, B, D_PARAM, DT,
    rainfall_24h, rainfall_21d, r0_vals,
    interventions_list, intervention_starts,
    drain_to_zone, zone_members, zone_starts, BASELINE
)
print(f"  Solution shape: {solution.shape}")

# Compute statistics (INT only, no PER for multi-year)
zone_integrals, zone_sums = compute_zone_integrals(solution, drain_to_zone, n_zones, n_days)
INT_total = compute_INT(zone_integrals)

# Print summary
print("\n" + "="*70)
print("SUMMARY STATISTICS (ALL YEARS)")
print("="*70)

print(f"\nZone integrals (INT):")
for z in range(n_zones):
    print(f"  {zone_names[z]}: {zone_integrals[z]:.2f}")
print(f"\n  TOTAL INT: {INT_total:.2f}")

# Max and mean
total = np.sum(solution, axis=1)
print(f"\nDynamics summary:")
print(f"  Max total drains: {np.max(total):.2f}")
print(f"  Mean total drains: {np.mean(total):.2f}")

# Yearly breakdown
print(f"\nYearly breakdown:")
years = [2019, 2020, 2021, 2022, 2023, 2024]
for year in years:
    d0_year = datetime(year, 1, 1)
    df_year = datetime(year, 12, 31)
    day_start = (d0_year - d0_global).days
    day_end = min((df_year - d0_global).days + 1, n_days)
    
    year_total = np.sum(zone_sums[day_start:day_end])
    year_max = np.max(np.sum(solution[day_start:day_end], axis=1))
    print(f"  {year}: INT={year_total:.2f}, Max={year_max:.2f}")

# Plot
print("\nGenerating plot...")
title = f"Mosquito Dynamics 2019-2024 - {INTERVENTION_MODE.replace('_', ' ')}"
filename = f"Dynamics_ALL_years_{mode_tag}{'_panel' if PANEL else ''}.pdf"

plot_stacked_zones_all_years(solution, n_days, title, filename, show_panel=PANEL)

print("\nDone!")

