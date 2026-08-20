#!/usr/bin/env python3
"""
7_Crear_Df_Resultados_R.py

Reads optimization results from pickle files and creates a unified DataFrame
for analysis in R. Re-runs simulations to compute zone-specific integrals.
Also computes zone×year predictor variables for analysis in 7_Plots_Results_Final.R
(using the same data pipeline as the model).

The main CSV contains three types of rows:
  - Optimized simulations (ID = 1, 2, ...): from simulated annealing results
  - ASPB simulations     (ID = "ASPB"):     real-world ASPB treatment schedules
  - Untreated baseline   (ID = "Untreated"): no interventions (reference for risk reduction)

Output files:
    Figures/Simulations_Results.csv  — one row per (sim, zone), all three types
    Figures/Predictors.csv           — one row per (zone, year), predictor variables

Key columns:
    Zone_INT      — raw zone-level integral (positive drain-days)
    Norm_Zone_INT — Zone_INT / INT_ASPB(year). Additive decomposition of Norm_INT:
                    sum(Norm_Zone_INT) = Norm_INT for every simulation.
                    For ASPB rows, the 7 values sum to ~1.0.

Usage:
    python 7_Crear_Df_Resultados_R.py
"""

import os
import sys
import pickle
import re
import json
import functools
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)


# ==============================================================================
# PATHS
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "Dades_netes"
PARAM_DIR = SCRIPT_DIR / "Parameters"
RESULTS_DIR = SCRIPT_DIR / "Results_Opt_Schedule"
OUTPUT_CSV = SCRIPT_DIR / "Figures" / "Simulations_Results.csv"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ZONE COLORS (same order as 6_Schedule_Optimization.py - ground truth)
# ==============================================================================

zone_colors = OrderedDict([
    ("Parc de la Ciutadella", "#FDB462"),
    ("Parc del Turó de la Peira", "#FCCDE5"),
    ("Jardins Mossèn Cinto Verdaguer", "#80B1D3"),
    ("Jardins del Teatre Grec", "#BEBADA"),
    ("Jardins del Turó del Putxet", "#FB8072"),
    ("Jardins de Vil·la Amèlia - Cecília", "#8DD3C7"),
    ("Parc de la Guineueta", "#B3DE69")
])


# ==============================================================================
# MODEL PARAMETERS (from 6_Schedule_Optimization.py)
# ==============================================================================

ALPHAS = np.array([0.088, 0.097, 0.069, 0.14, 0.017, 0.19, 0.086], dtype=np.float64)
W_MIN = 11.4
W_FLUSH = 9.6
B = 1.9e-4
D_PARAM = 1e-5

DT = 0.30  #RK4 integration step
BASELINE = 0.001  # Per-drain baseline probability


# ==============================================================================
# NORMALIZATION DICT — ASPB_Visits_Zona_risc (zone-level, all visits, merged <=2d)
# ==============================================================================

NORM_DICT = {
    2019: {"int_ASPB": 3845.58, "per_ASPB": 90},
    2020: {"int_ASPB": 3502.45, "per_ASPB": 75},
    2021: {"int_ASPB": 274.65, "per_ASPB": 59},
    2022: {"int_ASPB": 1010.74, "per_ASPB": 67},
    2023: {"int_ASPB": 1579.46, "per_ASPB": 46},
    2024: {"int_ASPB": 717.86, "per_ASPB": 48},
}


# ==============================================================================
# LOAD STATIC DATA (once, shared across all years)
# ==============================================================================

print("Loading static data...")

# Climate data (full dataset)
rainfall_data = xr.open_dataset(PARAM_DIR / "Rainfall.nc")['data']
rainfall_21_data = xr.open_dataset(PARAM_DIR / "Rainfall_21.nc")['data']
r0_data = xr.open_dataset(PARAM_DIR / "R0_both.nc")['data']
temp_data = xr.open_dataset(PARAM_DIR / "Mean_Temperature.nc")['data']

# Static data
with open(PARAM_DIR / "D_Matrix", 'rb') as f:
    D_global = pickle.load(f)
with open(PARAM_DIR / "X_coords", 'rb') as f:
    X_coords_dict = pickle.load(f)
with open(PARAM_DIR / "Y_coords", 'rb') as f:
    Y_coords_dict = pickle.load(f)

unique_bs = pd.read_csv(DATA_DIR / "Unique_Items.csv")

n_drains = D_global.shape[0]

# Zone setup (order from zone_colors)
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


# ==============================================================================
# NUMBA FUNCTIONS (copied from 6_Schedule_Optimization.py)
# ==============================================================================

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
def simulate_with_schedule_zones(u0, n_drains, n_days, M,
                                  t_reformas, w_min, w_flush, B, D_param, dt,
                                  rainfall_24h, rainfall_21d, r0_values,
                                  zone_intervention_days, zone_intervention_counts,
                                  drain_to_zone, zone_members, zone_starts, baseline, n_zones):
    """
    Full simulation with scheduled interventions.
    Returns: (total_integral, zone_integrals)
    
    This is a modified version that returns zone-level integrals.
    """
    u = u0.copy()
    n_substeps = int(1.0 / dt)
    can_evolve = np.zeros(n_drains, dtype=np.bool_)
    intervention_cancelled_day = np.full(n_drains, -1, dtype=np.int64)
    
    # Store zone sums per day
    zone_sums = np.zeros((n_days, n_zones))
    
    # Day 0
    for i in range(n_drains):
        zone_idx = drain_to_zone[i]
        zone_sums[0, zone_idx] += u[i]
    
    for day in range(1, n_days):
        # Precompute zone sums at start of day (ASPB protocol: treat only if n_z >= 1)
        current_zone_sums = np.zeros(n_zones)
        for i in range(n_drains):
            current_zone_sums[drain_to_zone[i]] += u[i]
        
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            
            # 1. Reformed -> u=0
            if day > t_reformas[i]:
                u[i] = 0.0
                can_evolve[i] = False
                continue
            
            is_flushed = rainfall_24h[day, i] > w_flush
            
            # 2. Flushed -> cancels intervention
            if is_flushed:
                intervention_cancelled_day[i] = day
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            # 3. Check intervention
            # ASPB protocol: only treat if zone has n_z(t) >= 1 expected positives
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
            
            # 4. Dry -> baseline
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
        
        # Store zone sums
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            zone_sums[day, zone_idx] += u[i]
    
    # Compute zone integrals (trapezoidal)
    zone_integrals = np.zeros(n_zones)
    for z in range(n_zones):
        for day in range(n_days - 1):
            zone_integrals[z] += 0.5 * (zone_sums[day, z] + zone_sums[day + 1, z])
    
    # Total integral
    total_integral = 0.0
    for z in range(n_zones):
        total_integral += zone_integrals[z]
    
    return total_integral, zone_integrals


@jit(nopython=True)
def simulate_with_drain_interventions(u0, n_drains, n_days, M,
                                       t_reformas, w_min, w_flush, B, D_param, dt,
                                       rainfall_24h, rainfall_21d, r0_values,
                                       interventions_list, intervention_starts,
                                       drain_to_zone, zone_members, zone_starts, baseline, n_zones):
    """
    Simulation with DRAIN-LEVEL interventions (like 5_Simple_plot_ONE_year.py).
    Each drain has its own intervention days.
    Returns: (total_integral, zone_integrals)
    """
    u = u0.copy()
    n_substeps = int(1.0 / dt)
    can_evolve = np.zeros(n_drains, dtype=np.bool_)
    intervention_cancelled_day = np.full(n_drains, -1, dtype=np.int64)
    
    # Store zone sums per day
    zone_sums = np.zeros((n_days, n_zones))
    
    # Day 0
    for i in range(n_drains):
        zone_idx = drain_to_zone[i]
        zone_sums[0, zone_idx] += u[i]
    
    for day in range(1, n_days):
        # Precompute zone sums at start of day (ASPB protocol: treat only if n_z >= 1)
        current_zone_sums = np.zeros(n_zones)
        for i in range(n_drains):
            current_zone_sums[drain_to_zone[i]] += u[i]
        
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            
            # 1. Reformed -> u=0
            if day > t_reformas[i]:
                u[i] = 0.0
                can_evolve[i] = False
                continue
            
            is_flushed = rainfall_24h[day, i] > w_flush
            
            # 2. Flushed -> cancels intervention
            if is_flushed:
                intervention_cancelled_day[i] = day
                u[i] = baseline
                can_evolve[i] = False
                continue
            
            # 3. Check intervention (DRAIN-LEVEL)
            # ASPB protocol: only treat if zone has n_z(t) >= 1 expected positives
            is_under_intervention = False
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
            
            # 4. Dry -> baseline
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
        
        # Store zone sums
        for i in range(n_drains):
            zone_idx = drain_to_zone[i]
            zone_sums[day, zone_idx] += u[i]
    
    # Compute zone integrals (trapezoidal)
    zone_integrals = np.zeros(n_zones)
    for z in range(n_zones):
        for day in range(n_days - 1):
            zone_integrals[z] += 0.5 * (zone_sums[day, z] + zone_sums[day + 1, z])
    
    # Total integral
    total_integral = 0.0
    for z in range(n_zones):
        total_integral += zone_integrals[z]
    
    return total_integral, zone_integrals


# ==============================================================================
# SCHEDULE GENERATION (from 6_Schedule_Optimization.py)
# ==============================================================================

def generate_schedule(init_times, periodicity, n_zones, end_season):
    """Generate intervention days for each zone."""
    counts = np.zeros(n_zones, dtype=np.int32)
    
    max_interventions = 0
    for z in range(n_zones):
        t0 = int(init_times[z])
        period = max(1, int(periodicity[z]))
        n_inter = (end_season - t0) // period + 1
        counts[z] = n_inter
        if n_inter > max_interventions:
            max_interventions = n_inter
    
    schedule = np.full((n_zones, max_interventions), -1, dtype=np.int32)
    
    for z in range(n_zones):
        t0 = int(init_times[z])
        period = max(1, int(periodicity[z]))
        for k in range(counts[z]):
            schedule[z, k] = t0 + k * period
    
    return schedule, counts


def count_treatments(schedule, counts, n_zones, init_season, end_season):
    """Count total treatments in season."""
    n_treatments = 0
    for z in range(n_zones):
        for k in range(counts[z]):
            day = schedule[z, k]
            if init_season <= day <= end_season:
                n_treatments += 1
    return n_treatments


# ==============================================================================
# ZONE-RESTRICTED MATRIX GENERATION
# ==============================================================================

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


# ==============================================================================
# YEAR-SPECIFIC DATA SETUP
# ==============================================================================

def setup_year_data(year):
    """Prepare all year-specific data for simulation."""
    
    # Season bounds
    d0_year = datetime(year, 1, 1)
    init_date = datetime(year, 4, 1)
    init_mosq_season = (init_date - d0_year).days
    end_date = datetime(year, 11, 30)
    end_mosq_season = (end_date - d0_year).days
    
    # Year setup
    d0_global = datetime(2019, 1, 1)
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31)
    n_days = (year_end - year_start).days + 1
    global_offset = (year_start - d0_global).days
    
    # Interpolate climate data
    lons_da = xr.DataArray(X_emb, dims='points')
    lats_da = xr.DataArray(Y_emb, dims='points')
    
    rainfall_24h_full = rainfall_data.interp(longitude=lons_da, latitude=lats_da).values
    rainfall_21d_full = rainfall_21_data.interp(longitude=lons_da, latitude=lats_da).values
    r0_full = r0_data.interp(longitude=lons_da, latitude=lats_da).values
    
    rainfall_24h = np.nan_to_num(rainfall_24h_full[global_offset:global_offset + n_days], nan=0.0)
    rainfall_21d = np.nan_to_num(rainfall_21d_full[global_offset:global_offset + n_days], nan=0.0)
    r0_values = np.nan_to_num(r0_full[global_offset:global_offset + n_days], nan=0.00001)
    
    rainfall_24h = np.ascontiguousarray(rainfall_24h)
    rainfall_21d = np.ascontiguousarray(rainfall_21d)
    r0_values = np.ascontiguousarray(r0_values)
    
    # Reform dates
    t_reformas = np.zeros(n_drains, dtype=np.float64)
    for i in range(n_drains):
        reform_date = unique_bs['data_reforma'].iloc[i]
        if pd.notna(reform_date):
            reform_year = reform_date.year
            if reform_year < year or reform_year == year:
                t_reformas[i] = -1
            else:
                t_reformas[i] = n_days + 1
        else:
            t_reformas[i] = n_days + 1
    t_reformas = np.ascontiguousarray(t_reformas)
    
    # Zone-restricted coupling matrix M
    M = generate_M(D_global, ALPHAS, drain_to_zone, zone_members, zone_starts_arr, n_drains)
    
    # Per-drain initial condition
    u0 = np.zeros(n_drains, dtype=np.float64)
    for i in range(n_drains):
        if 0 <= t_reformas[i]:  # Active drain at day 0
            u0[i] = BASELINE
    u0 = np.ascontiguousarray(u0)
    
    return {
        'n_days': n_days,
        'init_mosq_season': init_mosq_season,
        'end_mosq_season': end_mosq_season,
        'rainfall_24h': rainfall_24h,
        'rainfall_21d': rainfall_21d,
        'r0_values': r0_values,
        't_reformas': t_reformas,
        'M': M,
        'u0': u0,
    }


# ==============================================================================
# FILE PARSING
# ==============================================================================

def parse_filename(filename):
    """
    Parse result filename: Simulated_Annealing_{year}_b{beta}_{timestamp}
    Returns: dict with year, beta_str, beta_num or None
    """
    pattern = r"Simulated_Annealing_(\d{4})_b(\d{2})_(.+)"
    match = re.match(pattern, filename)
    if match:
        year = int(match.group(1))
        beta_str = match.group(2)
        # beta_str is "01"-"10", meaning 0.1 to 1.0
        beta_num = float(beta_str) / 10.0
        return {"year": year, "beta_str": beta_str, "beta_num": beta_num}
    return None


def extract_best_from_results(data, beta_num=None):
    """
    Extract best result from pickle file.
    Structure: data[step] = [J_tuple, init_times, periodicity, Temp, accepted]
    J_tuple = (J, INT_normalized, PER_normalized)
    
    For beta=0: J = PER is constant (all schedules have same number of
    treatments), so we select by min INT (index 1) — same as 6_B0.
    For beta>0: select by min J (index 0).
    """
    if beta_num is not None and beta_num == 0.0:
        best_step = min(data.keys(), key=lambda k: data[k][0][1])  # min INT
    else:
        best_step = min(data.keys(), key=lambda k: data[k][0][0])  # min J
    best = data[best_step]
    
    return {
        "J": best[0][0],
        "INT_norm": best[0][1],
        "PER_norm": best[0][2],
        "init_times": np.array(best[1]),
        "periodicity": np.array(best[2]),
    }


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

print("=" * 60)
print("CREATING DATAFRAME FROM OPTIMIZATION RESULTS")
print("=" * 60)

# Check results directory
if not RESULTS_DIR.exists():
    print(f"Results directory not found: {RESULTS_DIR}")
    sys.exit(1)

# Get all files
files = [f for f in RESULTS_DIR.iterdir() if f.is_file() and f.name.startswith("Simulated_Annealing_")]
print(f"Found {len(files)} result files")

# Group files by year for efficient processing
files_by_year = {}
for filepath in files:
    info = parse_filename(filepath.name)
    if info:
        year = info["year"]
        if year not in files_by_year:
            files_by_year[year] = []
        files_by_year[year].append((filepath, info))

print(f"Years found: {sorted(files_by_year.keys())}")

# Process each year
rows = []
simulation_counter = {}

for year in sorted(files_by_year.keys()):
    print(f"\n--- Processing year {year} ({len(files_by_year[year])} files) ---")
    
    # Setup year data once
    year_data = setup_year_data(year)
    
    for filepath, info in sorted(files_by_year[year], key=lambda x: x[0].name):
        print(f"  {filepath.name}")
        
        beta_str = info["beta_str"]
        beta_num = info["beta_num"]
        
        # Assign unique ID
        key = (year, beta_str)
        simulation_counter[key] = simulation_counter.get(key, 0) + 1
        sim_id = simulation_counter[key]
        
        # Load result file
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"    Error loading: {e}")
            continue
        
        # Extract best result (beta=0 selects by INT, others by J)
        best = extract_best_from_results(data, beta_num=beta_num)
        init_times = best["init_times"]
        periodicity = best["periodicity"]
        
        # Generate schedule
        schedule, counts = generate_schedule(
            init_times, periodicity, n_zones, year_data['end_mosq_season']
        )
        
        # Run simulation to get zone integrals
        total_integral, zone_integrals = simulate_with_schedule_zones(
            year_data['u0'], n_drains, year_data['n_days'],
            year_data['M'],
            year_data['t_reformas'], W_MIN, W_FLUSH, B, D_PARAM, DT,
            year_data['rainfall_24h'], year_data['rainfall_21d'], year_data['r0_values'],
            schedule, counts,
            drain_to_zone, zone_members, zone_starts_arr, BASELINE, n_zones
        )
        
        # Compute metrics
        INT = total_integral  # Raw integral
        Norm_INT = INT / NORM_DICT[year]["int_ASPB"]
        
        PER = count_treatments(schedule, counts, n_zones,
                                year_data['init_mosq_season'], year_data['end_mosq_season'])
        Norm_PER = PER / NORM_DICT[year]["per_ASPB"]
        
        J = beta_num * Norm_INT + (1 - beta_num) * Norm_PER
        
        # Add one row per zone
        for z in range(n_zones):
            # Count zone visits within season
            t = int(init_times[z])
            period = max(1, int(periodicity[z]))
            zone_visits = 0
            while t <= year_data['end_mosq_season']:
                if t >= year_data['init_mosq_season']:
                    zone_visits += 1
                t += period
            
            row = {
                "ID": sim_id,
                "Year": year,
                "Beta": beta_str,
                "J": J,
                "INT": INT,
                "Norm_INT": Norm_INT,
                "PER": PER,
                "Norm_PER": Norm_PER,
                "Zone": zone_names[z],
                "Zone_INT": zone_integrals[z],
                "Norm_Zone_INT": zone_integrals[z] / NORM_DICT[year]["int_ASPB"],
                "Init_times": init_times[z],
                "Periodicity": periodicity[z],
                "Zone_Visits": zone_visits,
            }
            rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

if df.empty:
    print("\nNo valid results found!")
    sys.exit(1)

# Sort
df = df.sort_values(["Year", "Beta", "ID", "Zone"]).reset_index(drop=True)

# Save
df.to_csv(OUTPUT_CSV, index=False)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Unique simulations: {len(df.drop_duplicates(['Year', 'Beta', 'ID']))}")
print(f"Years: {sorted(df['Year'].unique())}")
print(f"Betas: {sorted(df['Beta'].unique())}")
print(f"\nOutput saved to: {OUTPUT_CSV}")

print("\nSample of output:")
print(df.head(14).to_string())


# ==========================================================================
# ASPB SIMULATIONS (ASPB_Zona_risc mode) - Zone-level ALL visits
# ==========================================================================

print("\n" + "=" * 60)
print("ASPB SIMULATIONS (Zona Risc) - Zone-level, all visits, merged <=2d")
print("=" * 60)

# Load ASPB visits (zone-level, all visits, merged <=2d back-to-back)
with open(PARAM_DIR / "ASPB_Visits_Zona_risc", 'rb') as f:
    aspb_interventions_dict = pickle.load(f)

print(f"Loaded ASPB_Visits_Zona_risc")

aspb_rows = []
years_to_process = [2019, 2020, 2021, 2022, 2023, 2024]

for year in years_to_process:
    print(f"\n--- Processing ASPB year {year} ---")
    
    # Setup year data
    year_data = setup_year_data(year)
    n_days = year_data['n_days']
    
    # Convert global intervention days to year-local days
    d0_global = datetime(2019, 1, 1)
    d0_year = datetime(year, 1, 1)
    df_year = datetime(year, 12, 31)
    day_start_global = (d0_year - d0_global).days
    day_end_global = (df_year - d0_global).days
    
    # Build drain-level intervention arrays (like 5_Simple_plot_ONE_year.py)
    interventions_list = []
    intervention_starts = np.zeros(n_drains + 1, dtype=np.int64)
    
    # Also track zone-level unique days for schedule/periodicity info
    zone_treatment_days = {z: set() for z in range(n_zones)}
    
    total_interventions = 0
    for i in range(n_drains):
        intervention_starts[i] = len(interventions_list)
        if i in aspb_interventions_dict:
            zone_idx = drain_to_zone[i]
            for global_day in aspb_interventions_dict[i]:
                if day_start_global <= global_day <= day_end_global:
                    local_day = global_day - day_start_global
                    interventions_list.append(local_day)
                    zone_treatment_days[zone_idx].add(local_day)
                    total_interventions += 1
    
    intervention_starts[n_drains] = len(interventions_list)
    interventions_list = np.array(interventions_list, dtype=np.int64)
    
    # Convert zone treatment days to sorted lists
    for z in range(n_zones):
        zone_treatment_days[z] = sorted(list(zone_treatment_days[z]))
    
    # Print schedule summary (zone-level unique days)
    total_zone_treatments = 0
    for z in range(n_zones):
        n_treat = len(zone_treatment_days[z])
        total_zone_treatments += n_treat
        print(f"  {zone_names[z]}: {n_treat} zone-level treatment days")
    print(f"  Total zone-level: {total_zone_treatments}, drain-level: {total_interventions}")
    
    # Run simulation with DRAIN-LEVEL interventions
    total_integral, zone_integrals = simulate_with_drain_interventions(
        year_data['u0'], n_drains, n_days,
        year_data['M'],
        year_data['t_reformas'], W_MIN, W_FLUSH, B, D_PARAM, DT,
        year_data['rainfall_24h'], year_data['rainfall_21d'], year_data['r0_values'],
        interventions_list, intervention_starts,
        drain_to_zone, zone_members, zone_starts_arr, BASELINE, n_zones
    )
    
    # Compute metrics
    INT = total_integral
    Norm_INT = INT / NORM_DICT[year]["int_ASPB"]
    PER = total_zone_treatments  # Use zone-level count for PER
    Norm_PER = PER / NORM_DICT[year]["per_ASPB"]
    
    # J is not well-defined for ASPB (no beta), set to NA
    J = np.nan
    
    print(f"  INT: {INT:.2f}, Norm_INT: {Norm_INT:.4f}")
    print(f"  PER: {PER}, Norm_PER: {Norm_PER:.4f}")
    
    # One row per zone (same structure as optimized simulations)
    # Zone_Visits = number of zone-level treatment days in this year
    # Periodicity = JSON array with ALL inter-treatment intervals (for distribution analysis)
    #               For 0 or 1 treatments: empty array "[]"
    #               For optimized sims this column is a single number; for ASPB it's a JSON array.
    
    for z in range(n_zones):
        treatments = zone_treatment_days[z]
        zone_visits = len(treatments)
        
        if zone_visits == 0:
            init_time = np.nan
            intervals = []
        elif zone_visits == 1:
            init_time = treatments[0]
            intervals = []
        else:
            init_time = treatments[0]
            intervals = [treatments[k+1] - treatments[k] for k in range(len(treatments)-1)]
        
        # Store as JSON string so we keep the full distribution
        periodicity_val = json.dumps(intervals)
        
        row = {
            "ID": "ASPB",
            "Year": year,
            "Beta": np.nan,
            "J": J,
            "INT": INT,
            "Norm_INT": Norm_INT,
            "PER": PER,
            "Norm_PER": Norm_PER,
            "Zone": zone_names[z],
            "Zone_INT": zone_integrals[z],
            "Norm_Zone_INT": zone_integrals[z] / NORM_DICT[year]["int_ASPB"],
            "Init_times": init_time,
            "Periodicity": periodicity_val,
            "Zone_Visits": zone_visits,
        }
        aspb_rows.append(row)

# Append ASPB rows to main dataframe
df_aspb = pd.DataFrame(aspb_rows)
df = pd.concat([df, df_aspb], ignore_index=True)

# Re-sort (ASPB will sort after numeric IDs due to string comparison)
df = df.sort_values(["Year", "Beta", "ID", "Zone"]).reset_index(drop=True)

# Save updated CSV (before Untreated)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("SUMMARY (including ASPB)")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Optimized simulations: {len(df[~df['ID'].isin(['ASPB'])].drop_duplicates(['Year', 'Beta', 'ID']))}")
print(f"ASPB simulations: {len(df[df['ID'] == 'ASPB']) // n_zones} (one per year)")
print(f"Years: {sorted(df['Year'].unique())}")

print("\nSample ASPB rows:")
print(df[df['ID'] == 'ASPB'].head(14).to_string())


# ==========================================================================
# UNTREATED SIMULATIONS (no interventions) — same structure as ASPB rows
# ==========================================================================
# Run the model with ZERO interventions for each year. This gives the baseline
# Zone_INT values against which we measure risk reduction.
# Previously these were computed only in the climate section below and saved
# to a separate CSV. Now they're part of the main Simulations_Results.csv,
# with ID="Untreated", for a unified data pipeline.

print("\n" + "=" * 60)
print("UNTREATED SIMULATIONS (no interventions)")
print("=" * 60)

untreated_rows = []
years_to_process_untreated = [2019, 2020, 2021, 2022, 2023, 2024]

for year in years_to_process_untreated:
    print(f"\n--- Processing Untreated year {year} ---")

    year_data = setup_year_data(year)

    # Empty schedule: 0 interventions per zone
    empty_schedule = np.full((n_zones, 1), -1, dtype=np.int32)
    empty_counts = np.zeros(n_zones, dtype=np.int32)

    total_integral, zone_integrals = simulate_with_schedule_zones(
        year_data['u0'], n_drains, year_data['n_days'],
        year_data['M'],
        year_data['t_reformas'], W_MIN, W_FLUSH, B, D_PARAM, DT,
        year_data['rainfall_24h'], year_data['rainfall_21d'], year_data['r0_values'],
        empty_schedule, empty_counts,
        drain_to_zone, zone_members, zone_starts_arr, BASELINE, n_zones
    )

    INT = total_integral
    Norm_INT = INT / NORM_DICT[year]["int_ASPB"]

    print(f"  INT: {INT:.2f}, Norm_INT: {Norm_INT:.4f}")

    for z in range(n_zones):
        row = {
            "ID": "Untreated",
            "Year": year,
            "Beta": np.nan,
            "J": np.nan,
            "INT": INT,
            "Norm_INT": Norm_INT,
            "PER": 0,
            "Norm_PER": 0.0,
            "Zone": zone_names[z],
            "Zone_INT": zone_integrals[z],
            "Norm_Zone_INT": zone_integrals[z] / NORM_DICT[year]["int_ASPB"],
            "Init_times": np.nan,
            "Periodicity": np.nan,
            "Zone_Visits": 0,
        }
        untreated_rows.append(row)

# Append Untreated rows
df_untreated = pd.DataFrame(untreated_rows)
df = pd.concat([df, df_untreated], ignore_index=True)

# Re-sort and save
df = df.sort_values(["Year", "Beta", "ID", "Zone"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("FINAL SUMMARY (Optimized + ASPB + Untreated)")
print("=" * 60)
print(f"Total rows: {len(df)}")
print(f"Optimized simulations: {len(df[~df['ID'].isin(['ASPB', 'Untreated'])].drop_duplicates(['Year', 'Beta', 'ID']))}")
print(f"ASPB simulations: {len(df[df['ID'] == 'ASPB']) // n_zones} (one per year)")
print(f"Untreated simulations: {len(df[df['ID'] == 'Untreated']) // n_zones} (one per year)")
print(f"Years: {sorted(df['Year'].unique())}")
print(f"\nOutput saved to: {OUTPUT_CSV}")

print("\nSample Untreated rows:")
print(df[df['ID'] == 'Untreated'].head(14).to_string())


# ==============================================================================
# PREDICTOR VARIABLES (Zone × Year)
# ==============================================================================
# Compute predictor variables at the zone×year level for 7_Plots_Results_Final.R.
# Uses the SAME reform logic and data interpolation as the simulation above.
# Output: Figures/Predictors.csv (one row per zone per year)

print("\n" + "=" * 60)
print("COMPUTING PREDICTOR VARIABLES (Zone × Year)")
print("=" * 60)

YEARS_ALL = [2019, 2020, 2021, 2022, 2023, 2024]

# Interpolate temperature and rainfall at drain locations (full time series, once)
print("  Interpolating climate data at drain locations...")
lons_da = xr.DataArray(X_emb, dims='points')
lats_da = xr.DataArray(Y_emb, dims='points')

temp_full = np.nan_to_num(temp_data.interp(longitude=lons_da, latitude=lats_da).values, nan=0.0)
rain_full = np.nan_to_num(rainfall_data.interp(longitude=lons_da, latitude=lats_da).values, nan=0.0)
rain21_full = np.nan_to_num(rainfall_21_data.interp(longitude=lons_da, latitude=lats_da).values, nan=0.0)
r0_full = np.nan_to_num(r0_data.interp(longitude=lons_da, latitude=lats_da).values, nan=0.0)

d0_global_dt = datetime(2019, 1, 1)

# Compute reform status for each drain × year (same logic as 6_Schedule_Optimization.py)
reform_years = np.array([unique_bs['data_reforma'].iloc[i].year for i in range(n_drains)])

predictors_rows = []

for yr in YEARS_ALL:
    yr_start = datetime(yr, 1, 1)
    yr_end = datetime(yr, 12, 31)
    n_days_yr = (yr_end - yr_start).days + 1
    global_offset = (yr_start - d0_global_dt).days

    # Active drains this year: reform_year <= yr means inactive from Jan 1
    is_active = reform_years > yr  # active if reform happens AFTER this year

    # Slice climate for the full year (Jan 1 – Dec 31)
    # Using the full year makes predictors comparable to Zone_INT (also full-year)
    # and ensures rank correlations reflect annual climate, not just the season window.
    temp_yr = temp_full[global_offset : global_offset + n_days_yr]   # (n_days_yr, n_drains)
    rain_yr = rain_full[global_offset : global_offset + n_days_yr]

    for z_idx in range(n_zones):
        all_drains = np.where(drain_to_zone == z_idx)[0]
        active_mask = is_active[all_drains]
        active_drains = all_drains[active_mask]
        n_active = len(active_drains)

        # Mean pairwise distance among active drains
        if n_active >= 2:
            sub_D = D_global[np.ix_(active_drains, active_drains)]
            triu_idx = np.triu_indices(n_active, k=1)
            mean_intra_dist = np.mean(sub_D[triu_idx])
        elif n_active == 1:
            mean_intra_dist = 0.0
        else:
            mean_intra_dist = np.nan

        # Mean temperature over season days and active drains
        if n_active > 0:
            mean_temp = np.mean(temp_yr[:, active_drains])
        else:
            mean_temp = np.nan

        # Total rainfall: sum of daily spatial-mean rainfall over active drains
        if n_active > 0:
            total_rainfall = np.sum(np.mean(rain_yr[:, active_drains], axis=1))
        else:
            total_rainfall = np.nan

        # Dry days fraction: proportion of full-year days where mean 21d rainfall < W_MIN.
        # Dry drains revert to baseline in the model (drains cannot support larvae).
        # np.mean() of a boolean array automatically computes the fraction (0 to 1),
        # correctly handling leap years (365 vs 366 days).
        if n_active > 0:
            rain21_yr = rain21_full[global_offset : global_offset + n_days_yr]
            daily_mean_rain21 = np.mean(rain21_yr[:, active_drains], axis=1)
            dry_days_frac = np.mean(daily_mean_rain21 < W_MIN)
        else:
            dry_days_frac = np.nan

        # Flush events: days where zone-mean daily rainfall > W_FLUSH
        # Flushing resets drains to baseline — a protective mechanism
        if n_active > 0:
            daily_mean_rain24 = np.mean(rain_yr[:, active_drains], axis=1)
            n_flush_events = int(np.sum(daily_mean_rain24 > W_FLUSH))
        else:
            n_flush_events = 0

        # Mean effective R0: mean R_M(T) over full-year days when drains are wet
        # (21d rain >= W_MIN). Zero on dry days. Averaged over active drains.
        # Controls for climate forcing; INT / eff_R0 isolates structural effects.
        # Full-year window matches Zone_INT and avoids arbitrary season truncation;
        # rank concordance with season-only is rho=0.999 (negligible difference).
        #
        # Mean_Temp_wet: mean temperature on wet days only (same mask).
        # Unlike Mean_Effective_R0, this is the raw thermal signal without the
        # nonlinear R0(T) transformation, enabling clean decomposition.
        if n_active > 0:
            r0_yr = r0_full[global_offset : global_offset + n_days_yr]
            r0_zone = r0_yr[:, active_drains]                     # (n_days_yr, n_active)
            rain21_zone = rain21_full[global_offset : global_offset + n_days_yr, :][:, active_drains]
            wet_mask = rain21_zone >= W_MIN                        # True when drain is wet
            effective_r0 = np.where(wet_mask, r0_zone, 0.0)
            mean_effective_r0 = float(np.mean(effective_r0))
            # Mean temperature on wet days only
            temp_zone = temp_yr[:, active_drains]                 # (n_days_yr, n_active)
            n_wet = wet_mask.sum()
            mean_temp_wet = float(np.mean(temp_zone[wet_mask])) if n_wet > 0 else np.nan
        else:
            mean_effective_r0 = np.nan
            mean_temp_wet = np.nan

        predictors_rows.append({
            "Zone": zone_names[z_idx],
            "Year": yr,
            "N_active_drains": n_active,
            "Mean_Intra_Dist": mean_intra_dist,
            "Mean_Temperature": mean_temp,
            "Mean_Temp_wet": mean_temp_wet,
            "Total_Rainfall": total_rainfall,
            "Dry_days_frac": dry_days_frac,
            "N_flush_events": n_flush_events,
            "Mean_Effective_R0": mean_effective_r0,
        })

predictors_df = pd.DataFrame(predictors_rows)

# Add Zone_INT_Untreated from the untreated simulations already computed above
untreated_lookup = df_untreated[["Zone", "Year", "Zone_INT"]].rename(
    columns={"Zone_INT": "Zone_INT_Untreated"}
)
predictors_df = predictors_df.merge(untreated_lookup, on=["Zone", "Year"])

# Free the big interpolated arrays
del temp_full, rain_full, rain21_full, r0_full

# Save
predictors_csv = SCRIPT_DIR / "Figures" / "Predictors.csv"
predictors_df.to_csv(predictors_csv, index=False)

# Print summary
print("\n--- Predictor variables (Zone × Year) ---")
print(f"{'Zone':<20s} {'Year':>4s} {'N_act':>5s} {'IntraDist':>9s} {'MeanTemp':>8s} "
      f"{'Rainfall':>8s} {'WetFrac':>7s} {'Flushes':>7s} {'EffR0':>7s} {'Untreated':>9s}")
for _, r in predictors_df.iterrows():
    name = r["Zone"][:20]
    print(f"{name:<20s} {r['Year']:4.0f} {r['N_active_drains']:5.0f} {r['Mean_Intra_Dist']:9.1f} "
          f"{r['Mean_Temperature']:8.2f} {r['Total_Rainfall']:8.1f} {r['Dry_days_frac']:7.1%} "
          f"{r['N_flush_events']:7.0f} {r['Mean_Effective_R0']:7.4f} "
          f"{r['Zone_INT_Untreated']:9.2f}")

print(f"\nSaved: {predictors_csv}")

