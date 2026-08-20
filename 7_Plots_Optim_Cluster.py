#!/usr/bin/env python3
"""
7_Plots_Optim_Cluster.py

Plots optimization results from 6_Schedule_Optimization.py.
"""

import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# PARAMETERS - CHANGE THESE
# ==============================================================================

# --- Mode ---
# "single" : plot one specific year and beta (use YEAR and BETA_STR below)
# "all"    : loop over ALL_YEARS and ALL_BETAS and generate all plots
MODE = "all"

# Used only when MODE = "single"
YEAR = 2021
BETA_STR = "00"  # beta=0 -> "00", beta=0.1 -> "01", ..., beta=1 -> "10"

# Used only when MODE = "all"
ALL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
ALL_BETAS = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

SAVE_PLOTS = True  # Set to False to display plots without saving


# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "Results_Opt_Schedule"
FIGURES_DIR = BASE_DIR / "Figures" / "Optimization"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Zone colors (same order as 6_Schedule_Optimization.py)
ZONE_COLORS = OrderedDict([
    ("Parc de la Ciutadella", "#FDB462"),
    ("Parc del Turó de la Peira", "#FCCDE5"),
    ("Jardins Mossèn Cinto Verdaguer", "#80B1D3"),
    ("Jardins del Teatre Grec", "#BEBADA"),
    ("Jardins del Turó del Putxet", "#FB8072"),
    ("Jardins de Vil·la Amèlia - Cecília", "#8DD3C7"),
    ("Parc de la Guineueta", "#B3DE69"),
])

ZONE_NAMES = list(ZONE_COLORS.keys())
N_ZONES = len(ZONE_NAMES)


def beta_str_to_label(beta_str):
    """Convert beta string (00-10) to display label (0, 0.1, ..., 1)"""
    beta_val = int(beta_str) / 10
    if beta_val == int(beta_val):  # 0 or 1
        return str(int(beta_val))
    return str(beta_val)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_best_from_results(results_dict, beta_num=None):
    """
    Extract best result from a Results dictionary.
    Results[step] = [J0, init_times, periodicity, Temp, accepted]
    J0 = (cost, INT, PER)
    
    For beta=0: J = PER is constant, so select by min INT (index 1).
    For beta>0: select by min J (index 0).
    """
    if beta_num is not None and beta_num == 0.0:
        best_step = min(results_dict.keys(), key=lambda k: results_dict[k][0][1])  # min INT
    else:
        best_step = min(results_dict.keys(), key=lambda k: results_dict[k][0][0])  # min J
    best = results_dict[best_step]
    return {
        "step": best_step,
        "cost": best[0][0],
        "INT": best[0][1],
        "PER": best[0][2],
        "init_times": best[1],
        "periodicity": best[2],
        "total_steps": max(results_dict.keys()),
    }


def generate_schedule(init_times, periodicity, end_season):
    """Generate treatment schedule from init_times and periodicity"""
    schedule = {}
    for z in range(N_ZONES):
        t0 = int(init_times[z])
        period = max(1, int(periodicity[z]))
        times = []
        t = t0
        while t <= end_season:
            times.append(t)
            t += period
        schedule[ZONE_NAMES[z]] = times
    return schedule


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_J_track(all_results, year, beta_str, save=None):
    """Plot the cost function (J) evolution for all simulations
    For beta=00, plot INT; for beta=10, plot PER; otherwise plot J"""
    if save is None:
        save = SAVE_PLOTS
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine which component to plot based on beta
    if beta_str == "00" or beta_str == "10":
        component_idx = 1  # INT
        component_name = "J"
    else:
        component_idx = 0  # J (cost)
        component_name = "J"
    
    for i, (filename, results_dict) in enumerate(all_results):
        # Extract history for selected component
        steps = sorted(results_dict.keys())
        values = [results_dict[s][0][component_idx] for s in steps]
        total_steps = max(steps)
        ax.plot(steps, values, alpha=0.7, label=f"Run {i+1} ({total_steps} steps)")
    
    ax.set_xlabel("Step")
    ax.set_ylabel(component_name)
    ax.set_title(f"{component_name} Evolution - Year {year}, β = {beta_str_to_label(beta_str)}")
    
    if len(all_results) <= 10:
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"Track_{component_name}_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_schedule_histogram(best, year, beta_str, init_season, end_season, save=None):
    """Plot histogram of treatment times by zone"""
    if save is None:
        save = SAVE_PLOTS
    
    schedule = generate_schedule(best["init_times"], best["periodicity"], end_season)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_times = []
    labels = []
    colors = []
    
    for zone in ZONE_NAMES:
        if zone in schedule and len(schedule[zone]) > 0:
            # Convert to days from season start
            times_relative = [t - init_season for t in schedule[zone] if init_season <= t <= end_season]
            if times_relative:
                all_times.append(times_relative)
                labels.append(zone)
                colors.append(ZONE_COLORS[zone])
    
    if all_times:
        season_length = end_season - init_season
        bins = np.arange(0, season_length + 7, 7)
        ax.hist(all_times, bins=bins, stacked=True, label=labels, color=colors, 
                edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel("Days from season start (April 1)")
    ax.set_ylabel("Number of zone treatments")
    ax.set_title(f"Treatment Schedule - Year {year}, β = {beta_str_to_label(beta_str)}")
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"Schedule_histogram_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_first_treatment_day(best, year, beta_str, init_season, save=None):
    """Plot bar chart of first treatment day for each zone"""
    if save is None:
        save = SAVE_PLOTS
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(N_ZONES)
    colors = [ZONE_COLORS[zone] for zone in ZONE_NAMES]
    
    # Days from season start
    relative_init = [best["init_times"][i] - init_season for i in range(N_ZONES)]
    
    ax.bar(x, relative_init, color=colors, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(ZONE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Days from season start")
    ax.set_title(f"First Treatment Day - Year {year}, β = {beta_str_to_label(beta_str)}")
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"First_treatment_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_periodicity(best, year, beta_str, save=None):
    """Plot bar chart of treatment periodicity for each zone"""
    if save is None:
        save = SAVE_PLOTS
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(N_ZONES)
    colors = [ZONE_COLORS[zone] for zone in ZONE_NAMES]
    
    ax.bar(x, best["periodicity"], color=colors, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(ZONE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Days between treatments")
    ax.set_title(f"Treatment Periodicity - Year {year}, β = {beta_str_to_label(beta_str)}")
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"Periodicity_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_days_with_n_visits(best, year, beta_str, init_season, end_season, save=None):
    """Plot histogram showing how many days have N zones treated"""
    if save is None:
        save = SAVE_PLOTS
    
    schedule = generate_schedule(best["init_times"], best["periodicity"], end_season)
    
    # Flatten all treatment days within season
    all_days = []
    for zone_times in schedule.values():
        for t in zone_times:
            if init_season <= t <= end_season:
                all_days.append(t)
    
    if not all_days:
        return
    
    # Count visits per day
    day_counts = Counter(all_days)
    
    # Count how many days have each number of visits
    visit_freq = Counter(day_counts.values())
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = sorted(visit_freq.keys())
    y = [visit_freq[v] for v in x]
    
    ax.bar(x, y, color='steelblue', edgecolor='white')
    ax.set_xlabel("Number of zones treated")
    ax.set_ylabel("Number of days")
    ax.set_title(f"Days with N Zones Treated - Year {year}, β = {beta_str_to_label(beta_str)}")
    ax.set_xticks(x)
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"Days_with_N_visits_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_weekly_activity(best, year, beta_str, init_season, end_season, save=None):
    """Plot how many treatment days fit in each 7-day window"""
    if save is None:
        save = SAVE_PLOTS
    
    schedule = generate_schedule(best["init_times"], best["periodicity"], end_season)
    
    # Get all unique treatment days within season (sorted)
    all_days = sorted(set(
        int(t) for times in schedule.values() for t in times 
        if init_season <= t <= end_season
    ))
    
    if len(all_days) < 2:
        return
    
    # For each day, count how many treatment days fall in the 7-day window [day, day+7)
    treatments_per_week = []
    for day in all_days:
        count = sum(1 for d in all_days if day <= d < day + 7)
        treatments_per_week.append(count)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Count frequency of each value
    freq = Counter(treatments_per_week)
    x = sorted(freq.keys())
    y = [freq[v] for v in x]
    
    ax.bar(x, y, color='coral', edgecolor='black', width=0.8)
    ax.set_xlabel("Number of treatments in 7-day window")
    ax.set_ylabel("Count (how many such windows)")
    ax.set_title(f"Weekly Activity - Year {year}, β = {beta_str_to_label(beta_str)}")
    ax.set_xticks(x)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"Weekly_activity_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


def plot_all_sims_summary(all_results, year, beta_str, init_season, save=None):
    """Plot boxplots summarizing all simulations"""
    if save is None:
        save = SAVE_PLOTS
    
    if len(all_results) < 2:
        return
    
    # Collect init times and periodicities from all simulations
    all_init_times = [[] for _ in range(N_ZONES)]
    all_periodicities = [[] for _ in range(N_ZONES)]
    
    for filename, results_dict in all_results:
        beta_num = float(beta_str) / 10.0 if beta_str != "00" else 0.0
        best = extract_best_from_results(results_dict, beta_num=beta_num)
        for z in range(N_ZONES):
            all_init_times[z].append(best["init_times"][z] - init_season)
            all_periodicities[z].append(best["periodicity"][z])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = [ZONE_COLORS[zone] for zone in ZONE_NAMES]
    
    # Init times boxplot
    ax = axes[0]
    bp = ax.boxplot(all_init_times, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(ZONE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Days from season start")
    ax.set_title("First Treatment Day (all runs)")
    
    # Periodicity boxplot
    ax = axes[1]
    bp = ax.boxplot(all_periodicities, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(ZONE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Days between treatments")
    ax.set_title("Periodicity (all runs)")
    
    plt.suptitle(f"All Simulations Summary - Year {year}, β = {beta_str_to_label(beta_str)}")
    plt.tight_layout()
    
    if save:
        outpath = FIGURES_DIR / f"AllSims_summary_{year}_beta_{beta_str}.png"
        plt.savefig(outpath, dpi=150)
        print(f"    Saved: {outpath.name}")


# ==============================================================================
# MAIN
# ==============================================================================

def run_plots_for(year, beta_str):
    """Load results and generate all plots for a given year and beta."""
    d0 = datetime(year, 1, 1)
    init_season = (datetime(year, 4, 1) - d0).days
    end_season  = (datetime(year, 11, 30) - d0).days
    beta_num = float(beta_str) / 10.0

    files = list(RESULTS_DIR.glob(f"Simulated_Annealing_{year}_b{beta_str}_*"))
    if not files:
        print(f"  No results found for year {year}, beta={beta_str_to_label(beta_str)}")
        return

    print(f"  Found {len(files)} file(s)")

    all_results = []
    for filepath in files:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        all_results.append((filepath.name, data))

    # Find best result across all files
    best_overall = None
    best_file = None
    for filename, results_dict in all_results:
        best = extract_best_from_results(results_dict, beta_num=beta_num)
        if best_overall is None or best["cost"] < best_overall["cost"]:
            best_overall = best
            best_file = filename

    print(f"  Best: J={best_overall['cost']:.4f}, INT={best_overall['INT']:.4f}, "
          f"PER={best_overall['PER']:.4f}  (file: {best_file}, step {best_overall['step']})")

    # Generate all plots
    plot_J_track(all_results, year, beta_str)
    plot_schedule_histogram(best_overall, year, beta_str, init_season, end_season)
    plot_first_treatment_day(best_overall, year, beta_str, init_season)
    plot_periodicity(best_overall, year, beta_str)
    plot_days_with_n_visits(best_overall, year, beta_str, init_season, end_season)
    plot_weekly_activity(best_overall, year, beta_str, init_season, end_season)
    plot_all_sims_summary(all_results, year, beta_str, init_season)

    # Print per-zone details
    schedule = generate_schedule(best_overall["init_times"], best_overall["periodicity"], end_season)
    print(f"  {'Zone':<40} {'First':>8} {'Period':>8} {'Visits':>8}")
    print("  " + "-" * 70)
    total_visits = 0
    for z, zone in enumerate(ZONE_NAMES):
        first_day = best_overall["init_times"][z] - init_season
        period    = best_overall["periodicity"][z]
        n_visits  = len([t for t in schedule[zone] if init_season <= t <= end_season])
        total_visits += n_visits
        print(f"  {zone:<40} {first_day:>8.1f} {period:>8.1f} {n_visits:>8}")
    print("  " + "-" * 70)
    print(f"  {'Total':<40} {'':>8} {'':>8} {total_visits:>8}")


# ---- Dispatch ----------------------------------------------------------------

if MODE == "single":
    print("=" * 60)
    print(f"MODE=single  |  Year {YEAR}, β = {beta_str_to_label(BETA_STR)}")
    print(f"Save plots: {SAVE_PLOTS}")
    print("=" * 60)
    run_plots_for(YEAR, BETA_STR)
    plt.show()

elif MODE == "all":
    print("=" * 60)
    print(f"MODE=all  |  Years: {ALL_YEARS}")
    print(f"Betas: {[beta_str_to_label(b) for b in ALL_BETAS]}")
    print(f"Save plots: {SAVE_PLOTS}")
    print("=" * 60)
    for year in ALL_YEARS:
        for beta_str in ALL_BETAS:
            print(f"\nYear {year}, β = {beta_str_to_label(beta_str)}")
            run_plots_for(year, beta_str)
            plt.close('all')  # close after each combo to free memory

else:
    raise ValueError(f"Unknown MODE '{MODE}'. Use 'single' or 'all'.")

print("\nDone!")

