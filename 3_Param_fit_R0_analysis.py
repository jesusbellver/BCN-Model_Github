import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "Parameter_estimation"
FIGURES_DIR = RESULTS_DIR / "Figures"
FIGURES_DIR.mkdir(exist_ok=True)

species = "both"
MODE = "fit"
WEIGHTING = "balanced"
RESULTS_FILE = RESULTS_DIR / f"Estimated_params_{species}_nll_{MODE}_{WEIGHTING}_4"

# Only show runs with NLL <= Xth percentile. None = show all.
PERCENTILE_FILTER = 50

# ============================================================================
# LOAD RESULTS
# ============================================================================

print(f"Loading: {RESULTS_FILE}")
if not RESULTS_FILE.exists():
    print("ERROR: Results file not found.")
    sys.exit(1)

with open(RESULTS_FILE, 'rb') as f:
    results = pickle.load(f)

best_params_full    = results['best_params']
best_loss           = results.get('best_nll', results.get('best_mse'))
free_indices        = results['free_indices']
fixed_indices       = results['fixed_indices']
fixed_values        = results['fixed_values']
n_total             = results['n_total_params']
exploration_results = results['exploration_results']
refinement_results  = results.get('refinement_results', [])
baseline_nll        = results.get('baseline_nll', {})

zone_short = ["Ciutadella", "Turó Peira", "Mossèn Cinto", "Teatre Grec",
              "Turó Putxet", "Vil·la Amèlia", "Guineueta"]
n_zones = len(zone_short)

all_names  = [f"α_{i} ({zone_short[i]})" for i in range(n_zones)] + ['w_min', 'w_flush', 'B', 'D']
all_labels = [rf"$\alpha_{i}$ {zone_short[i]}" for i in range(n_zones)] + \
             [r'$w_{min}$', r'$w_{flush}$', r'$B$', r'$D$']

free_names  = [all_names[i]  for i in free_indices]
free_labels = [all_labels[i] for i in free_indices]
n_free      = len(free_indices)

# ============================================================================
# EXTRACT EXPLORATION DATA
# ============================================================================

explore_params = []
explore_losses = []
for r in exploration_results:
    full = np.zeros(n_total)
    for j, idx in enumerate(free_indices):
        full[idx] = r['xbest'][j]
    for idx, val in fixed_values.items():
        full[idx] = val
    explore_params.append(full)
    explore_losses.append(r['fbest'])

explore_params = np.array(explore_params)
explore_losses = np.array(explore_losses)
n_runs         = len(explore_losses)

# Apply percentile filter (keeps runs with NLL <= Nth percentile value)
if PERCENTILE_FILTER is not None:
    nll_threshold = np.percentile(explore_losses, PERCENTILE_FILTER)
    pct_mask = explore_losses <= nll_threshold
    explore_params = explore_params[pct_mask]
    explore_losses = explore_losses[pct_mask]
    n_runs         = len(explore_losses)
    print(f"  Percentile filter: keeping NLL <= {nll_threshold:.4f} (p{PERCENTILE_FILTER}), {n_runs} runs remaining")

print(f"  {n_runs} exploration runs, NLL [{explore_losses.min():.4f}, {explore_losses.max():.4f}]")

# ============================================================================
# UNCERTAINTY ESTIMATES  —  profile-likelihood confidence intervals
# ============================================================================

DELTA_NLL = 1.92   # χ², 95 % profile-likelihood threshold

C = None  # kept for Figure 4 correlation matrix (still uses CMA-ES cov if available)
if refinement_results:
    best_ref = sorted(refinement_results, key=lambda r: r.get('final_loss', 1e10))[0]
    C = best_ref.get('covariance')
    if C is not None:
        C = np.array(C)

best_free = np.array([best_params_full[i] for i in free_indices])

print(f"\n  Raw best_params_full: {best_params_full}")
print(f"  best_free (free params only): {best_free}")

# Profile-likelihood CI per free parameter
ci_lo  = np.empty(n_free)
ci_hi  = np.empty(n_free)
ci_mask_counts = np.empty(n_free, dtype=int)

for j, idx in enumerate(free_indices):
    mask = explore_losses <= best_loss + DELTA_NLL
    if mask.sum() < 2:
        mask = np.ones(n_runs, dtype=bool)   # fallback: all runs
    pvals = explore_params[mask, idx]
    ci_lo[j] = pvals.min()
    ci_hi[j] = pvals.max()
    ci_mask_counts[j] = mask.sum()

# stds kept for Figure 1 bar heights (use half CI width as a display stand-in)
stds = (ci_hi - ci_lo) / 2.0

print(f"  Profile-likelihood CIs  (ΔNLL ≤ {DELTA_NLL}, χ²₁ 95 %)")
print(f"\n  {'Param':<28} {'Best':>12}  {'CI low':>10}  {'CI high':>10}  {'n_runs':>7}")
print("  " + "-"*72)
for i in range(n_free):
    print(f"  {free_names[i]:<28} {best_free[i]:>12.4g}  "
          f"{ci_lo[i]:>10.4g}  {ci_hi[i]:>10.4g}  {ci_mask_counts[i]:>7d}")

if baseline_nll:
    print(f"\n  NLL baselines  —  all-zeros: {baseline_nll.get('nll_all_zeros','?'):.4f}  "
          f"global-mean: {baseline_nll.get('nll_global_mean','?'):.4f}  "
          f"zone-means: {baseline_nll.get('nll_zone_mean','?'):.4f}")
    print(f"  Model best: {best_loss:.4f}")

# ============================================================================
# FIGURE 1 — Parameter estimates + uncertainty
#
# Best-fit value with ±2σ error bar.  Color encodes normalized uncertainty
# (σ / bound-range): red = high uncertainty / poorly identified,
# green = low uncertainty / well identified.
# Sorted most-uncertain first so the eye immediately sees what's constrained.
# ============================================================================

# Physical bound ranges (for normalizing uncertainty)
# alpha: (0.001, 0.1), w_min: (1,50), w_flush: (5,50), B: (1e-5,5e-3), D: (0.05,5)
phys_ranges_all = [(0.001, 0.1)] * n_zones + [(1.0, 50.0), (5.0, 50.0), (1e-5, 5e-3), (0.05, 5.0)]
bound_ranges = np.array([phys_ranges_all[i][1] - phys_ranges_all[i][0] for i in free_indices])
norm_unc     = stds / (bound_ranges + 1e-15)

sort_order = np.argsort(norm_unc)[::-1]   # most uncertain first
colors      = plt.cm.RdYlGn_r(norm_unc[sort_order] / (norm_unc.max() + 1e-15))

fig1, ax1 = plt.subplots(figsize=(12, 5))
xs = np.arange(n_free)
ax1.bar(xs, best_free[sort_order], color=colors, alpha=0.85,
        edgecolor='black', linewidth=0.6, zorder=3)

# Asymmetric error bars: distance from best value to CI bound (clamp to 0)
yerr_lo = np.maximum(0.0, (best_free - ci_lo)[sort_order])
yerr_hi = np.maximum(0.0, (ci_hi - best_free)[sort_order])
ax1.errorbar(xs, best_free[sort_order], yerr=[yerr_lo, yerr_hi],
             fmt='none', color='black', capsize=5, linewidth=1.5, zorder=4)
ax1.set_xticks(xs)
ax1.set_xticklabels([free_labels[i] for i in sort_order], rotation=40, ha='right', fontsize=9)
ax1.set_ylabel('Parameter value', fontsize=11)
ax1.set_title(f'Best-fit parameters with 95% profile-likelihood CI  (ΔNLL ≤ {DELTA_NLL})  '
              f'(red = poorly identified, green = well identified)', fontsize=11)
ax1.grid(True, alpha=0.25, axis='y', zorder=0)

for k, orig_idx in enumerate(sort_order):
    v = best_free[orig_idx]
    lo, hi = ci_lo[orig_idx], ci_hi[orig_idx]
    ax1.text(k, hi * 1.05 + abs(v) * 0.02,
             f'{v:.3g}\n[{lo:.2g}, {hi:.2g}]',
             ha='center', va='bottom', fontsize=7, color='#333333')

plt.tight_layout()
fig1.savefig(FIGURES_DIR / f"1_param_estimates_{species}.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: 1_param_estimates_{species}.png")

# ============================================================================
# FIGURE 2 — CMA-ES exploration progress
#
# Left:  runs sorted by NLL with baseline reference lines — gives a feeling
#        of how much the optimizer improved over random starting points.
# Right: NLL distribution — shows if runs converged to a tight region
#        (peaked histogram) or spread out (flat = multi-modal landscape).
# ============================================================================

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 4.5))

sorted_idx  = np.argsort(explore_losses)
rank_colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, n_runs))

ax2a.bar(np.arange(n_runs), explore_losses[sorted_idx], color=rank_colors,
         edgecolor='black', linewidth=0.3)
ax2a.axhline(best_loss, color='red', linewidth=1.5, label=f'Best = {best_loss:.4f}')
if baseline_nll:
    ax2a.axhline(baseline_nll.get('nll_zone_mean', np.nan), color='orange',
                 linestyle='--', linewidth=1.4, label='Zone-mean baseline')
    ax2a.axhline(baseline_nll.get('nll_global_mean', np.nan), color='gray',
                 linestyle=':', linewidth=1.2, label='Global-mean baseline')
ax2a.set_xlabel('Run (sorted by NLL)', fontsize=10)
ax2a.set_ylabel('NLL', fontsize=10)
ax2a.set_title('Exploration runs (sorted)', fontsize=11)
ax2a.legend(fontsize=8)
ax2a.grid(True, alpha=0.25, axis='y')

ax2b.hist(explore_losses, bins=max(5, n_runs // 3), color='steelblue',
          alpha=0.75, edgecolor='black', linewidth=0.5)
ax2b.axvline(best_loss, color='red', linewidth=2, label=f'Best = {best_loss:.4f}')
if baseline_nll:
    ax2b.axvline(baseline_nll.get('nll_zone_mean', np.nan), color='orange',
                 linestyle='--', linewidth=1.4, label='Zone-mean')
ax2b.set_xlabel('NLL', fontsize=10)
ax2b.set_ylabel('Count', fontsize=10)
ax2b.set_title('NLL distribution', fontsize=11)
ax2b.legend(fontsize=8)
ax2b.grid(True, alpha=0.25, axis='y')

plt.tight_layout()
fig2.savefig(FIGURES_DIR / f"2_exploration_progress_{species}.png", dpi=150, bbox_inches='tight')
print(f"Saved: 2_exploration_progress_{species}.png")

# ============================================================================
# FIGURE 3 — NLL landscape slices per free parameter
#
# Each panel shows NLL vs the value of one parameter across all runs.
# V-shape / clear minimum = well-identified.
# Flat cloud = unidentifiable from the data.
# ============================================================================

n_cols = 4
n_rows = (n_free + n_cols - 1) // n_cols
fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
axes3 = axes3.flatten()

# Precompute weighted mean per free parameter (weight = exp(-NLL) ∝ likelihood)
weights_exp = np.exp(-(explore_losses - explore_losses.min()))
weights_exp /= weights_exp.sum()

for j, idx in enumerate(free_indices):
    ax = axes3[j]
    pvals = explore_params[:, idx]

    # CI mask (same as used for printed table)
    ci_mask = explore_losses <= best_loss + DELTA_NLL
    if ci_mask.sum() < 2:
        ci_mask = np.ones(n_runs, dtype=bool)

    # Shaded CI region (spans full y-range of the panel)
    ax.axvspan(ci_lo[j], ci_hi[j], color='steelblue', alpha=0.15, zorder=1,
               label=f'95% CI [{ci_lo[j]:.3g}, {ci_hi[j]:.3g}]')

    # Scatter points, coloured by NLL
    ax.scatter(pvals, explore_losses, c=explore_losses, cmap='viridis_r',
               s=35, alpha=0.8, edgecolors='black', linewidth=0.3, zorder=3)

    # Horizontal line at best NLL + threshold
    ax.axhline(best_loss + DELTA_NLL, color='steelblue', linestyle=':', linewidth=1.0,
               alpha=0.7, zorder=2, label=f'NLL threshold (+{DELTA_NLL})')

    # Vertical line: best parameter value
    ax.axvline(best_params_full[idx], color='red', linestyle='--', linewidth=1.5,
               zorder=4, label=f'Best = {best_params_full[idx]:.3g}')

    # Vertical line: likelihood-weighted mean
    wmean = np.sum(weights_exp * pvals)
    ax.axvline(wmean, color='darkorange', linestyle='-', linewidth=1.5,
               zorder=4, label=f'Mean = {wmean:.3g}')

    ax.set_xlabel(free_labels[j], fontsize=9)
    ax.set_ylabel('NLL', fontsize=9)
    ax.set_title(free_names[j], fontsize=9)
    ax.legend(fontsize=6.5, loc='upper center', framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.get_major_formatter().set_scientific(True)

for j in range(n_free, len(axes3)):
    axes3[j].axis('off')

fig3.suptitle(
    'NLL landscape slices  |  red dashed = best  |  orange = likelihood-weighted mean  |  blue shading = 95% CI',
    fontsize=11, y=1.01)
plt.tight_layout()
fig3.savefig(FIGURES_DIR / f"3_loss_landscape_{species}.png", dpi=150, bbox_inches='tight')
print(f"Saved: 3_loss_landscape_{species}.png")

# ============================================================================
# FIGURE 4 — Parameter correlation matrix
#
# From CMA-ES covariance if available, otherwise Pearson
# correlation across exploration runs.
# Strong off-diagonal correlations mean those parameters trade off —
# you can't identify them individually, only their combination.
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(9, 7))

if C is not None and C.shape[0] == n_free:
    stds_c = np.sqrt(np.diag(C)) + 1e-15
    corr   = np.clip(C / np.outer(stds_c, stds_c), -1, 1)
    title_suffix = "(from CMA-ES covariance)"
else:
    mat  = np.array([explore_params[:, idx] for idx in free_indices])
    corr = np.corrcoef(mat)
    title_suffix = "(Pearson across exploration runs)"

im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax4.set_xticks(range(n_free))
ax4.set_yticks(range(n_free))
ax4.set_xticklabels(free_labels, rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(free_labels, fontsize=8)
ax4.set_title(f'Parameter correlations {title_suffix}', fontsize=11)
plt.colorbar(im, ax=ax4, label='Correlation')

for i in range(n_free):
    for j in range(n_free):
        c_val     = corr[i, j]
        txt_color = 'white' if abs(c_val) > 0.6 else 'black'
        ax4.text(j, i, f'{c_val:.2f}', ha='center', va='center',
                 fontsize=6, color=txt_color)

plt.tight_layout()
fig4.savefig(FIGURES_DIR / f"4_param_correlations_{species}.png", dpi=150, bbox_inches='tight')
print(f"Saved: 4_param_correlations_{species}.png")

# ============================================================================
# DONE
# ============================================================================

print(f"\nAll figures saved to: {FIGURES_DIR}")
plt.show()
