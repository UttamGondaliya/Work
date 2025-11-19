# anomaly_age_profiles.py
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import warnings
warnings.filterwarnings("ignore")

# --- Helper functions -------------------------------------------------------

def safe_style():
    # enforce safe plotting styles so ADTK / seaborn issues don't break plotting
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        plt.style.use("default")

def detect_breaks_pelt(signal, model="rbf", min_size=3, pen_values=None):
    """Run Pelt with a small grid of penalties and return the segmentation
       chosen by a conservative rule (prefer fewer breaks unless a clear change).
       Returns chosen_breaks (list of end indices), dict of pen->breaks.
    """
    if pen_values is None:
        pen_values = [1, 3, 5, 10, 20]
    results = {}
    for pen in pen_values:
        try:
            algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
            bkps = algo.predict(pen=pen)
            results[pen] = bkps
        except MemoryError:
            results[pen] = None
        except Exception as e:
            # fallback with l2 if rbf fails
            results[pen] = f"error: {e}"
    # pick pen with stable small number of breaks; prefer moderate complexity
    # rule: choose the segmentation with median number of breakpoints among non-error results,
    # but prefer fewer breaks if counts vary widely.
    valid = [(pen, bkps) for pen, bkps in results.items() if isinstance(bkps, list)]
    if not valid:
        return None, results
    counts = [(pen, max(0, len(bkps)-1)) for pen, bkps in valid]  # minus last = end index
    # Sort by number of breaks asc, pen asc
    counts_sorted = sorted(counts, key=lambda x: (x[1], x[0]))
    # Pick the median-small segmentation: take the one at 25th percentile of break counts if available
    break_counts = [c for _, c in counts_sorted]
    # if all same, pick the smallest pen among them
    if len(set(break_counts)) == 1:
        chosen_pen = counts_sorted[0][0]
    else:
        # choose a low-complexity reasonable segmentation: the one at 25% quantile of counts
        q = int(max(0, math.floor(len(counts_sorted) * 0.25)))
        chosen_pen = counts_sorted[q][0]
    chosen_breaks = dict(valid)[chosen_pen]
    return chosen_breaks, results

def iqr_point_anomalies(series, c=1.5):
    """Return boolean Series marking IQR-based single-point anomalies."""
    q1 = np.nanpercentile(series, 25)
    q3 = np.nanpercentile(series, 75)
    iqr = q3 - q1
    lower = q1 - c * iqr
    upper = q3 + c * iqr
    mask = (series < lower) | (series > upper)
    return mask

def monotonicity_issues(signal):
    """Return indices where prediction decreases (non-monotonic downward step)."""
    diffs = np.diff(signal)
    neg_idx = np.where(diffs < 0)[0] + 1  # +1 to map to the age index where drop occurs
    return neg_idx

def summarize_segments(signal, breaks):
    """Compute per-segment stats for interpretation."""
    segs = []
    start = 0
    for end in breaks:
        seg = signal[start:end]
        segs.append({
            "start_idx": start,
            "end_idx": end-1,
            "length": len(seg),
            "mean": float(np.nanmean(seg)) if len(seg)>0 else np.nan,
            "std": float(np.nanstd(seg)) if len(seg)>0 else np.nan,
            "min": float(np.nanmin(seg)) if len(seg)>0 else np.nan,
            "max": float(np.nanmax(seg)) if len(seg)>0 else np.nan
        })
        start = end
    return segs

# --- Main pipeline ----------------------------------------------------------

def analyze_profiles(df,
                     profile_col="profile",
                     age_col="age",
                     pred_col="pred_prob",
                     min_size=3,
                     pen_values=[1, 3, 5, 10, 20],
                     iqr_c=1.5,
                     extreme_break_fraction=0.10,
                     plot=True):
    """
    df: DataFrame with profile, age, pred_prob
    Returns summary dict keyed by profile
    """
    safe_style()
    profiles = df[profile_col].unique()
    summaries = {}
    for prof in profiles:
        print("\n" + "="*80)
        print(f"Profile: {prof}")
        sub = df[df[profile_col]==prof].copy()
        # ensure sorted by age
        sub = sub.sort_values(age_col)
        ages = sub[age_col].values
        signal = sub[pred_col].values.astype(float)
        N = len(signal)
        print(f"Observations: {N} (expected 80)")

        # --- 1. monotonicity check
        neg_idx = monotonicity_issues(signal)
        if len(neg_idx)==0:
            print("Monotonicity: no downward steps detected (pred monotonically non-decreasing).")
        else:
            print(f"Monotonicity: {len(neg_idx)} downward steps detected at ages: {ages[neg_idx].tolist()}")

        # --- 2. ruptures: detect breaks
        # try rbf first, fallback to l2 if needed
        try_models = ["rbf", "l2"]
        chosen_breaks = None
        all_results = {}
        for model in try_models:
            bkps, results = detect_breaks_pelt(signal, model=model, min_size=min_size, pen_values=pen_values)
            all_results[model] = results
            if bkps is not None:
                chosen_breaks = bkps
                used_model = model
                break
        if chosen_breaks is None:
            print("Ruptures failed to produce valid breakpoints for any model.")
            summaries[prof] = {"error":"ruptures_failed"}
            continue

        true_bkps = chosen_breaks[:-1]  # exclude final index which equals len(signal)
        num_breaks = len(true_bkps)
        print(f"Ruptures chosen model: {used_model}, breakpoints found: {num_breaks}, positions (age indices): {true_bkps}")
        # translate indices to ages
        break_ages = [int(ages[idx]) for idx in true_bkps] if len(true_bkps)>0 else []
        print(f"Break ages: {break_ages}")

        # is count extreme?
        extreme_flag = False
        if num_breaks > max(1, int(math.floor(extreme_break_fraction * N))):
            extreme_flag = True
            print(f"Extreme: number of breaks {num_breaks} > {int(extreme_break_fraction * 100)}% of observations ({int(extreme_break_fraction * N)})")
        # check many segments of length 1
        segs = summarize_segments(signal, chosen_breaks)
        tiny_segments = sum(1 for s in segs if s["length"] <= 1)
        if tiny_segments > 0:
            print(f"Note: {tiny_segments} tiny segments (length<=1) detected - could indicate overfitting or noisy signal.")

        # --- 3. point anomalies via IQR
        iqr_mask = iqr_point_anomalies(signal, c=iqr_c)
        iqr_ages = ages[iqr_mask].tolist()
        print(f"IQR point anomalies (c={iqr_c}): {len(iqr_ages)} points at ages {iqr_ages}")

        # --- 4. combine: anomalies near breaks or isolated spikes
        # Define 'near' as within window of 1 index
        near_break_anoms = []
        isolated_anoms = []
        for idx, flag in enumerate(iqr_mask):
            if not flag: continue
            # if within +/-1 of any breakpoint index, mark near_break_anoms
            if any(abs(idx - b) <= 1 for b in true_bkps):
                near_break_anoms.append(int(ages[idx]))
            else:
                isolated_anoms.append(int(ages[idx]))
        print(f"Point anomalies near breakpoints: {near_break_anoms}")
        print(f"Isolated point anomalies: {isolated_anoms}")

        # --- 5. ready summary structure
        summaries[prof] = {
            "N": N,
            "monotonicity_drops_idxs": neg_idx.tolist(),
            "ruptures": {
                "model_used": used_model,
                "break_indices": true_bkps,
                "break_ages": break_ages,
                "segments": segs
            },
            "iqr_anomalies": {
                "count": int(iqr_mask.sum()),
                "ages": iqr_ages,
                "near_break_ages": near_break_anoms,
                "isolated_ages": isolated_anoms
            },
            "extreme_flag": extreme_flag
        }

        # --- 6. plotting
        if plot:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(ages, signal, marker='o', linewidth=1.25, label='pred_prob')
            # mark IQR anomalies
            if len(iqr_ages) > 0:
                ax.scatter(iqr_ages, signal[np.isin(ages, iqr_ages)], s=110, facecolors='none', edgecolors='red', label='IQR anomaly')
            # mark ruptures break verticals (at the age where break occurs)
            for bidx in true_bkps:
                if bidx < N:
                    ax.axvline(x=ages[bidx], color='k', linestyle='--', alpha=0.7)
                    ax.text(ages[bidx], ax.get_ylim()[1], f" break@{ages[bidx]}", rotation=90, va='top', ha='right', fontsize=8)
            # mark monotonic drops
            if len(neg_idx)>0:
                ax.scatter(ages[neg_idx], signal[neg_idx], s=90, marker='x', color='orange', label='monotonic drop')
            ax.set_xlabel("age")
            ax.set_ylabel("pred_prob")
            ax.set_title(f"Profile: {prof}  |  N={N}  |  breaks={len(true_bkps)}  |  iqr_anoms={int(iqr_mask.sum())}")
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

    print("\n" + "="*80)
    print("Analysis complete for all profiles.")
    return summaries

# --- Example usage ----------------------------------------------------------
# Prepare df before calling analyze_profiles:
# df should have 800 rows: 10 profiles x ages 1..80 repeated per profile design
# Columns: 'profile', 'age', 'pred_prob'
#
# Example synthetic df snippet:
# profiles = [f"profile_{i}" for i in range(10)]
# rows = []
# for p in profiles:
#     for age in range(1,81):
#         rows.append({"profile": p, "age": age, "pred_prob": model.predict(... )})
# df = pd.DataFrame(rows)
#
# Then call:
# summaries = analyze_profiles(df, profile_col="profile", age_col="age", pred_col="pred_prob")

# If you want me to run this on your actual df here, paste df.head() or the data.
