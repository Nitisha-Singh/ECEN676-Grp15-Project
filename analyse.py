import os
import re
import pandas as pd
import numpy as np
import math
from scipy.stats import gmean
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULTS_DIR = "results"
CONFIGS = ["nop", "epi_2k", "epi_4k", "epi_8k", "epi_16k", "next", "mana", "djolt", "fnlmma"]
BASELINE_CONFIG = "nop"
OUTPUT_EXCEL_FILE = "simulation_analysis_results.xlsx"
GENERATE_PLOTS = True  # Set to False to skip plot generation

# Extract workload names from one of the folders
try:
    workload_files = [f for f in os.listdir(os.path.join(RESULTS_DIR, CONFIGS[0])) if f.endswith('.txt')]
    if not workload_files:
        raise FileNotFoundError(f"No .txt files found in {RESULTS_DIR}/{CONFIGS[0]}")
    WORKLOADS = sorted([f.replace('.txt', '') for f in workload_files])
except FileNotFoundError:
    print(f"Error: Base directory '{RESULTS_DIR}/{CONFIGS[0]}' not found or contains no result files.")
    print("Please ensure the directory structure and file naming are correct.")
    exit()
except Exception as e:
    print(f"An error occurred getting workload names: {e}")
    exit()

print(f"Found workloads: {WORKLOADS}")
print(f"Analyzing configurations: {CONFIGS}")

# --- Regular Expressions to Extract Data ---
patterns = {
    # Core performance metrics
    "ipc": re.compile(r"CPU\s+\d+\s+cumulative\s+IPC:\s+(\d+\.?\d*)"),
    "instructions": re.compile(r"CPU\s+\d+\s+cumulative\s+IPC:.*?instructions:\s+(\d+)"),
    "cycles": re.compile(r"CPU\s+\d+\s+cumulative\s+IPC:.*?cycles:\s+(\d+)"),
    "wp_cycles": re.compile(r"CPU\s+\d+\s+cumulative\s+IPC:.*?wp_cycles:\s+(\d+)"),
    
    # Branch prediction metrics
    "branch_prediction_accuracy": re.compile(r"CPU\s+\d+\s+Branch\s+Prediction\s+Accuracy:\s+(\d+\.?\d*)%"),
    "branch_mpki": re.compile(r"CPU\s+\d+\s+Branch\s+Prediction\s+Accuracy:.*?MPKI:\s+(\d+\.?\d*)"),
    "rob_occupancy": re.compile(r"CPU\s+\d+\s+Branch\s+Prediction\s+Accuracy:.*?Average\s+ROB\s+Occupancy\s+at\s+Mispredict:\s+(\d+\.?\d*)"),
    
    # Wrong path metrics
    "wrong_path_insts": re.compile(r"CPU\s+\d+\s+wrong_path_insts:\s+(\d+)"),
    "wrong_path_skipped": re.compile(r"CPU\s+\d+\s+wrong_path_insts:.*?wrong_path_insts_skipped:\s+(\d+)"),
    "wrong_path_executed": re.compile(r"CPU\s+\d+\s+wrong_path_insts:.*?wrong_path_insts_executed:\s+(\d+)"),
    
    # Footprint metrics
    "instr_foot_print": re.compile(r"CPU\s+\d+\s+instr_foot_print:\s+(\d+)"),
    "data_foot_print": re.compile(r"CPU\s+\d+\s+data_foot_print:\s+(\d+)"),
    
    # Prefetch metrics
    "is_prefetch_insts": re.compile(r"CPU\s+\d+\s+is_prefetch_insts:\s+(\d+)"),
    "is_prefetch_skipped": re.compile(r"CPU\s+\d+\s+is_prefetch_insts:.*?is_prefetch_skipped:\s+(\d+)"),
    
    # L1I cache metrics
    "l1i_total_access": re.compile(r"cpu\d+_L1I\s+TOTAL\s+ACCESS:\s+(\d+)"),
    "l1i_total_hit": re.compile(r"cpu\d+_L1I\s+TOTAL\s+ACCESS:.*?HIT:\s+(\d+)"),
    "l1i_total_miss": re.compile(r"cpu\d+_L1I\s+TOTAL\s+ACCESS:.*?MISS:\s+(\d+)"),
    "l1i_load_access": re.compile(r"cpu\d+_L1I\s+LOAD\s+ACCESS:\s+(\d+)"),
    "l1i_load_hit": re.compile(r"cpu\d+_L1I\s+LOAD\s+ACCESS:.*?HIT:\s+(\d+)"),
    "l1i_load_miss": re.compile(r"cpu\d+_L1I\s+LOAD\s+ACCESS:.*?MISS:\s+(\d+)"),
    "l1i_rfo_access": re.compile(r"cpu\d+_L1I\s+RFO\s+ACCESS:\s+(\d+)"),
    "l1i_rfo_hit": re.compile(r"cpu\d+_L1I\s+RFO\s+ACCESS:.*?HIT:\s+(\d+)"),
    "l1i_rfo_miss": re.compile(r"cpu\d+_L1I\s+RFO\s+ACCESS:.*?MISS:\s+(\d+)"),
    "l1i_prefetch_access": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+ACCESS:\s+(\d+)"),
    "l1i_prefetch_hit": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+ACCESS:.*?HIT:\s+(\d+)"),
    "l1i_prefetch_miss": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+ACCESS:.*?MISS:\s+(\d+)"),
    
    # Prefetch details
    "l1i_prefetch_requested": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+REQUESTED:\s+(\d+)"),
    "l1i_prefetch_issued": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+REQUESTED:.*?ISSUED:\s+(\d+)"),
    "l1i_prefetch_useful": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+REQUESTED:.*?USEFUL:\s+(\d+)"),
    "l1i_prefetch_useless": re.compile(r"cpu\d+_L1I\s+PREFETCH\s+REQUESTED:.*?USELESS:\s+(\d+)"),
    
    # Average latency metrics
    "l1i_avg_miss_latency": re.compile(r"cpu\d+_L1I\s+AVERAGE\s+MISS\s+LATENCY:\s+(\d+\.?\d*)"),
    "l1i_avg_instr_miss_latency": re.compile(r"cpu\d+_L1I\s+AVERAGE\s+INSTR\s+MISS\s+LATENCY:\s+(\d+\.?\d*)"),
    "l1i_avg_data_miss_latency": re.compile(r"cpu\d+_L1I\s+AVERAGE\s+DATA\s+MISS\s+LATENCY:\s+(\d+\.?\d*)")
}

# --- Data Storage ---
# Use dictionaries to store DataFrames for each raw metric
raw_data = {metric: pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float)
            for metric in patterns.keys()}

# --- Data Extraction Loop ---
for config in CONFIGS:
    config_dir = os.path.join(RESULTS_DIR, config)
    if not os.path.isdir(config_dir):
        print(f"Warning: Directory not found for config '{config}', skipping.")
        continue
        
    for workload in WORKLOADS:
        file_path = os.path.join(config_dir, f"{workload}.txt")
        print(f"Processing: {file_path}")
        
        if not os.path.exists(file_path):
            print(f" Warning: File not found, setting NaNs for {config}/{workload}")
            for metric in patterns.keys():
                raw_data[metric].loc[workload, config] = np.nan
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Extract each metric
            for metric, pattern in patterns.items():
                match = pattern.search(content)
                if match:
                    value = float(match.group(1))
                    raw_data[metric].loc[workload, config] = value
                else:
                    print(f" Warning: Metric '{metric}' not found in {file_path}")
                    raw_data[metric].loc[workload, config] = np.nan
                    
        except Exception as e:
            print(f" Error processing file {file_path}: {e}")
            for metric in patterns.keys():
                raw_data[metric].loc[workload, config] = np.nan

# --- Calculate Derived Metrics ---
# Combine L1I demand misses
raw_data['l1i_demand_miss'] = raw_data['l1i_load_miss'].fillna(0) + raw_data['l1i_rfo_miss'].fillna(0)

# Calculate L1I hit rate
raw_data['l1i_hit_rate'] = pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float)
for config in CONFIGS:
    for workload in WORKLOADS:
        if (pd.notna(raw_data['l1i_total_access'].loc[workload, config]) and 
            raw_data['l1i_total_access'].loc[workload, config] > 0):
            hits = raw_data['l1i_total_hit'].loc[workload, config]
            raw_data['l1i_hit_rate'].loc[workload, config] = hits / raw_data['l1i_total_access'].loc[workload, config]

# Initialize DataFrames for derived metrics
calculated_metrics_dfs = {
    "IPC": raw_data["ipc"].copy(),  # Start with a copy
    "Speedup": pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float),
    "L1I_MPKI": pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float),
    "L1I_Coverage": pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float),
    "L1I_Accuracy": pd.DataFrame(index=WORKLOADS, columns=CONFIGS, dtype=float),
    "L1I_Hit_Rate": raw_data["l1i_hit_rate"].copy(),
}

# Get Baseline Data (handle potential NaNs)
baseline_ipc = raw_data["ipc"][BASELINE_CONFIG]
baseline_demand_misses = raw_data["l1i_demand_miss"][BASELINE_CONFIG]

# Calculate per-workload derived metrics
for config in CONFIGS:
    # MPKI
    valid_instr = raw_data["instructions"][config].replace(0, np.nan).dropna()
    valid_demand_miss = raw_data["l1i_demand_miss"][config].loc[valid_instr.index]
    calculated_metrics_dfs["L1I_MPKI"].loc[valid_instr.index, config] = (valid_demand_miss / valid_instr) * 1000
    
    # Speedup
    valid_baseline_ipc = baseline_ipc.replace(0, np.nan).dropna()
    valid_current_ipc = raw_data["ipc"][config].loc[valid_baseline_ipc.index]
    calculated_metrics_dfs["Speedup"].loc[valid_baseline_ipc.index, config] = valid_current_ipc / valid_baseline_ipc
    
    if config == BASELINE_CONFIG:
        calculated_metrics_dfs["L1I_Coverage"][config] = 0.0  # Base configuration has no coverage by definition
        calculated_metrics_dfs["L1I_Accuracy"][config] = np.nan  # Base configuration has no prefetches
    else:
        # Coverage
        denom = baseline_demand_misses.replace(0, np.nan).dropna()  # Avoid division by zero
        valid_current_misses = raw_data["l1i_demand_miss"][config].loc[denom.index]
        coverage = (denom - valid_current_misses) / denom
        calculated_metrics_dfs["L1I_Coverage"].loc[denom.index, config] = coverage
        
        # Accuracy
        issued = raw_data["l1i_prefetch_issued"][config].replace(0, np.nan).dropna()  # Avoid division by zero
        valid_useful = raw_data["l1i_prefetch_useful"][config].loc[issued.index]
        accuracy = valid_useful / issued
        calculated_metrics_dfs["L1I_Accuracy"].loc[issued.index, config] = accuracy

# --- Calculate Aggregate Statistics ---
aggregate_stats = pd.DataFrame(index=["Geomean Speedup", "Mean L1I MPKI", "Mean L1I Coverage", 
                                     "Mean L1I Accuracy", "Mean L1I Hit Rate"])

for config in CONFIGS:
    # Geometric Mean Speedup
    valid_speedups = calculated_metrics_dfs["Speedup"][config].dropna()
    positive_speedups = valid_speedups[valid_speedups > 0]
    if len(positive_speedups) > 0:
        aggregate_stats.loc["Geomean Speedup", config] = gmean(positive_speedups)
    else:
        aggregate_stats.loc["Geomean Speedup", config] = np.nan
    
    # Arithmetic Means for others
    aggregate_stats.loc["Mean L1I MPKI", config] = calculated_metrics_dfs["L1I_MPKI"][config].mean()
    aggregate_stats.loc["Mean L1I Hit Rate", config] = calculated_metrics_dfs["L1I_Hit_Rate"][config].mean()
    
    # Exclude baseline for Coverage/Accuracy mean calculation
    if config == BASELINE_CONFIG:
        aggregate_stats.loc["Mean L1I Coverage", config] = 0.0
        aggregate_stats.loc["Mean L1I Accuracy", config] = np.nan
    else:
        aggregate_stats.loc["Mean L1I Coverage", config] = calculated_metrics_dfs["L1I_Coverage"][config].mean()
        aggregate_stats.loc["Mean L1I Accuracy", config] = calculated_metrics_dfs["L1I_Accuracy"][config].mean()

# --- Create Comparison Sheets ---
# Improvement metrics versus baseline
improvement_metrics = pd.DataFrame(index=WORKLOADS)
for config in CONFIGS:
    if config != BASELINE_CONFIG:
        # IPC improvement percentage
        improvement_metrics[f"{config}_IPC_improv%"] = ((raw_data["ipc"][config] / 
                                                     raw_data["ipc"][BASELINE_CONFIG] - 1) * 100)
        
        # MPKI reduction percentage
        improvement_metrics[f"{config}_MPKI_reduc%"] = ((raw_data["l1i_demand_miss"][BASELINE_CONFIG] - 
                                                     raw_data["l1i_demand_miss"][config]) / 
                                                    raw_data["l1i_demand_miss"][BASELINE_CONFIG] * 100)

# Prefetch effectiveness - combine issued and useful metrics
prefetch_effectiveness = pd.DataFrame(index=WORKLOADS)
for config in CONFIGS:
    prefetch_effectiveness[f"{config}_issued"] = raw_data["l1i_prefetch_issued"][config]
    prefetch_effectiveness[f"{config}_useful"] = raw_data["l1i_prefetch_useful"][config]
    if config != BASELINE_CONFIG:
        prefetch_effectiveness[f"{config}_accuracy%"] = (raw_data["l1i_prefetch_useful"][config] / 
                                                    raw_data["l1i_prefetch_issued"][config] * 100)

# --- Visualization Functions ---
def plot_metric(df, metric_name, ylabel, ylim=None, filename=None):
    """Generate a bar plot for a specific metric across all workloads and configurations."""
    plt.figure(figsize=(14, 8))
    df = df.reset_index() 
    # Melt the DataFrame for easier plotting
    melted_df = df.melt(id_vars=df.index.name, var_name='Configuration', value_name=metric_name)
    melted_df = melted_df.rename(columns={df.index.name: 'Workload'})
    
    # Use seaborn for better aesthetics
    sns.set(style='whitegrid', font_scale=1.2)
    ax = sns.barplot(x='Workload', y=metric_name, hue='Configuration', data=melted_df)
    
    # Customize appearance
    plt.title(f'{metric_name} per Workload for Different Configurations', fontsize=16)
    plt.xlabel('Workload', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    if ylim:
        plt.ylim(ylim)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_comparison(ipc_df, mpki_df, filename=None):
    """Create a comparison plot showing IPC and MPKI means per configuration."""
    # Calculate mean values per configuration
    ipc_means = ipc_df.mean(numeric_only=True)
    mpki_means = mpki_df.mean(numeric_only=True)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Plot data
    x = np.arange(len(ipc_means))
    width = 0.35
    
    ax1.bar(x - width/2, ipc_means, width, color='steelblue', label='Avg IPC')
    ax2.bar(x + width/2, mpki_means, width, color='indianred', label='Avg MPKI')
    
    # Customize plot
    ax1.set_xlabel('Configuration', fontsize=14)
    ax1.set_ylabel('Average IPC', color='steelblue', fontsize=14)
    ax2.set_ylabel('Average L1I MPKI', color='indianred', fontsize=14)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(ipc_means.index, rotation=45)
    
    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title('IPC vs L1I MPKI Comparison Across Configurations', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

# --- Save Results to Excel ---
try:
    with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
        # Sheet 1: Aggregate Statistics
        aggregate_stats.round(4).to_excel(writer, sheet_name="Aggregate Stats")
        
        # Sheets for calculated metrics
        for metric_name, df in calculated_metrics_dfs.items():
            df.round(4).to_excel(writer, sheet_name=f"PerWorkload_{metric_name}")
        
        # Comparison sheets
        improvement_metrics.round(2).to_excel(writer, sheet_name="Configuration_Improvements")
        prefetch_effectiveness.round(0).to_excel(writer, sheet_name="Prefetch_Effectiveness")
        
        # Raw data sheets for detailed analysis
        for metric_name, df in raw_data.items():
            # Format metric name for sheet name
            sheet_name = metric_name.replace('_', ' ').title()
            # Only include most important raw metrics to avoid Excel sheet limit
            if metric_name in ["ipc", "instructions", "l1i_load_miss", "l1i_prefetch_issued", 
                              "l1i_prefetch_useful", "l1i_prefetch_useless", "l1i_demand_miss",
                              "l1i_total_access", "l1i_total_hit", "l1i_total_miss"]:
                df.round(4).to_excel(writer, sheet_name=f"Raw_{sheet_name[:28]}")
        
        # Instruction Processing - Instructions and IPC combined
        instr_processing = pd.DataFrame(index=WORKLOADS)
        for config in CONFIGS:
            instr_processing[f"{config}_instructions"] = raw_data["instructions"][config]
            instr_processing[f"{config}_IPC"] = raw_data["ipc"][config]
        instr_processing.to_excel(writer, sheet_name="Instruction_Processing")
        
        # Cache metrics combined view
        cache_metrics = pd.DataFrame(index=WORKLOADS)
        for config in CONFIGS:
            cache_metrics[f"{config}_L1I_total_access"] = raw_data["l1i_total_access"][config]
            cache_metrics[f"{config}_L1I_total_hit"] = raw_data["l1i_total_hit"][config]
            cache_metrics[f"{config}_L1I_total_miss"] = raw_data["l1i_total_miss"][config]
            cache_metrics[f"{config}_L1I_hit_rate"] = raw_data["l1i_hit_rate"][config]
        cache_metrics.round(4).to_excel(writer, sheet_name="Cache_Metrics")

    print(f"\nResults successfully saved to: {OUTPUT_EXCEL_FILE}")
    
except ImportError:
    print("\nError: 'openpyxl' library not found. Cannot write Excel file.")
    print("Please install it using: pip install openpyxl")
except PermissionError:
    print(f"\nError: Permission denied to write to {OUTPUT_EXCEL_FILE}.")
    print("Please ensure the file is not open and you have write permissions.")
except Exception as e:
    print(f"\nAn error occurred while writing the Excel file: {e}")

# --- Generate Plots ---
if GENERATE_PLOTS:
    try:
        print("\nGenerating plots...")
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
            
        # Generate individual metric plots
        plot_metric(calculated_metrics_dfs["IPC"], "IPC", "IPC", 
                   ylim=(0, 2.0), filename="plots/IPC_per_Workload.png")
        plot_metric(calculated_metrics_dfs["Speedup"], "Speedup", "Speedup", 
                   ylim=(0, 1.2), filename="plots/Speedup_per_Workload.png")
        plot_metric(calculated_metrics_dfs["L1I_MPKI"], "L1I MPKI", "L1I MPKI", 
                   filename="plots/L1I_MPKI_per_Workload.png")
        plot_metric(calculated_metrics_dfs["L1I_Coverage"], "L1I Coverage", "L1I Coverage", 
                   ylim=(0, 1.0), filename="plots/L1I_Coverage_per_Workload.png")
        plot_metric(calculated_metrics_dfs["L1I_Accuracy"], "L1I Accuracy", "L1I Accuracy", 
                   ylim=(0, 0.5), filename="plots/L1I_Accuracy_per_Workload.png")
        
        # Generate comparison plot
        plot_comparison(calculated_metrics_dfs["IPC"], calculated_metrics_dfs["L1I_MPKI"], 
                      filename="plots/IPC_vs_MPKI_Comparison.png")
        
        print("All plots generated successfully in the 'plots' directory.")
    except Exception as e:
        print(f"Error generating plots: {e}")

print("\nAnalysis complete!")

