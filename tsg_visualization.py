"""
Visualization module for TSG experiment results.
Generates comprehensive visualizations of experiment results in English.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from collections import Counter
from matplotlib.gridspec import GridSpec
import os

def load_results(experiment_dir):
    """Load experiment result files"""
    results_file = Path(experiment_dir) / "final_results.json"
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Try to load model info
    model_info_file = Path(experiment_dir) / "model_info.json"
    model_info = {}
    if model_info_file.exists():
        with open(model_info_file, "r") as f:
            model_info = json.load(f)
    
    return results, model_info

def create_visualization(experiment_dir):
    """Create visualization for experiment results"""
    # Load experiment results
    results, model_info = load_results(experiment_dir)
    
    if not results.get("iterations"):
        print("No iteration data to visualize.")
        return
        
    # Set global parameters
    plt.rcParams['font.size'] = 11
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Create a multi-panel figure layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 2, 2])
    
    # Get iteration count
    iterations = len(results["iterations"])
    iter_nums = list(range(1, iterations + 1))
    
    # === Part 1: Experiment info ===
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis('off')
    
    # Title and info
    scenario_name = results['scenario']['name']
    model_name = model_info.get('model', 'Not specified')
    model_family = model_info.get('model_family', 'Not specified')
    history_window = model_info.get('history_window', 'Not specified')
    
    header_text = f"TSG Experiment Results: {scenario_name}\n"
    header_text += f"Model: {model_name} (Model Family: {model_family})\n"
    header_text += f"History Window: {history_window} | Iterations: {iterations}"
    
    ax_header.text(0.5, 0.5, header_text, fontsize=16, weight='bold', ha='center', va='center')
    
    # === Part 2: Detection probability trend ===
    ax_detect = fig.add_subplot(gs[1])
    
    # Extract detection probability data
    detection_probs = [iter_data["evaluation"].get("detection_probability", 0.5) 
                     for iter_data in results["iterations"]]
    
    # Create color map based on attack methods
    method_colors = {
        "OnBodyMetallic": "#1f77b4",       # blue
        "OnBodyNonMetallic": "#2ca02c",    # green
        "InBagExplosive": "#d62728",       # red
        "ConcealedWeapon": "#9467bd",      # purple
        "LiquidExplosive": "#ff7f0e"       # orange
    }
    
    # Get attack method sequence
    adversary_methods = [iter_data["parsed_adversary_strategy"]["method"] 
                       for iter_data in results["iterations"]]
    
    # Plot detection probability bars
    colors = [method_colors.get(method, "gray") for method in adversary_methods]
    bars = ax_detect.bar(iter_nums, detection_probs, color=colors)
    
    # Add value labels and attack target info
    for i, bar in enumerate(bars):
        height = bar.get_height()
        target_tw = results["iterations"][i]["parsed_adversary_strategy"]["time_window"]
        target_cat = results["iterations"][i]["parsed_adversary_strategy"]["category"]
        
        # Format display text
        ax_detect.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{detection_probs[i]:.2f}', ha='center', fontsize=9)
        ax_detect.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{target_tw[:3]}/{target_cat[:3]}', ha='center', fontsize=9, color='white', 
                    rotation=90, weight='bold')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=method)
                      for method, color in method_colors.items() if method in adversary_methods]
    ax_detect.legend(handles=legend_elements, title="Attack Method", loc="upper right", fontsize=10)
    
    # Format chart
    ax_detect.set_ylabel("Detection Probability", fontsize=12)
    ax_detect.set_xlabel("Round", fontsize=12)
    ax_detect.set_title("Detection Probability by Round", fontsize=14)
    ax_detect.set_ylim(0, 1.05)
    ax_detect.set_xticks(iter_nums)
    ax_detect.grid(axis='y', linestyle='--', alpha=0.7)
    
    # === Part 3: Utility chart ===
    ax_utility = fig.add_subplot(gs[2])
    
    # Extract utility data
    screener_utilities = [iter_data["evaluation"]["screener_utility"] 
                        for iter_data in results["iterations"]]
    
    # Calculate average utility
    avg_utility = sum(screener_utilities) / len(screener_utilities) if screener_utilities else 0
    
    # Get random baseline
    random_baseline = results.get("baseline_comparison", {}).get("random_baseline", {}).get("expected_utility", None)
    random_std = results.get("baseline_comparison", {}).get("random_baseline", {}).get("std_deviation", 0)
    
    # Plot utility curve
    ax_utility.plot(iter_nums, screener_utilities, 'o-', linewidth=2, color='blue', label='Screener Utility')
    
    # Plot average utility
    ax_utility.axhline(y=avg_utility, color='darkblue', linestyle='-.', alpha=0.7, 
                     label=f'Average Utility ({avg_utility:.2f})')
    
    # Plot random baseline
    if random_baseline is not None:
        ax_utility.axhline(y=random_baseline, color='red', linestyle='--', 
                        label=f'Random Baseline ({random_baseline:.2f})')
        if random_std:
            ax_utility.fill_between(iter_nums, 
                                [random_baseline - random_std] * len(iter_nums),
                                [random_baseline + random_std] * len(iter_nums),
                                alpha=0.2, color='red')
    
    # Plot zero line
    ax_utility.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add utility value labels
    for i, utility in enumerate(screener_utilities):
        label_color = 'green' if utility >= 0 else 'red'
        y_offset = 0.5 if utility > 0 else -1.0
        ax_utility.text(i+1, utility + y_offset,
                     f'{utility:.2f}', ha='center', fontsize=9, color=label_color)
    
    # Format chart
    ax_utility.set_xlabel("Round", fontsize=12)
    ax_utility.set_ylabel("Utility Value", fontsize=12)
    ax_utility.set_title("Screener Utility vs. Baseline", fontsize=14)
    ax_utility.grid(linestyle='--', alpha=0.7)
    ax_utility.set_xticks(iter_nums)
    ax_utility.legend(loc="upper right")
    
    # Add utility calculation explanation
    explanation_text = (
        "Screener Utility = (1-p) × Failure Value\n"
        "Failure Values: HighRisk: -30 | MediumRisk: -20 | LowRisk: -5"
    )
    ax_utility.text(0.02, 0.02, explanation_text, transform=ax_utility.transAxes,
                 fontsize=10, ha='left', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    output_path = Path(experiment_dir) / "tsg_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def visualize_results(experiment_dir):
    """Generate visualizations for experiment results."""
    try:
        create_visualization(experiment_dir)
    except Exception as e:
        print(f"Error generating visualization: {e}")
        import traceback
        traceback.print_exc()
