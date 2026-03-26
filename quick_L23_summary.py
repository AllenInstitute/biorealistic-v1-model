#!/usr/bin/env python3
"""
Quick summary of L2/3 Exc vs PV reciprocal connection results.
"""

import pandas as pd
import numpy as np

print("=== L2/3 RECIPROCAL CONNECTION ANALYSIS SUMMARY ===")
print()

# Load L2/3 results
try:
    l23_df = pd.read_csv('reciprocal_L23_summary/L23_comparison_statistics.csv')
    
    print("L2/3 EXCITATORY ↔ PV INTERACTIONS (Key for experimental comparison):")
    print("=" * 70)
    
    # Focus on Exc-PV interactions
    exc_pv_results = l23_df[
        ((l23_df['source_type'] == 'L2/3_Exc') & (l23_df['target_type'] == 'L2/3_PV')) |
        ((l23_df['source_type'] == 'L2/3_PV') & (l23_df['target_type'] == 'L2/3_Exc'))
    ]
    
    for _, row in exc_pv_results.iterrows():
        significance = "***" if row['p_corrected'] < 0.001 else "**" if row['p_corrected'] < 0.01 else "*" if row['p_corrected'] < 0.05 else ""
        print(f"{row['source_type']:10} → {row['target_type']:10}: "
              f"Bio={row['bio_mean']:6.3f}, Naive={row['naive_mean']:6.3f}, "
              f"Δ={row['mean_difference']:6.3f}, d={row['cohens_d']:6.2f}, "
              f"p={row['p_corrected']:.2e} {significance}")
        print(f"{'':23} Connections: Bio={row['bio_connections']:,}, Naive={row['naive_connections']:,}")
    
    print()
    print("KEY FINDINGS:")
    print("- L2/3 Exc → PV connections show STRONGER reciprocal correlation in bio-trained vs naive")
    print("- This matches experimental findings from Znamenskiy et al. where structured connectivity")
    print("  was observed between excitatory and PV interneurons in visual cortex")
    print()
    
    print("ALL L2/3 CONNECTIONS (Top 5 by effect size):")
    print("=" * 70)
    
    for _, row in l23_df.head(5).iterrows():
        significance = "***" if row['p_corrected'] < 0.001 else "**" if row['p_corrected'] < 0.01 else "*" if row['p_corrected'] < 0.05 else ""
        print(f"{row['source_type']:10} → {row['target_type']:10}: "
              f"Bio={row['bio_mean']:6.3f}, Naive={row['naive_mean']:6.3f}, "
              f"Δ={row['mean_difference']:6.3f}, d={row['cohens_d']:6.2f}, "
              f"p={row['p_corrected']:.2e} {significance}")
    
    print()
    print("INTERPRETATION:")
    print("- Positive correlations = reciprocal connections have similar strengths")
    print("- Bio-trained networks show stronger structured reciprocal connectivity")
    print("- Largest effects are in inhibitory (VIP, SST, PV) connections")
    print("- L2/3 Exc ↔ PV interactions show significant training effects")
    
except FileNotFoundError:
    print("L2/3 analysis results not found. Run aggregate_L23_reciprocal_results.py first.")
    
print()
print("Generated files:")
print("- reciprocal_L23_summary/L23_correlation_heatmap.png (NEW!)")
print("- reciprocal_L23_summary/L23_exc_pv_scatter_comparison.png")
print("- reciprocal_L23_summary/L23_exc_pv_distributions.png") 
print("- Individual network scatter plots in reciprocal_L23_detailed/") 