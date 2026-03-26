import pandas as pd

df = pd.read_csv('reciprocal_summary_plots/network_comparison_statistics.csv')
print('=== TOP 10 LARGEST DIFFERENCES (Bio-trained vs Naive) ===')
print('Positive values = Bio-trained > Naive, Negative values = Naive > Bio-trained')
print()
top_diffs = df.nlargest(10, 'mean_difference')[['source_type', 'target_type', 'bio_mean', 'naive_mean', 'mean_difference', 'p_value']]
for _, row in top_diffs.iterrows():
    print(f'{row["source_type"]:8} → {row["target_type"]:8}: Bio={row["bio_mean"]:6.3f}, Naive={row["naive_mean"]:6.3f}, Δ={row["mean_difference"]:6.3f}, p={row["p_value"]:8.3e}')

print()
print('=== STRONGEST POSITIVE CORRELATIONS IN BIO-TRAINED ===')
bio_positive = df[df['bio_mean'] > 0.1].nlargest(10, 'bio_mean')[['source_type', 'target_type', 'bio_mean', 'naive_mean', 'mean_difference']]
for _, row in bio_positive.iterrows():
    print(f'{row["source_type"]:8} → {row["target_type"]:8}: Bio={row["bio_mean"]:6.3f}, Naive={row["naive_mean"]:6.3f}, Δ={row["mean_difference"]:6.3f}')

print()
print('=== KEY L2/3 RESULTS (matching Znamenskiy et al.) ===')
l23_results = df[(df['source_type'] == 'L2/3_Exc') | (df['target_type'] == 'L2/3_Exc')]
l23_key = l23_results[l23_results['target_type'].isin(['PV', 'SST', 'L2/3_Exc'])]
for _, row in l23_key.iterrows():
    print(f'{row["source_type"]:8} → {row["target_type"]:8}: Bio={row["bio_mean"]:6.3f}, Naive={row["naive_mean"]:6.3f}, Δ={row["mean_difference"]:6.3f}') 