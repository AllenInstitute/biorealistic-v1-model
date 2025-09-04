# Scatter Plot Analysis: Decoding Accuracy vs Firing Rates

## Overview

This analysis examines the relationship between 30-cell decoding accuracy and firing rates across different cell types in bio-trained and naive V1 networks. Two key firing rate measures were analyzed:
1. **Evoked rate**: Firing rate during natural image stimulation
2. **Evoked - spontaneous rate**: Stimulated firing rate minus spontaneous firing rate

## Key Findings

### Correlation Analysis

**Bio-trained Network:**
- Correlation (accuracy vs evoked rate): **r = 0.444**
- Correlation (accuracy vs evoked - spontaneous rate): **r = 0.614**

**Naive Network:**
- Correlation (accuracy vs evoked rate): **r = 0.460**
- Correlation (accuracy vs evoked - spontaneous rate): **r = 0.662**

### Main Insights

1. **Evoked - spontaneous rate is a better predictor** of decoding accuracy than absolute evoked rate in both networks
   - Evoked - spontaneous rate correlations are ~40% stronger than evoked rate correlations
   - This suggests that stimulus-driven activity (above baseline) is more informative for image decoding

2. **Naive network shows slightly stronger correlations** than bio-trained network
   - Naive networks may have more direct stimulus-response relationships
   - Bio-trained networks may have more complex, non-linear dynamics

3. **Firing rate ranges:**
   - **Bio-trained**: 1.43-17.51 Hz (evoked), -1.73-11.05 Hz (evoked - spontaneous)
   - **Naive**: 1.09-19.13 Hz (evoked), -0.91-12.56 Hz (evoked - spontaneous)
   - Naive networks show slightly higher maximum firing rates

4. **Decoding accuracy ranges:**
   - **Bio-trained**: 0.123-0.733 (30-cell accuracy)
   - **Naive**: 0.167-0.758 (30-cell accuracy)
   - Naive networks achieve slightly higher maximum accuracy

## Cell Type Patterns

### High-performing cell types (>0.7 accuracy):
- **L5_PV**: Bio-trained (0.733), Naive (0.758)
- **L4_PV**: Bio-trained (0.672), Naive (0.735)

### Moderate performers (0.4-0.7 accuracy):
- **L4_SST, L5_ET, L5_SST**: Consistent across both networks
- **L2/3_PV, L2/3_SST**: Good performance in both networks

### Lower performers (<0.4 accuracy):
- **L6_Exc, L6_VIP**: Consistently low across networks
- **L5_NP**: Particularly low in bio-trained (0.123)

## Biological Interpretation

1. **PV interneurons** (especially L4_PV, L5_PV) show the highest decoding accuracy
   - Fast-spiking PV cells provide precise temporal information
   - Strong stimulus-driven responses make them excellent for discrimination

2. **Evoked - spontaneous rate correlation** suggests that:
   - Stimulus-specific responses (above baseline) carry the most information
   - Spontaneous activity may add noise rather than signal

3. **Network type differences**:
   - Naive networks show more direct stimulus-response relationships
   - Bio-trained networks may have more realistic cortical dynamics but slightly reduced linear decodability

## Generated Files

**Separate figures for each network:**
- `decoding_accuracy_vs_evoked_rate_bio_trained.png`: Bio-trained network accuracy vs evoked firing rate
- `decoding_accuracy_vs_evoked_rate_naive.png`: Naive network accuracy vs evoked firing rate
- `decoding_accuracy_vs_evoked_minus_spontaneous_rate_bio_trained.png`: Bio-trained network accuracy vs evoked - spontaneous firing rate
- `decoding_accuracy_vs_evoked_minus_spontaneous_rate_naive.png`: Naive network accuracy vs evoked - spontaneous firing rate
- `decoding_vs_firing_rate_summary.csv`: Complete data summary with all metrics

**Legacy combined figures:**
- `decoding_accuracy_vs_evoked_rate.png`: Combined plot (bio-trained: circles, naive: squares)
- `decoding_accuracy_vs_net_evoked_rate.png`: Combined plot (bio-trained: circles, naive: squares)

## Visualization Details

- **Colors**: Official cell type color scheme from `cell_type_naming_scheme.csv`
- **Markers**: Circles (size 120, full opacity) for all data points in separate figures
- **Error bars**: Standard error of the mean across 10 networks (enhanced visibility)
- **Text labels**: Clean, direct cell type labels without background boxes for easy identification
  - Excitatory: "23E" (L2/3_Exc), "4E" (L4_Exc), "5IT", "5ET", "5NP", "6E" (L6_Exc)
  - Inhibitory: "1I" (L1_Inh), "23PV", "23SST", "23VIP", "4PV", "4SST", "4VIP", etc.
- **Correlations**: Individual correlation coefficients displayed in bottom-right corner
- **Legends**: Complete legend with all 19 cell types positioned outside plot area (right side)
- **Figure size**: Compact 8×5 inches for enhanced text visibility
- **Fonts**: Optimized sizes (title: 18pt, axes: 14pt, ticks: 12pt, legend: 9pt, labels: 9pt)
- **Layout**: Clean design with simplified titles showing only network names

## Statistical Notes

- Data represents mean ± SEM across 10 networks per condition
- 30-cell decoding accuracy from multinomial logistic regression
- Firing rates from 30.25s epochs (gray vs natural images)
- All 19 cell types included in analysis 