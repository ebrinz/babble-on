# Coherence spike report

Samples excluded for a missing primary metric: 0

| comparison | metric | n1 | n2 | mean1 | mean2 | d | p |
|---|---|---|---|---|---|---|---|
| positive_control | gpt2_nll | 100 | 100 | 3.7325 | 3.5551 | +0.883 | 0.0000 |
| null_calibration | gpt2_nll | 50 | 50 | 3.5682 | 3.5420 | +0.145 | 0.4708 |
| primary | gpt2_nll | 100 | 100 | 3.5748 | 3.5781 | -0.014 | 0.9197 |
| anomaly_vs_csprng | gpt2_nll | 100 | 100 | 3.5748 | 3.5551 | +0.093 | 0.5096 |
| positive_control | steps | 100 | 100 | 11.6300 | 11.6300 | +0.000 | 1.0000 |
| null_calibration | steps | 50 | 50 | 11.6200 | 11.6400 | -0.017 | 1.0000 |
| primary | steps | 100 | 100 | 11.7400 | 11.5500 | +0.132 | 0.3754 |
| anomaly_vs_csprng | steps | 100 | 100 | 11.7400 | 11.6300 | +0.082 | 0.5920 |
| positive_control | distinct_ratio | 100 | 100 | 0.7582 | 0.7565 | +0.069 | 0.6308 |
| null_calibration | distinct_ratio | 50 | 50 | 0.7565 | 0.7565 | +0.003 | 0.9877 |
| primary | distinct_ratio | 100 | 100 | 0.7593 | 0.7577 | +0.064 | 0.6469 |
| anomaly_vs_csprng | distinct_ratio | 100 | 100 | 0.7593 | 0.7565 | +0.110 | 0.4356 |

**Verdict:** NO DETECTABLE DIFFERENCE: anomaly vs ordinary gpt2_nll p=0.920
