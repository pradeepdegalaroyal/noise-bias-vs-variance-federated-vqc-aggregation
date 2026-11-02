## Shots sweep (sigma=1.2, beta=1)
| Shots | Method | Test MSE | Test accuracy |
|---|---|---|---|
| 8 | Classical MLP, 19 parameters | 0.172 ± 0.023 | 0.945 ± 0.016 |
| 8 | Corrected, data-size weights | 0.440 ± 0.015 | 0.889 ± 0.040 |
| 8 | FedAvg (data-size weights) | 0.465 ± 0.043 | 0.841 ± 0.052 |
| 8 | Fisher-weighted only (alpha=2) | 0.512 ± 0.105 | 0.828 ± 0.080 |
| 8 | Ideal devices, exact gradients | 0.448 ± 0.034 | 0.855 ± 0.030 |
| 8 | Corrected + Fisher-weighted (alpha=2) | 0.499 ± 0.070 | 0.823 ± 0.059 |
| 8 | Reliability-weighted (alpha=1) | 0.480 ± 0.051 | 0.818 ± 0.056 |
| 32 | Corrected, data-size weights | 0.477 ± 0.093 | 0.842 ± 0.081 |
| 32 | FedAvg (data-size weights) | 0.476 ± 0.038 | 0.829 ± 0.039 |
| 32 | Corrected + Fisher-weighted (alpha=2) | 0.474 ± 0.068 | 0.853 ± 0.044 |
| 32 | Reliability-weighted (alpha=1) | 0.476 ± 0.041 | 0.843 ± 0.047 |
| exact | Corrected, data-size weights | 0.450 ± 0.037 | 0.854 ± 0.028 |
| exact | FedAvg (data-size weights) | 0.460 ± 0.026 | 0.834 ± 0.036 |
| exact | Corrected + Fisher-weighted (alpha=2) | 0.478 ± 0.057 | 0.842 ± 0.041 |
| exact | Reliability-weighted (alpha=1) | 0.486 ± 0.051 | 0.826 ± 0.050 |

## Noise heterogeneity sigma
| Noise heterogeneity sigma | Method | Test MSE | Test accuracy |
|---|---|---|---|
| 0 | Corrected, data-size weights | 0.463 ± 0.044 | 0.857 ± 0.057 |
| 0 | FedAvg (data-size weights) | 0.487 ± 0.079 | 0.830 ± 0.065 |
| 0 | Corrected + Fisher-weighted (alpha=2) | 0.458 ± 0.046 | 0.854 ± 0.047 |
| 0 | Reliability-weighted (alpha=1) | 0.472 ± 0.051 | 0.835 ± 0.040 |
| 0.6 | Corrected, data-size weights | 0.442 ± 0.030 | 0.880 ± 0.037 |
| 0.6 | FedAvg (data-size weights) | 0.469 ± 0.046 | 0.832 ± 0.045 |
| 0.6 | Corrected + Fisher-weighted (alpha=2) | 0.464 ± 0.042 | 0.848 ± 0.051 |
| 0.6 | Reliability-weighted (alpha=1) | 0.464 ± 0.028 | 0.837 ± 0.040 |
| 1.2 | Corrected, data-size weights | 0.440 ± 0.015 | 0.889 ± 0.040 |
| 1.2 | FedAvg (data-size weights) | 0.465 ± 0.043 | 0.841 ± 0.052 |
| 1.2 | Corrected + Fisher-weighted (alpha=2) | 0.499 ± 0.070 | 0.823 ± 0.059 |
| 1.2 | Reliability-weighted (alpha=1) | 0.480 ± 0.051 | 0.818 ± 0.056 |

## Injected calibration error
| Injected calibration error | Method | Test MSE | Test accuracy |
|---|---|---|---|
| 0 | Corrected, data-size weights | 0.440 ± 0.015 | 0.889 ± 0.040 |
| 0 | Corrected + Fisher-weighted (alpha=2) | 0.499 ± 0.070 | 0.823 ± 0.059 |
| 0.2 | Corrected, data-size weights | 0.463 ± 0.051 | 0.874 ± 0.063 |
| 0.2 | Corrected + Fisher-weighted (alpha=2) | 0.489 ± 0.059 | 0.833 ± 0.074 |
| 0.5 | Corrected, data-size weights | 0.459 ± 0.067 | 0.877 ± 0.007 |
| 0.5 | Corrected + Fisher-weighted (alpha=2) | 0.498 ± 0.066 | 0.839 ± 0.056 |

## Dirichlet beta
| Dirichlet beta | Method | Test MSE | Test accuracy |
|---|---|---|---|
| 0.3 | Corrected, data-size weights | 0.443 ± 0.019 | 0.861 ± 0.011 |
| 0.3 | FedAvg (data-size weights) | 0.468 ± 0.033 | 0.867 ± 0.071 |
| 0.3 | Corrected + Fisher-weighted (alpha=2) | 0.484 ± 0.048 | 0.878 ± 0.023 |
| 1 | Corrected, data-size weights | 0.440 ± 0.015 | 0.889 ± 0.040 |
| 1 | FedAvg (data-size weights) | 0.465 ± 0.043 | 0.841 ± 0.052 |
| 1 | Corrected + Fisher-weighted (alpha=2) | 0.499 ± 0.070 | 0.823 ± 0.059 |
| 100 | Corrected, data-size weights | 0.453 ± 0.056 | 0.863 ± 0.045 |
| 100 | FedAvg (data-size weights) | 0.450 ± 0.019 | 0.878 ± 0.039 |
| 100 | Corrected + Fisher-weighted (alpha=2) | 0.445 ± 0.019 | 0.891 ± 0.035 |

## Amplitude damping instead of depolarising
| Noise | Method | Test MSE | Test accuracy |
|---|---|---|---|
| amplitude_damping | Corrected, data-size weights | 0.516 ± 0.135 | 0.870 ± 0.090 |
| amplitude_damping | FedAvg (data-size weights) | 0.465 ± 0.034 | 0.881 ± 0.020 |
| amplitude_damping | Corrected + Fisher-weighted (alpha=2) | 0.487 ± 0.108 | 0.862 ± 0.068 |

## Weight exponent
| Corrected | alpha | Test MSE |
|---|---|---|
| False | 0 | 0.465 ± 0.043 |
| False | 1 | 0.480 ± 0.051 |
| False | 2 | 0.512 ± 0.105 |
| False | 4 | 0.528 ± 0.102 |
| True | 0 | 0.440 ± 0.015 |
| True | 1 | 0.461 ± 0.058 |
| True | 2 | 0.499 ± 0.070 |
| True | 4 | 0.506 ± 0.059 |

## Paired differences vs FedAvg (negative = better)
| Experiment | Setting | Method | Δ MSE (mean ± std) | Seeds improved | Paired t p |
|---|---|---|---|---|---|
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corr_a1 | -0.004 ± 0.057 | 3/5 | 0.87 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corr_a4 | +0.041 ± 0.050 | 0/5 | 0.14 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | -0.025 ± 0.029 | 4/5 | 0.12 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | fisher | +0.047 ± 0.065 | 1/5 | 0.18 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | +0.034 ± 0.048 | 2/5 | 0.19 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | raw_a4 | +0.063 ± 0.074 | 2/5 | 0.13 |
| alpha | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | reliability | +0.015 ± 0.043 | 1/5 | 0.47 |
| beta | S=8, sigma=1.2, cal_err=0, beta=0.3, depolarizing | corrected | -0.025 ± 0.030 | 4/5 | 0.14 |
| beta | S=8, sigma=1.2, cal_err=0, beta=0.3, depolarizing | proposed | +0.016 ± 0.061 | 2/5 | 0.60 |
| beta | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | -0.025 ± 0.029 | 4/5 | 0.12 |
| beta | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | +0.034 ± 0.048 | 2/5 | 0.19 |
| beta | S=8, sigma=1.2, cal_err=0, beta=100, depolarizing | corrected | +0.002 ± 0.043 | 4/5 | 0.92 |
| beta | S=8, sigma=1.2, cal_err=0, beta=100, depolarizing | proposed | -0.006 ± 0.016 | 4/5 | 0.48 |
| noisekind | S=8, sigma=1.2, cal_err=0, beta=1, amplitude_damping | corrected | +0.051 ± 0.107 | 3/5 | 0.34 |
| noisekind | S=8, sigma=1.2, cal_err=0, beta=1, amplitude_damping | proposed | +0.022 ± 0.075 | 4/5 | 0.55 |
| shots | S=exact, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | -0.010 ± 0.037 | 4/5 | 0.58 |
| shots | S=exact, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | +0.018 ± 0.044 | 3/5 | 0.40 |
| shots | S=exact, sigma=1.2, cal_err=0, beta=1, depolarizing | reliability | +0.026 ± 0.033 | 1/5 | 0.15 |
| shots | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | -0.025 ± 0.029 | 4/5 | 0.12 |
| shots | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | fisher | +0.047 ± 0.065 | 1/5 | 0.18 |
| shots | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | oracle | -0.017 ± 0.045 | 3/5 | 0.44 |
| shots | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | +0.034 ± 0.048 | 2/5 | 0.19 |
| shots | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | reliability | +0.015 ± 0.043 | 1/5 | 0.47 |
| shots | S=32, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | +0.001 ± 0.067 | 4/5 | 0.97 |
| shots | S=32, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | -0.002 ± 0.036 | 4/5 | 0.91 |
| shots | S=32, sigma=1.2, cal_err=0, beta=1, depolarizing | reliability | +0.000 ± 0.005 | 3/5 | 0.98 |
| sigma | S=8, sigma=0, cal_err=0, beta=1, depolarizing | corrected | -0.025 ± 0.041 | 4/5 | 0.25 |
| sigma | S=8, sigma=0, cal_err=0, beta=1, depolarizing | proposed | -0.030 ± 0.034 | 5/5 | 0.13 |
| sigma | S=8, sigma=0, cal_err=0, beta=1, depolarizing | reliability | -0.015 ± 0.041 | 4/5 | 0.46 |
| sigma | S=8, sigma=0.6, cal_err=0, beta=1, depolarizing | corrected | -0.027 ± 0.028 | 4/5 | 0.10 |
| sigma | S=8, sigma=0.6, cal_err=0, beta=1, depolarizing | proposed | -0.006 ± 0.017 | 3/5 | 0.50 |
| sigma | S=8, sigma=0.6, cal_err=0, beta=1, depolarizing | reliability | -0.005 ± 0.036 | 3/5 | 0.77 |
| sigma | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | corrected | -0.025 ± 0.029 | 4/5 | 0.12 |
| sigma | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | proposed | +0.034 ± 0.048 | 2/5 | 0.19 |
| sigma | S=8, sigma=1.2, cal_err=0, beta=1, depolarizing | reliability | +0.015 ± 0.043 | 1/5 | 0.47 |

Shot cost at S=8 (per run, mean): total 302592, of which calibration 49152 (16.2 %). Mean true attenuation 0.612, mean minimum client attenuation 0.231.