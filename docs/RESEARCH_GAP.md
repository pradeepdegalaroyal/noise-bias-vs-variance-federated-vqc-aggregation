# Research gap

## State of the art (from the supplied literature)

The literature supplied for this study contains three papers, one each from the *recent*, *review* and
*adjacent* roles; it contains no *landmark* paper. The related-work table therefore has three rows, not
ten, and every statement about the wider field below is limited to what these three papers report.
Standard components that are not in the supplied set (federated averaging, the parameter-shift rule) are
used as textbook tools and are not cited.

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Youm, *Future Generation Computer Systems*, doi:10.1016/j.future.2026.108886 | 2027 | adjacent | Noise-aware quantum federated learning (QFL) with a shallow variational quantum circuit (VQC) per client; a reliability-weighted aggregation rule that down-weights noisier clients; reports no uniform quantum advantage, a gain of 1.1 percentage points over 15 seeds under shot-noise-limited training, and no gain when expectation values are exact | Does not derive the weights from an estimation-precision argument; does not separate the *bias* of an attenuated noisy signal from its *variance*; does not vary the label skew between clients against the weighting exponent |
| Ibrahim et al., *Modern Physics Letters A*, doi:10.1142/S0217732326502068 | 2026 | recent | Studies global quantum discord (GQD) and quantum Fisher information (QFI) of a multipartite two-level system under Kerr nonlinearity, parametric amplification and intrinsic decoherence; finds that the two quantities have different robustness to decoherence | Single closed-system model; no learning task, no distributed or multi-device setting, no use of estimation precision as an aggregation weight |
| Yuan et al., *Physics Reports*, doi:10.1016/j.physrep.2022.03.002 | 2022 | review | Survey of quantum magnonics: hybrid platforms (cavity photons, superconducting qubits, nitrogen-vacancy centres, phonons) with very different coherence properties | Not about learning; names heterogeneous quantum platforms as the setting but gives no protocol for combining their data when their noise differs |

## Gap

Youm (2027) shows that weighting federated clients by hardware reliability helps only a little, and only
when shots are limited. Two things are left open by that result. (i) A noisy device does not only add
variance: under depolarising-type noise it multiplies the signal (and every parameter-shift derivative) by an
attenuation factor `lam < 1`, which is a *bias* that a plain average of client gradients inherits and that a
weight cannot remove. (ii) Estimation theory (the Fisher-information view used by Ibrahim et al., 2026 for
decoherence-limited estimation) says that once the bias is removed, the variance of a client's estimate grows as
`1 / (lam^2 * shots)`, so the precision-optimal weight is proportional to `lam^2`. Whether this
precision-optimal weight survives non-independent-and-identically-distributed (non-IID) client data, where
a noisy client may be the only holder of a class, has not been tested. The hardware heterogeneity that
motivates the question is the one described in the quantum-magnonics review (Yuan et al., 2022).

**Novelty (one sentence).** This study separates the two effects of device noise on federated gradients,
bias (removed by a calibrated attenuation correction whose shot cost is charged) and variance (handled by
a Fisher-information weight `lam^alpha`), and measures each in an ablation against the data-size-weighted
average and a reliability-weighted rule of the kind reported by Youm (2027), as a function of shot budget, noise
heterogeneity, calibration error and label skew.

## Hypotheses and success metric

Metric: ideal-device test mean squared error (MSE) of the final global model on held-out data (smaller is
better), and test accuracy; mean and standard deviation over 5 seeds; paired differences against federated
averaging (FedAvg) with the same seed.

* **H1 (bias).** With few shots and heterogeneous noise, attenuation-corrected clients with data-size weights
  reach a lower test MSE than FedAvg. Falsified if the paired mean difference is not negative at the base
  operating point.
* **H2 (variance).** Adding the Fisher weight (`alpha = 2`) lowers test MSE further. Falsified if the corrected
  model with `alpha = 2` is not better than with `alpha = 0`. A pilot of 5 seeds on the same data, run before the
  final experiments to choose the operating point, suggested that H2 may fail under label skew; the
  label-skew sweep was added to test this directly.
* **H3 (null control).** With exact expectation values (infinite shots) the aggregation rule makes no
  practical difference, consistent with the exact-expectation observation in Youm (2027).

## Feasibility and scale

All results are classical density-matrix simulations of a 4-qubit, 16-parameter circuit with four clients
(exact superoperator noise, binomial shot sampling), 40 communication rounds, 5 seeds, on one laptop. They say
nothing about real-device performance or quantum advantage; the classical comparison model is included only
to put the absolute test loss in context.
