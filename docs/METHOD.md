# Method

## Task, model and clients

* **Data.** The Wisconsin breast-cancer set shipped with scikit-learn (569 samples, binary label in {-1, +1}).
  Per seed: stratified 70/30 train/test split, standardisation, principal component analysis (PCA) to 4
  features fitted on the training part, min-max scaling to angles in [0, pi].
* **Clients.** 4 clients. The training set is split by class with a Dirichlet(beta) share per client
  (label skew; beta = 1 by default, 0.3 = strongly skewed, 100 = independent and identically distributed (IID)).
* **Model.** 4-qubit variational quantum circuit (VQC): RY angle encoding, 2 layers of [RY, RZ on every qubit,
  controlled-Z ring], readout `f(x, theta) = <Z_0>` in [-1, 1], 16 parameters. Loss: squared error to the label.
* **Noisy devices.** Client `i` has a depolarising channel after every gate (two-qubit error probability `p_i`,
  single-qubit `p_i / 10`). `p_i` is log-normal across clients: geometric mean `p_med = 0.08`, standard
  deviation of `ln p` equal to `sigma = 1.2` (exactly, after standardising the draw). Amplitude damping with the
  same probabilities is used as a noise-model mismatch test. The simulator is an exact density-matrix
  simulator with the noise folded into precomputed 256 x 256 superoperators; it is checked against a Qiskit
  statevector for the ideal circuit (`tests/test_simulator.py`).
* **Shots.** Every circuit evaluation returns a binomial estimate of `<Z_0>` from `S` shots (`S = 8` at the
  base point).

## Federated protocol

Each round, every client draws a minibatch of 6 local samples and computes the squared-loss gradient by the
parameter-shift rule (`1 + 2 * 16 = 33` circuits per sample, all on its noisy device with `S` shots). The server
forms `g = sum_i w_i g_i` and takes an Adam step (learning rate 0.1). 40 rounds. The global model is scored on
the held-out test set with the **ideal** circuit, i.e. the question is how good the trained parameters are, not
how good a noisy deployment is.

## Compared aggregation rules

| Key | Client step | Weight `w_i` |
|---|---|---|
| `fedavg` (standard quantum baseline) | raw noisy `f`, `df` | data size `n_i` |
| `reliability` (rule of the class reported by Youm, 2027; our implementation, not theirs) | raw noisy `f`, `df` | `n_i * lam_i` |
| `corrected` (H1) | `f / lam_i` clipped to [-1, 1], `df / lam_i` | `n_i` |
| `fisher` | raw | `n_i * lam_i^2` |
| `proposed` (H1 + H2) | corrected | `n_i * lam_i^2` |
| `oracle` | ideal devices, exact gradients | `n_i` |
| `classical` | 4-3-1 tanh network, 19 parameters, exact gradients, same splits, minibatches, optimiser | `n_i` |

`lam_i` is a **calibrated** attenuation: before training each client runs 24 random probe circuits (same ansatz
and encoding, random inputs and angles) with 512 shots each, and takes the through-origin slope of noisy
against ideal `<Z_0>`. These shots are counted in the budget (see `shots_total` in `results/runs.csv`).
The weight exponent `alpha` generalises the table: `w_i = n_i * lam_i^alpha`. For a corrected client the
estimate has variance `proportional to 1 / (lam_i^2 * S)`, so `alpha = 2` is the inverse-variance
(Fisher-information) choice; for an uncorrected client the gradient is already scaled by `lam_i^2`, so a plain
average implicitly weights by `lam_i^2`.

## Metrics, ablation, sweeps

* **Metrics.** Final ideal-device test mean squared error (MSE) (primary) and test accuracy; mean and sample standard deviation over
  5 seeds (0-4); seed-paired differences against `fedavg` (federated averaging, FedAvg) with a paired t-test (indicative only with 5 seeds).
* **Ablation (waterfall).** FedAvg -> + correction -> + Fisher weights, and the reverse order, at the base point.
* **Exponent sweep.** `alpha in {0, 1, 2, 4}` with and without correction.
* **Noise / robustness sweeps.** Shots `S in {8, 32, exact}`; noise heterogeneity `sigma in {0, 0.6, 1.2}`;
  error injected into the calibrated attenuation (`ln lam` perturbed by 0, 0.2, 0.5 standard deviations);
  amplitude damping instead of depolarising noise; label skew `beta in {0.3, 1, 100}`.
* **Seeds.** The seed fixes the data split, client partition, noise levels, initial parameters, minibatches and
  the shot-noise generator, so all methods see the same problem instance.

## Scope

Four clients, 4 qubits, 40 rounds, one gradient step per round (FedSGD, not multi-step local training),
one dataset. Everything is a simulation; no real-device data are used.
