# AIND Stan Fit and Simulation Capsule

### Table of Contents
- [Overview](#overview)
- [Standard Run Sequence](#standard-run-sequence)
- [Step 1: Session Curation](#step-1-session-curation)
- [Step 2: Stan Data Preparation and Model Fit](#step-2-stan-data-preparation-and-model-fit)
  - [How the Stan Model Works](#how-the-stan-model-works)
    - [Session-Level Parameters](#session-level-parameters)
    - [Hierarchy and Constraints](#hierarchy-and-constraints)
    - [Choice Likelihood](#choice-likelihood)
    - [Q-Value Update Rule](#q-value-update-rule)
    - [Generated Quantities](#generated-quantities)
- [Step 3: Decision-Variable Inference](#step-3-decision-variable-inference)
- [Practical Notes](#practical-notes)

---

## Overview

This capsule runs a full behavioral-modeling pipeline for dynamic foraging sessions:

1. Curate and preprocess valid sessions per animal.
2. Fit a hierarchical Stan Q-learning model to choice/outcome history.
3. Infer trial-by-trial decision variables from posterior parameter samples.

The three scripts are designed to run in order because each stage consumes files saved by the previous stage.

---

## Standard Run Sequence

From the project root, run the scripts in this exact order:

```bash
python code/beh_0_curate_sessions.py
python "code/beh_1_load&fit.py"
python code/beh_2_dv_inference.py
```

> [!CAUTION]
> These scripts must be run **in the exact order** listed above. Step 2 depends on Step 1 outputs, and Step 3 depends on Step 2 outputs.

---

## Step 1: Session Curation

**Script:** [`code/beh_0_curate_sessions.py`](code/beh_0_curate_sessions.py)

What it does:

- Finds sessions for a target animal by scanning `/root/capsule/data` for assets ending with `_raw_data`.
- Loads each session and ensures an NWB file is available (either existing, transferred, or generated from Bonsai JSON when needed).
- Produces session-level QC plots (session timeline and GLM view).
- Stores per-session metadata, including `session_id`, `box`, and `session_cut`.

**Primary output:**

- `/root/capsule/scratch/{animal_id}/{animal_id}_session_data.csv`

This CSV is the curated session manifest used in model fitting.

---

## Step 2: Stan Data Preparation and Model Fit

**Script:** [`code/beh_1_load&fit.py`](code/beh_1_load&fit.py)

What it does:

- Reads curated sessions from `{animal_id}_session_data.csv`.
- Loads trial data and converts each session into binary vectors:
  - `choice`: left/right encoded as 0/1
  - `outcome`: unrewarded/rewarded encoded as 0/1
- Pads sessions to common length $T$ and builds Stan data:
  - $N$: number of sessions
  - $T$: max session length
  - $T_{\text{sesh},n}$: true trial count for session $n$
  - $\text{choice}[N, T]$, $\text{outcome}[N, T]$
- Compiles and samples [`code/stan_qLearning_5params.stan`](code/stan_qLearning_5params.stan) using:
  - 16 chains
  - 5000 post-warmup samples per chain
  - 2500 warmup iterations

**Saved outputs (per animal):**

- `~/capsule/scratch/{animal_id}/stan_qLearning_5params/summary.csv`
- `~/capsule/scratch/{animal_id}/stan_qLearning_5params/paramsFit.csv`
- `~/capsule/scratch/{animal_id}/stan_qLearning_5params/ani_session_data.csv`
- `~/capsule/scratch/{animal_id}/stan_qLearning_5params/samples/`

### How the Stan Model Works

**Model file:** [`code/stan_qLearning_5params.stan`](code/stan_qLearning_5params.stan)

This is a hierarchical reinforcement-learning model with session-level parameters and shared animal-level hyperparameters.

#### Session-Level Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| $\alpha_N$ | `aN` | Learning rate for negative prediction errors |
| $\alpha_P$ | `aP` | Learning rate for positive prediction errors |
| $\alpha_F$ | `aF` | Forgetting rate for the unchosen option |
| $\beta$ | `beta` | Inverse temperature controlling choice stochasticity |
| $b$ | `bias` | Side bias in the logit choice rule |

#### Hierarchy and Constraints

- Raw session parameters are drawn as standard normals, then transformed by animal-level means $\mu_p$ and scales $\sigma$.
- $\alpha_N$, $\alpha_P$, $\alpha_F \in [0, 1]$ via the $\Phi_{\text{approx}}$ transform.
- $\beta \in [0, 10]$ via $\Phi_{\text{approx}} \times 10$.
- $b$ is session-specific with a broad normal prior.

#### Choice Likelihood

On each trial $t$ in session $n$, the probability of choosing right is:

$$P(\text{choice}_{n,t} = 1) = \sigma\!\left(\beta_n \left(Q^R_{n,t} - Q^L_{n,t}\right) + b_n\right)$$

where $\sigma(\cdot)$ is the logistic function. Choices are modeled with `bernoulli_logit`.

#### Q-Value Update Rule

Let $\delta_{n,t} = r_{n,t} - Q^{\text{chosen}}_{n,t}$ be the prediction error (PE) on trial $t$. Q-values are updated as:

$$Q^{\text{chosen}}_{n,t+1} = Q^{\text{chosen}}_{n,t} + \begin{cases} \alpha_P \, \delta_{n,t} & \text{if } \delta_{n,t} \geq 0 \\ \alpha_N \, \delta_{n,t} & \text{if } \delta_{n,t} < 0 \end{cases}$$

$$Q^{\text{unchosen}}_{n,t+1} = \alpha_F \, Q^{\text{unchosen}}_{n,t}$$

where $r_{n,t} \in \{0, 1\}$ is the trial outcome.

#### Generated Quantities

- Animal-level transformed means: $\mu_{\alpha_N}$, $\mu_{\alpha_P}$, $\mu_{\alpha_F}$, $\mu_\beta$
- Per-session log-likelihood $\log p(\text{choices}_n \mid \theta_n)$ and mean log-likelihood

---

## Step 3: Decision-Variable Inference

**Script:** [`code/beh_2_dv_inference.py`](code/beh_2_dv_inference.py)

What it does:

- Loads Stan posterior samples and fitted summaries from Step 2.
- For each session, draws posterior parameter samples $(\alpha_N, \alpha_P, \alpha_F, \beta, b)$.
- Replays observed choices and outcomes through `QLearningModel` to compute trial-level latent variables.
- Averages across sampled trajectories to obtain posterior-mean decision variables.

**Generated decision variables per session:**

| Variable | Description |
|----------|-------------|
| $Q^L$, $Q^R$ | Left and right action values |
| $\delta$ | Trial-level prediction error |
| $p(\text{choice})$ | Model probability assigned to the observed choice |

**Saved outputs (per animal):**

- One CSV per session: `{session_id}_session_model_dv.csv`
- One PDF per session: `{session_id}_session_model_dv.pdf`
- `params_session_sample.csv` — per-session sampled parameter means
- Animal-level posterior sample histogram PDF

---

## Practical Notes

- **Run in sequence only:** Step 2 depends on Step 1 outputs, and Step 3 depends on Step 2 outputs.
- If curation filters or session cuts change, re-run all downstream steps.
