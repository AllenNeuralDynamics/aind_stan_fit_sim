# AIND Stan Fit Simulation Capsule

This capsule runs a full behavioral-modeling pipeline for dynamic foraging sessions:

1. Curate and preprocess valid sessions per animal.
2. Fit a hierarchical Stan Q-learning model to choice/outcome history.
3. Infer trial-by-trial decision variables from posterior parameter samples.

The three scripts are designed to run in order because each stage consumes files saved by the previous stage.

## Standard Run Sequence

From the project root, run:

1. python code/beh_0_curate_sessions.py
2. python "code/beh_1_load&fit.py"
3. python code/beh_2_dv_inference.py

## Step 1: Session Curation

Script: code/beh_0_curate_sessions.py

What it does:

- Finds sessions for a target animal by scanning /root/capsule/data for assets ending with _raw_data.
- Loads each session and ensures an NWB file is available (either existing, transferred, or generated from Bonsai JSON when needed).
- Produces session-level QC plots (session timeline and GLM view).
- Stores per-session metadata, including session_id, box, and session_cut.

Primary output:

- /root/capsule/scratch/{animal_id}/{animal_id}_session_data.csv

This CSV is the curated session manifest used in model fitting.

## Step 2: Stan Data Preparation and Model Fit

Script: code/beh_1_load&fit.py

What it does:

- Reads curated sessions from {animal_id}_session_data.csv.
- Loads trial data and converts each session into binary vectors:
	- choice: left/right encoded as 0/1
	- outcome: unrewarded/rewarded encoded as 0/1
- Pads sessions to common length T and builds Stan data:
	- N: number of sessions
	- T: max session length
	- Tsesh: true trial count per session
	- choice[N, T], outcome[N, T]
- Compiles and samples code/stan_qLearning_5params.stan using:
	- 16 chains
	- 5000 post-warmup samples per chain
	- 2500 warmup iterations

Saved outputs (per animal):

- ~/capsule/scratch/{animal_id}/stan_qLearning_5params/summary.csv
- ~/capsule/scratch/{animal_id}/stan_qLearning_5params/paramsFit.csv
- ~/capsule/scratch/{animal_id}/stan_qLearning_5params/ani_session_data.csv
- ~/capsule/scratch/{animal_id}/stan_qLearning_5params/samples

## How the Stan Model Works

Model file: code/stan_qLearning_5params.stan

This is a hierarchical reinforcement-learning model with session-level parameters and shared animal-level hyperparameters.

Session-level parameters:

- aN: learning rate for negative prediction errors
- aP: learning rate for positive prediction errors
- aF: forgetting rate for unchosen option
- beta: inverse temperature controlling choice stochasticity
- bias: side bias in logit choice rule

Hierarchy and constraints:

- Raw session parameters are standard normal and transformed by animal-level mu_p and sigma.
- aN, aP, aF are constrained to [0, 1] via Phi_approx transform.
- beta is constrained to [0, 10] via Phi_approx * 10.
- bias is session-specific with a broad normal prior.

Choice likelihood:

- On each trial t in session n, the right-choice probability is:
	P(choice = 1) = logistic(beta_n * (Q_right - Q_left) + bias_n)
- Choices are modeled with bernoulli_logit.

Q-value update rule:

- Chosen option is updated by prediction error (PE = outcome - Q_chosen).
- Positive PE uses aP; negative PE uses aN.
- Unchosen option decays by forgetting rate aF.

Generated quantities include:

- Animal-level transformed means (mu_aN, mu_aP, mu_aF, mu_beta)
- Per-session log_lik and mean log-likelihood

## Step 3: Decision-Variable Inference

Script: code/beh_2_dv_inference.py

What it does:

- Loads Stan posterior samples and fitted summaries from Step 2.
- For each session, draws posterior parameter samples.
- Replays observed choices and outcomes through QLearningModel to compute trial-level latent variables.
- Averages across sampled trajectories to get posterior-mean decision variables.

Generated decision variables per session:

- Q_l, Q_r: left/right action values
- pe: trial prediction error
- pChoice: model probability assigned to the observed choice

Saved outputs (per animal):

- One CSV per session: {session_id}_session_model_dv.csv
- One PDF per session: {session_id}_session_model_dv.pdf
- params_session_sample.csv with per-session sampled parameter means
- Animal-level posterior sample histogram PDF

## Practical Notes

- Run in sequence only: Step 2 depends on Step 1 outputs, and Step 3 depends on Step 2 outputs.
- If curation filters or session cuts change, re-run all downstream steps.
- The ampersand in beh_1_load&fit.py usually requires quotes in shell commands.