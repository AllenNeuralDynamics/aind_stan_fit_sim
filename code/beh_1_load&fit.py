# %%
import nest_asyncio
import stan
import numpy as np
import pandas as pd
import os
import sys
import ast
import pickle
from os import path

# --- external utils bootstrap ---
_utils_candidates = [
    '/src/external/aind-beh-ephys-analysis/code/beh_ephys_analysis/utils',
]
_utils_path = next((p for p in _utils_candidates if p and os.path.isdir(p)), None)
if _utils_path is None:
    raise RuntimeError(
        'aind-beh-ephys-analysis utils not found. Checked:\n'
        + '\n'.join([p for p in _utils_candidates if p])
        + '\nRun postInstall or set AIND_BEH_EPHYS_UTILS to the correct path.'
    )
sys.path.insert(0, _utils_path)
print(f'[bootstrap] using beh_ephys utils from: {_utils_path}')
try:
    import beh_functions  # smoke-test
except ImportError as e:
    raise ImportError(f'beh_functions not importable from {_utils_path}: {e}')
# --- end bootstrap ---

from RLmodels import getSessionFitParams

import arviz as az
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
from beh_functions import session_dirs, makeSessionDF
from capsule_migration import capsule_directories
capsule_dirs = capsule_directories()

nest_asyncio.apply()


def fit_animal(animalID, model_path='/code/stan_qLearning_5params.stan'):
    print(f'\n=== Processing animal {animalID} ===')

    # load curated session data
    animal_dir = f'{capsule_dirs["output_dir"]}/{animalID}'
    ani_session_file = f'{animal_dir}/{animalID}_session_data.csv'

    if not os.path.exists(ani_session_file):
        print(f'File {ani_session_file} does not exist, session is not curated yet')
        return

    ani_session_data = pd.read_csv(ani_session_file)
    print(f'{len(ani_session_data)} sessions are curated for animal {animalID}')

    if len(ani_session_data) == 0:
        print(f'No curated sessions found for animal {animalID}')
        return

    # session data load
    choices = []
    outcomes = []
    sessionLens = []
    sessNum = len(ani_session_data)

    for sessionInd in range(len(ani_session_data)):
        session = ani_session_data.loc[sessionInd, 'session_id']
        print(f'Extracting session data for session {session}')

        session_dir = session_dirs(session)
        nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
        nwb = load_nwb_from_filename(nwb_file)

        curr_cut = ast.literal_eval(ani_session_data.loc[sessionInd, 'session_cut'])
        choice_tbl = makeSessionDF(session, curr_cut)

        choices.append(list(choice_tbl['choice'].values))
        outcomes.append(list(choice_tbl['outcome'].values))
        sessionLens.append(len(choice_tbl))

    if len(sessionLens) == 0:
        print(f'No valid session data found for animal {animalID}')
        return

    # make data for stan
    maxLen = max(sessionLens)

    allChoiceArray = np.array([
        np.pad(choiceSimCurr, (0, maxLen - sessionLenCurr), mode='constant')
        for choiceSimCurr, sessionLenCurr in zip(choices, sessionLens)
    ]).astype(int)

    allOutcomeArray = np.array([
        np.pad(outcomeSimCurr, (0, maxLen - sessionLenCurr), mode='constant')
        for outcomeSimCurr, sessionLenCurr in zip(outcomes, sessionLens)
    ]).astype(int)

    sim_data = {
        "N": sessNum,
        "T": maxLen,
        "Tsesh": sessionLens,
        "choice": allChoiceArray,
        "outcome": allOutcomeArray,
    }

    # Read the Stan model from a file
    with open(model_path, "r") as file:
        model_code = file.read()

    # fitting
    print(f'Building Stan model for animal {animalID}')
    posterior = stan.build(model_code, data=sim_data)

    print(f'Sampling Stan model for animal {animalID}')
    fit = posterior.sample(num_chains=16, num_samples=5000, num_warmup=2500)

    # summarize
    summaryMean = az.summary(fit, stat_focus='mean')
    summaryMedian = az.summary(fit, stat_focus='median')
    summary = pd.merge(summaryMean, summaryMedian, left_index=True, right_index=True)

    # save
    paramNames = ['aN', 'aP', 'aF', 'beta', 'bias']
    saveDir = path.expanduser(f'~/capsule/results/{animalID}/stan_qLearning_5params')
    os.makedirs(saveDir, exist_ok=True)

    paramsFit = getSessionFitParams(summary, paramNames, focus='mean')

    summary.to_csv(f'{saveDir}/summary.csv', index=True)
    paramsFit.to_csv(f'{saveDir}/paramsFit.csv')
    ani_session_data.to_csv(f'{saveDir}/ani_session_data.csv', index=False)

    samples = dict(fit)
    with open(f'{saveDir}/samples', 'wb') as pickle_file:
        pickle.dump(samples, pickle_file)

    print(f'Finished animal {animalID}. Results saved to {saveDir}')


def main():
    if len(sys.argv) > 1:
        animalIDs = sys.argv[1:]
    else:
        animalIDs = ['754897']

    for animalID in animalIDs:
        try:
            fit_animal(animalID)
        except Exception as e:
            print(f'Error processing animal {animalID}: {e}')


if __name__ == '__main__':
    main()