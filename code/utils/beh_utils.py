import os
from pynwb import NWBFile, TimeSeries, NWBHDF5IO, NWBFile
from hdmf_zarr import NWBZarrIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from aind_ephys_utils import align
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
import re
from scipy.stats import norm
from datetime import datetime
import statsmodels.api as sm

def get_history_from_nwb(nwb):
    """Get choice and reward history from nwb file"""
    df_trial = nwb.trials.to_dataframe()

    autowater_offered = (df_trial.auto_waterL == 1) | (df_trial.auto_waterR == 1)
    choice_history = df_trial.animal_response.map({0: 0, 1: 1, 2: np.nan}).values
    reward_history = df_trial.rewarded_historyL | df_trial.rewarded_historyR
    p_reward = [
        df_trial.reward_probabilityL.values,
        df_trial.reward_probabilityR.values,
    ]
    random_number = [
        df_trial.reward_random_number_left.values,
        df_trial.reward_random_number_right.values,
    ]

    baiting = False if "without baiting" in nwb.protocol.lower() else True

    return (
        baiting,
        choice_history,
        reward_history,
        p_reward,
        autowater_offered,
        random_number,
    )


def loadnwb(sessionID):
    nwb_folder = "/root/capsule/data/foraging_nwb_bonsai"
    nwb_file = f"{nwb_folder}/{sessionID}"

    io = NWBHDF5IO(nwb_file, mode='r')
    nwb = io.read()
    return nwb

def parseSessionID(file_name):
    if len(re.split('[_.]', file_name)[0]) == 6:
        aniID = re.split('[_.]', file_name)[0]
        date = re.split('[_.]', file_name)[1]
        dateObj = datetime.strptime(date, "%Y-%m-%d")
    else:
        aniID = None
        dateObj = None
    
    return aniID, dateObj

def makeSessionDF(nwb, cut = [0, 0]):
    tblTrials = nwb.trials.to_dataframe()
    if cut[1] != 0:
        tblTrials = tblTrials.iloc[cut[0]:-cut[1]].copy()
    else:
        tblTrials = tblTrials.iloc[cut[0]:].copy()
    trialStarts = tblTrials.loc[tblTrials['animal_response']!=2, 'goCue_start_time'].values
    responseTimes = tblTrials[tblTrials['animal_response']!=2]
    responseTimes = responseTimes['reward_outcome_time'].values

    # responseInds
    responseInds = tblTrials['animal_response']!=2
    leftRewards = tblTrials.loc[responseInds, 'rewarded_historyL']

    # oucome 
    leftRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyL']
    rightRewards = tblTrials.loc[tblTrials['animal_response']!=2, 'rewarded_historyR']
    outcomes = leftRewards | rightRewards
    outcomePrev = np.concatenate((np.full((1), np.nan), outcomes[:-1]))

    # choices
    choices = tblTrials.loc[tblTrials['animal_response']!=2, 'animal_response'] == 1
    choicesPrev = np.concatenate((np.full((1), np.nan), choices[:-1]))
    
    # laser
    laserChoice = tblTrials.loc[tblTrials['animal_response']!=2, 'laser_on_trial'] == 1
    laser = tblTrials['laser_on_trial'] == 1
    laserPrev = np.concatenate((np.full((1), np.nan), laserChoice[:-1]))
    trialData = pd.DataFrame({
        'outcomes': outcomes.values.astype(float), 
        'choices': choices.values.astype(float),
        'laser': laserChoice.values.astype(float),
        'outcomePrev': outcomePrev,
        'laserPrev': laserPrev,
        'choicesPrev': choicesPrev,
        })
    return trialData

def plot_session_glm(nwb):
    tbl = makeSessionDF(nwb, cut = [0, 0])
    allChoices = 2 * (tbl['choices'].values - 0.5)
    allRewards = allChoices * tbl['outcomes'].values
    allNoRewards = allChoices * (1 - tbl['outcomes'].values)
    allChoice_R = tbl['choices'].values
    # Example data - replace these with your actual data
    tMax = 10

    # Creating rwdMatx
    rwdMatx = []
    for i in range(1, tMax + 1):
        rwdMatx.append(np.concatenate([np.full(i, np.nan), allRewards[:len(allRewards) - i]]))

    rwdMatx = np.array(rwdMatx)

    # Creating noRwdMatx
    noRwdMatx = []
    for i in range(1, tMax + 1):
        noRwdMatx.append(np.concatenate([np.full(i, np.nan), allNoRewards[:len(allNoRewards) - i]]))

    noRwdMatx = np.array(noRwdMatx)

    # Combining rwdMatx and noRwdMatx
    X = np.vstack([rwdMatx, noRwdMatx]).T

    # Remove rows with NaN values
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    y = allChoice_R[valid_idx]

    # Adding a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fitting the GLM model
    glm_binom = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
    glm_result = glm_binom.fit()

    # R-squared calculation (pseudo R-squared)
    rsq = glm_result.pseudo_rsquared(kind="cs")

    # Coefficients and confidence intervals
    coef_vals = glm_result.params[1:tMax + 1]
    ci_bands = glm_result.conf_int()[1:tMax + 1]
    error_l = np.abs(coef_vals - ci_bands[:, 0])
    error_u = np.abs(coef_vals - ci_bands[:, 1])

    fig = plt.figure(figsize=(8, 6))
    # Plotting reward coefficients
    plt.errorbar(np.arange(1, tMax + 1) + 0.2, coef_vals, yerr=[error_l, error_u], fmt='o', color='c', linewidth=2, label='Reward')
    plt.plot(np.arange(1, tMax + 1) + 0.2, coef_vals, 'c-', linewidth=1)

    # Coefficients and confidence intervals for no reward
    coef_vals_no_rwd = glm_result.params[tMax + 1:]
    ci_bands_no_rwd = glm_result.conf_int()[tMax + 1:]
    error_l_no_rwd = np.abs(coef_vals_no_rwd - ci_bands_no_rwd[:, 0])
    error_u_no_rwd = np.abs(coef_vals_no_rwd - ci_bands_no_rwd[:, 1])

    # Plotting no reward coefficients
    plt.errorbar(np.arange(1, tMax + 1) + 0.2, coef_vals_no_rwd, yerr=[error_l_no_rwd, error_u_no_rwd], fmt='o', color='m', linewidth=2, label='No Reward')
    plt.plot(np.arange(1, tMax + 1) + 0.2, coef_vals_no_rwd, 'm-', linewidth=1)

    # Labels and legend
    plt.xlabel('Outcome n Trials Back')
    plt.ylabel('β Coefficient')
    plt.xlim([0.5, tMax + 0.5])
    plt.axhline(0, color='k', linestyle='--')

    # Adding R-squared and intercept information in the legend
    intercept_info = f'R² = {rsq:.2f} | Int: {glm_result.params[0]:.2f}'
    plt.legend(loc='upper right')

    return fig, nwb.session_id