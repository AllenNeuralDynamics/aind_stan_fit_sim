# %%
import sys
sys.path.append('/root/capsule/aind-beh-ephys-analysis/code/beh_ephys_analysis/utils')
import nest_asyncio
import stan
import numpy as np
import pandas as pd
import os
from os import path
import glob
from RLmodels import QLearningModel
# from RLmodels import qLearningModel_5params_simNoPlot
from RLmodels import RestlessBanditDecoupled
from RLmodels import QLearningModelSim, myPairPlot, getSessionFitParams
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import re
from scipy.stats import norm
from scipy.stats import halfcauchy
from scipy.stats import cauchy
from sklearn.linear_model import LinearRegression
import pickle
import json
nest_asyncio.apply()
from aind_dynamic_foraging_basic_analysis.licks.lick_analysis import plot_lick_analysis, load_data, load_nwb, cal_metrics, plot_met
from beh_functions import parseSessionID, session_dirs, plot_session_glm, plot_session_in_time_all, bonsai_to_nwb, transfer_nwb
from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
import statsmodels.api as sm
import json

# %%
# pip install PyPDF2
# pip install git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis@plot_session_in_time_sue 

# %%
# pip install aind_dynamic_foraging_data_utils

# %%
# # load sessions
# # get list of available sessions
# file_pattern = "*.nwb"
# nwb_folder = "/root/capsule/data/foraging_nwb_bonsai"
# file_list = glob.glob(os.path.join(nwb_folder + "/" + file_pattern))
# file_names = [os.path.basename(file) for file in file_list]
# # file_name = file_names[20]
    
# results = [parseSessionID(file_name) for file_name in file_names]
# aniIDs, dates = zip(*results)

# sessionInfo = pd.DataFrame({'sessionID': file_names,
#                             'aniID': aniIDs,
#                             'date': dates})

# %%
# reference_date = datetime(2024, 5, 15)
# animalID = '717121'
# targetInds = np.where((sessionInfo['date'] >= reference_date) & (sessionInfo['aniID'] == animalID))[0]

# %%
def process_animal_sessions(ani_id):
    save_csv = True
    session_list = [data_asset[:-9] for data_asset in os.listdir('/root/capsule/data') if data_asset.endswith('_raw_data') and ani_id in data_asset]
    # session_df = pd.read_csv('/root/capsule/aind-beh-ephys-analysis/code/data_management/hopkins_session_assets.csv')
    # session_list = [session_id for session_id in session_df['session_id'] if ani_id in session_id]

    # %%
    plt.close('all')
    animal_dir = f'/root/capsule/scratch/{ani_id}'
    os.makedirs(animal_dir, exist_ok=True)
    ani_session_data = {'session_id': [], 'session_cut': [], 'box': []}
    ani_session_file = f'/root/capsule/scratch/{ani_id}/{ani_id}_session_data.csv'
    for session in session_list:
        print(session)
        session_dir = session_dirs(session)
        aniID, datetime, string = parseSessionID(session)
        nwb_file = os.path.join(session_dir['beh_fig_dir'], session + '.nwb')
        if not os.path.exists(nwb_file):
            if aniID.startswith('ZS'):
                print('Hopkins session, transferring from existing nwb.')
                transfer_nwb(session)
            else:
                session_json_dir = os.path.join(session_dir['raw_dir'], 'behavior')
                session_json_files = []
                for dir, _, files in os.walk(session_json_dir):
                    for file in files:
                        if file.endswith('.json') and aniID in file and 'model' not in file:
                            session_json_files.extend([os.path.join(dir, file)])
                print(f'{len(session_json_files)} session json files found.')
                if len(session_json_files) == 1:
                    session_json_file = session_json_files[0]
                    if os.path.exists(os.path.join(session_dir['beh_fig_dir'], session + '.nwb')):
                        print('NWB file already exists.')
                    else:
                        print('Processing NWB:')
                        success, nwb_file = bonsai_to_nwb(session_json_file, os.path.join(session_dir['beh_fig_dir'], session + '.nwb'))


        print('Plotting session.')
        nwb = load_nwb_from_filename(nwb_file)
        trial_df = nwb.trials.to_dataframe()
        ani_session_data['session_id'].append(session)
        ani_session_data['box'].append(nwb.scratch['metadata'][0].box.values[0])
        ani_session_data['session_cut'].append([0, len(trial_df)])
        fig = plot_session_in_time_all(nwb, in_time=False)
        display(fig)
        fig.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_session.png'))
        fig, session_id, _ = plot_session_glm(session, tMax=5)
        fig.savefig(os.path.join(session_dir['beh_fig_dir'], session + '_glm.png'))
        display(fig)
    ani_session_dataframe = pd.DataFrame(ani_session_data)
    if save_csv:
        ani_session_dataframe.to_csv(ani_session_file, index=False)

        print(f"Dictionary has been saved to {ani_session_file}")   

if __name__ == '__main__':
    ani_id = 'ZS061'
    process_animal_sessions(ani_id)


# %%



