# %%
import sys
import os

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

import nest_asyncio
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
nest_asyncio.apply()
from beh_functions import parseSessionID, session_dirs, plot_session_glm, plot_session_in_time_all, bonsai_to_nwb, transfer_nwb
from capsule_migration import capsule_directories
from aind_dynamic_foraging_data_utils.nwb_utils import load_nwb_from_filename
capsule_dirs = capsule_directories()

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
    animal_dir = f'/{capsule_dirs["derived_dir"]}/{ani_id}'
    session_list = [file_name for file_name in os.listdir(animal_dir) if file_name.startswith(f'behavior_{ani_id}')]
    # sort by nartsort
    session_list = natsorted(session_list)
    # session_df = pd.read_csv('/root/capsule/aind-beh-ephys-analysis/code/data_management/hopkins_session_assets.csv')
    # session_list = [session_id for session_id in session_df['session_id'] if ani_id in session_id]

    # %%
    plt.close('all')
    animal_dir = f'{capsule_dirs["output_dir"]}/{ani_id}'
    os.makedirs(animal_dir, exist_ok=True)
    ani_session_data = {'session_id': [], 'session_cut': [], 'box': []}
    ani_session_file = f'{animal_dir}/{ani_id}_session_data.csv'
    for session in session_list:
        print(session)
        session_dir = session_dirs(session)
        aniID, datetime, string = parseSessionID(session)
        session_output_dir = os.path.join(animal_dir, session, 'behavior')
        # make dir
        os.makedirs(session_output_dir, exist_ok=True)
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
        # display(fig)
        fig.savefig(os.path.join(session_output_dir, session + '_session.png'))
        fig, session_id, _ = plot_session_glm(session, tMax=5)
        fig.savefig(os.path.join(session_output_dir, session + '_glm.png'))
        # display(fig)
    ani_session_dataframe = pd.DataFrame(ani_session_data)
    if save_csv:
        ani_session_dataframe.to_csv(ani_session_file, index=False)

        print(f"Dictionary has been saved to {ani_session_file}")   

if __name__ == '__main__':
    ani_id = sys.argv[1] if len(sys.argv) > 1 else '754897'
    process_animal_sessions(ani_id)


# %%



