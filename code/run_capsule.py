""" top level run script """
import sys
import os
import importlib.util


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(animal_ids):
    code_dir = os.path.dirname(os.path.abspath(__file__))

    beh0 = _load_module('beh_0_curate_sessions', os.path.join(code_dir, 'beh_0_curate_sessions.py'))
    beh1 = _load_module('beh_1_load_fit',        os.path.join(code_dir, 'beh_1_load&fit.py'))
    beh2 = _load_module('beh_2_dv_inference',    os.path.join(code_dir, 'beh_2_dv_inference.py'))

    for animal_id in animal_ids:
        try:
            print(f'\n=== Step 1: Curating sessions for {animal_id} ===')
            beh0.process_animal_sessions(animal_id)

            print(f'\n=== Step 2: Fitting Stan model for {animal_id} ===')
            beh1.fit_animal(animal_id)

            print(f'\n=== Step 3: Decision-variable inference for {animal_id} ===')
            beh2.run_animal(animal_id)

            print(f'\n=== Finished {animal_id} ===')
        except Exception as e:
            print(f'\n=== Error processing {animal_id}: {e} — skipping to next animal ===')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ids = sys.argv[1:]
    else:
        ids = ['754897']