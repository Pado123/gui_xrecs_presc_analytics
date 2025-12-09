# %% Run PGTNet (mirrors structure of test_catboost.py)
import os
import sys
import json
import tqdm
import subprocess
import shlex

curr_dir = '/home/padela/Desktop/LLMs_PM'
os.chdir(curr_dir)

case_studies = ['hospital']
samples = [10, 100, 'max']

PGTNET_REPO = os.environ.get('PGTNET_REPO', '').strip()
PGTNET_INFERENCE_CFG = os.environ.get('PGTNET_INFERENCE_CFG', '').strip()
PGTNET_RESULTS_DIR = os.environ.get('PGTNET_RESULTS_DIR', '').strip()

DATASET_NAME_MAP = {
    'hospital': 'Hospital',
    'bpi12': 'BPI_Challenge_2012',
    'bac': 'BPIC20_DomesticDeclarations',
}

class SkipPGTNet(Exception):
    pass

def _check_pgtnet_setup():
    if not PGTNET_REPO or not os.path.isdir(PGTNET_REPO):
        raise SkipPGTNet('PGTNET_REPO not set or invalid. Set env PGTNET_REPO to your PGTNet clone path.')
    if not PGTNET_INFERENCE_CFG:
        raise SkipPGTNet('PGTNET_INFERENCE_CFG not set. Set env to an inference config path within PGTNet.')
    cfg_path = PGTNET_INFERENCE_CFG if os.path.isabs(PGTNET_INFERENCE_CFG) else os.path.join(PGTNET_REPO, PGTNET_INFERENCE_CFG)
    if not os.path.isfile(cfg_path):
        raise SkipPGTNet(f'Inference config not found: {cfg_path}')
    return cfg_path

def _run(cmd: str, cwd: str):
    proc = subprocess.Popen(shlex.split(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in proc.stdout:
        print(line.rstrip())
        out_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'Command failed ({proc.returncode}): {cmd}')
    return ''.join(out_lines)

def run_pgtnet_inference(exp_name: str, seed: int):
    cfg_path = _check_pgtnet_setup()
    cfg_arg = cfg_path if os.path.isabs(cfg_path) else os.path.relpath(cfg_path, start=PGTNET_REPO)

    cmd = f"python main.py --cfg {shlex.quote(cfg_arg)} run_multiple_splits [0,1,2,3,4] seed {seed}"
    print(f'Running PGTNet inference: {cmd} (cwd={PGTNET_REPO})')
    _run(cmd, cwd=PGTNET_REPO)

    dataset_name = DATASET_NAME_MAP.get(exp_name, exp_name)
    infer_cfg_name = os.path.splitext(os.path.basename(cfg_arg))[0]
    cmd2 = (
        f"python ResultHandler.py --dataset_name {shlex.quote(dataset_name)} "
        f"--seed_number {seed} --inference_config {shlex.quote(infer_cfg_name)}"
    )
    print(f'Running PGTNet ResultHandler: {cmd2} (cwd={PGTNET_REPO})')
    _run(cmd2, cwd=PGTNET_REPO)

    results_dir = PGTNET_RESULTS_DIR or os.path.join(PGTNET_REPO, 'results')
    print(f'PGTNet results expected under: {results_dir}')

if __name__ == '__main__':
    for exp_name in case_studies:
        print('\n' * 2)
        print('Case study is', exp_name)
        try:
            with open(f'hparams/{exp_name}.json') as f:
                hparams = json.load(f)
        except Exception:
            hparams = {}

        for n_samples in samples:
            print('\n' * 1)
            print('Samples (placeholder for parity):', n_samples)
            if n_samples == 'max':
                n_simulations = 1
            else:
                n_simulations = 2

            for seed in range(n_simulations):
                try:
                    rseed = int(1618 + seed)
                    print(f'Running PGTNet with seed {rseed}')
                    run_pgtnet_inference(exp_name, rseed)
                except SkipPGTNet as e:
                    print(f'SKIP PGTNet: {e}')
                    sys.exit(0)
                except Exception as e:
                    print(f'Error running PGTNet: {e}')
                    continue

        print('Finished case study:', exp_name)

