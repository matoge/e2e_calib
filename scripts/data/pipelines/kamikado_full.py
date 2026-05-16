"""End-to-end kamikado pipeline:
   adapter → tile_cutter → lmdb_writer → register_dataset → train (3 GPU).

Run on the DGX2 ClearML controller:
    python -m scripts.data.pipelines.kamikado_full \\
        --raw-root /home/hfunaya/cache/kamikado/scenes \\
        --cache    /cache/kamikado_v3_tiled \\
        --gpus 5,6,7 --epochs 50 --local

(Use --local because we want full GPU access on this host. Without
--local the training step would be enqueued onto a ClearML worker.)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from clearml import PipelineDecorator  # noqa: E402


@PipelineDecorator.component(
    return_values=['cache_dir'],
    cache=False,
    packages=['clearml'],
)
def register_existing_cache(cache_dir: str,
                              dataset_name: str = 'kamikado_v3_tiled',
                              dataset_project: str = 'e2e_calib/cache') -> str:
    """Snapshot the existing cache as a fresh ClearML Dataset version."""
    from clearml import Dataset
    parents = None
    try:
        prev = Dataset.get(dataset_name=dataset_name,
                            dataset_project=dataset_project)
        parents = [prev.id]
        print(f'  inheriting from parent {prev.id}')
    except Exception as e:
        print(f'  no parent dataset (first version): {e}')
    ds = Dataset.create(dataset_name=dataset_name,
                         dataset_project=dataset_project,
                         parent_datasets=parents)
    ds.add_files(cache_dir, verbose=False)
    ds.upload(verbose=False, show_progress=False)
    ds.finalize()
    print(f'[register_existing_cache] id={ds.id} cache={cache_dir}')
    return cache_dir


@PipelineDecorator.component(
    return_values=['exp_dir'],
    cache=False,
    packages=['torch', 'numpy', 'accelerate', 'clearml', 'pandas',
              'scipy', 'pillow', 'lmdb'],
)
def train_kamikado(cache_dir: str, gpus: str, epochs: int,
                    name: str, n_layers: int, batch_size: int,
                    workers: int, oversample: int) -> str:
    """Launch accelerate-driven DDP training on the requested GPUs."""
    import os
    import subprocess
    from pathlib import Path
    REPO_ROOT = Path('/workspace')
    n_gpus = len([g for g in gpus.split(',') if g.strip()])
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    cmd = [
        'accelerate', 'launch',
        f'--num_processes={n_gpus}',
        '--mixed_precision=fp16',
        str(REPO_ROOT / 'scripts/training/train_ps_v3_ddp.py'),
        '--name', name,
        '--cache', cache_dir,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--workers', str(workers),
        '--oversample', str(oversample),
        '--n-layers', str(n_layers),
        '--clearml',
        '--why', 'pipeline_kamikado_full',
    ]
    print('[train_kamikado] cmd:', ' '.join(cmd))
    print('[train_kamikado] CUDA_VISIBLE_DEVICES =', env['CUDA_VISIBLE_DEVICES'])
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    exp_dir = str(REPO_ROOT / 'experiments' / name)
    print(f'[train_kamikado] done → {exp_dir}')
    return exp_dir


@PipelineDecorator.pipeline(
    name='kamikado_full',
    project='e2e_calib/data',
    version='0.1.0',
)
def pipeline(cache_dir: str, gpus: str, epochs: int, name: str,
              n_layers: int, batch_size: int, workers: int, oversample: int):
    cache = register_existing_cache(cache_dir)
    exp_dir = train_kamikado(cache, gpus, epochs, name,
                                n_layers, batch_size, workers, oversample)
    print(f'pipeline done: cache={cache} exp_dir={exp_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/cache/kamikado_v3_tiled')
    ap.add_argument('--gpus', default='5,6,7')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--name', default='km_pipeline_smoke')
    ap.add_argument('--n-layers', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=192)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--oversample', type=int, default=4)
    ap.add_argument('--queue', default='dgx2')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    if args.local:
        PipelineDecorator.run_locally()
    else:
        PipelineDecorator.set_default_execution_queue(args.queue)
    pipeline(args.cache, args.gpus, args.epochs, args.name,
             args.n_layers, args.batch_size, args.workers, args.oversample)


if __name__ == '__main__':
    main()
