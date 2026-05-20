"""Wrapper: set torch sharing_strategy='file_system' before launching train_ps_v3."""
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import runpy, sys
sys.argv = [
    'train_ps_v3.py',
    '--cache', '/home/hfunaya/cache_v4/kamikado_v3_tiled',
    '--epochs', '3', '--val-size', '256',
    '--batch-size', '128', '--lr', '1e-3',
    '--oversample', '12', '--workers', '8',
    '--no-clearml', '--name', 'baseline_fs',
]
runpy.run_path('scripts/training/train_ps_v3.py', run_name='__main__')
