"""Lazy ckpt loading with optional ClearML model-registry pull.

Two modes selectable via env var `CALIB_MODEL_SOURCE`:
  - `local` (default): read `CALIB_MODEL_CKPT` directly. Best for dev / CI.
  - `clearml`: pull the latest model under
      `CALIB_MODEL_PROJECT` / `CALIB_MODEL_NAME` (+ optional
       `CALIB_MODEL_TAG`, default 'latest') from ClearML's Model Registry
       and cache it to disk. Hot-swap via /reload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LoadedModel:
    path: Path
    version: str
    state_dict: dict
    model_cfg: dict


def _resolve_local() -> tuple[Path, str]:
    p = Path(os.environ['CALIB_MODEL_CKPT']).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'CALIB_MODEL_CKPT not found: {p}')
    # Version label: filename + mtime hash so repeated loads from the same
    # path register as the same version, but a swap-in is detectable.
    mtime = int(p.stat().st_mtime)
    return p, f'{p.parent.name}/{p.name}@{mtime}'


def _resolve_clearml() -> tuple[Path, str]:
    from clearml import Model
    project = os.environ.get('CALIB_MODEL_PROJECT', 'e2e_calib/calib')
    name    = os.environ.get('CALIB_MODEL_NAME',    'best_model')
    tag     = os.environ.get('CALIB_MODEL_TAG',     'latest')
    candidates = Model.query_models(
        project_name=project, model_name=name, tags=[tag] if tag else None,
        max_results=1, only_published=False,
    )
    if not candidates:
        raise RuntimeError(
            f'No ClearML model matches project={project!r} name={name!r} tag={tag!r}'
        )
    m = candidates[0]
    p = Path(m.get_local_copy()).resolve()
    return p, f'{project}/{name}#{m.id}'


def load_ckpt() -> LoadedModel:
    """Resolve, read, and bundle the next active checkpoint."""
    source = os.environ.get('CALIB_MODEL_SOURCE', 'local').lower()
    if source == 'clearml':
        ckpt_path, version = _resolve_clearml()
    elif source == 'local':
        ckpt_path, version = _resolve_local()
    else:
        raise ValueError(f'CALIB_MODEL_SOURCE must be local|clearml, got {source!r}')

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # The .pt files we save are pure state_dicts; arch knobs come from env
    # so swapping ckpts of the same family doesn't need a re-deploy.
    model_cfg = {
        'img_size':       int(os.environ.get('CALIB_MODEL_IMG_SIZE', 128)),
        'in_channels':    int(os.environ.get('CALIB_MODEL_IN_CHANNELS', 3)),
        'n_layers':       int(os.environ.get('CALIB_MODEL_N_LAYERS', 4)),
        'use_convnext':   os.environ.get('CALIB_MODEL_CONVNEXT', '1') != '0',
        'use_frustum':    os.environ.get('CALIB_MODEL_FRUSTUM', '1') != '0',
        'deform_mode':    os.environ.get('CALIB_MODEL_DEFORM_MODE', 'sl'),
        'use_intensity':  os.environ.get('CALIB_MODEL_USE_INTENSITY', '0') != '0',
        'use_frame_pose': os.environ.get('CALIB_MODEL_USE_FRAME_POSE', '0') != '0',
        'frame_pose_dof': int(os.environ.get('CALIB_MODEL_FRAME_POSE_DOF', 6)),
    }
    return LoadedModel(path=ckpt_path, version=version,
                        state_dict=state, model_cfg=model_cfg)
