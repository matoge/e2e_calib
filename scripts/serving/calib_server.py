"""FastAPI calibration server: image + lidar + declared calib → corrections.

Endpoints:
  POST /infer    main inference; CalibRequest in, CalibResponse out
  GET  /healthz  liveness + active model version
  POST /reload   hot-swap to whatever the model_loader resolves NOW
                 (e.g. ClearML registry has a newer tag pointing here)

Auto-versioning:
  The training side already publishes its best_model.pt into ClearML
  with project/name/tag metadata. A small daemon thread polls the
  registry every CALIB_RELOAD_INTERVAL_SEC (default 60s) and swaps in
  any newer model atomically — no request mid-flight is interrupted
  because the swap is a pointer flip.

TensorRT (future / opt-in):
  CALIB_TRT_ENGINE=/path/to/engine.plan loads a prebuilt engine for the
  CalibNetDepth backbone + deform CA. The custom MSDeformAttn op needs
  a TRT plugin (mmcv has one; vendor + register in pipeline.py). PyTorch
  bf16 path stays as fallback. Engine build is offline:
      trtexec --onnx=calib.onnx --bf16 --saveEngine=calib.plan
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException

from .model_loader import LoadedModel, load_ckpt
from .pipeline import CalibInferencePipeline, correction_response
from .schemas import CalibRequest, CalibResponse, HealthResponse


log = logging.getLogger('calib_server')
logging.basicConfig(level=os.environ.get('CALIB_LOG_LEVEL', 'INFO'))


class PipelineHolder:
    """Pipeline ref + atomic swap. Reader requests grab `self.current` once
    per call; concurrent /reload publishes a fully-built new pipeline and
    replaces the pointer."""
    def __init__(self):
        self._lock = threading.Lock()
        self.current: Optional[CalibInferencePipeline] = None

    def reload(self) -> CalibInferencePipeline:
        loaded = load_ckpt()
        # Skip rebuild when the resolved version is identical (cheap nop
        # for polling thread; matters for ClearML mode where get_local_copy
        # is cached).
        if self.current is not None and self.current.version == loaded.version:
            return self.current
        log.info('reload: building pipeline for version=%s', loaded.version)
        pipe = CalibInferencePipeline(loaded)
        with self._lock:
            self.current = pipe
        return pipe


HOLDER = PipelineHolder()


def _auto_reload_loop(interval_sec: float):
    """Background polling. Catches any ckpt swap pushed by a finished
    training task. Bound to the FastAPI lifespan."""
    while True:
        try:
            HOLDER.reload()
        except Exception as e:
            log.warning('auto-reload skipped: %s', e)
        time.sleep(interval_sec)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initial load + reload thread.
    HOLDER.reload()
    interval = float(os.environ.get('CALIB_RELOAD_INTERVAL_SEC', '60'))
    if interval > 0 and os.environ.get('CALIB_MODEL_SOURCE') == 'clearml':
        t = threading.Thread(target=_auto_reload_loop, args=(interval,),
                              daemon=True, name='auto-reload')
        t.start()
        log.info('auto-reload thread started (every %.0fs)', interval)
    yield


app = FastAPI(title='calib_server',
              version=os.environ.get('CALIB_SERVER_VERSION', 'dev'),
              lifespan=lifespan)


@app.get('/healthz', response_model=HealthResponse)
def healthz():
    pipe = HOLDER.current
    return HealthResponse(
        status='ok' if pipe is not None else 'cold',
        model_version=(pipe.version if pipe else ''),
        cuda_available=torch.cuda.is_available(),
    )


@app.post('/reload', response_model=HealthResponse)
def reload_model():
    pipe = HOLDER.reload()
    return HealthResponse(status='reloaded', model_version=pipe.version,
                           cuda_available=torch.cuda.is_available())


@app.post('/infer', response_model=CalibResponse)
def infer(req: CalibRequest):
    pipe = HOLDER.current
    if pipe is None:
        raise HTTPException(status_code=503, detail='model not loaded')
    try:
        with torch.no_grad():
            result = pipe.infer(req)
    except Exception as e:
        log.exception('infer failed')
        raise HTTPException(status_code=400, detail=f'inference error: {e}')
    return correction_response(result, model_version=pipe.version)


def main():
    """Convenience entry: python -m scripts.serving.calib_server"""
    uvicorn.run(
        'scripts.serving.calib_server:app',
        host=os.environ.get('CALIB_HOST', '0.0.0.0'),
        port=int(os.environ.get('CALIB_PORT', '8080')),
        workers=1,  # GPU model is process-bound; scale via container replicas.
        log_level=os.environ.get('CALIB_LOG_LEVEL', 'info').lower(),
    )


if __name__ == '__main__':
    main()
