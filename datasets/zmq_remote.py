"""Distributed dataloader sink: bind a ZMQ PULL socket and yield samples from remote workers.

Use as IterableDataset alongside a local DataLoader. Remote workers (e.g. on yokohama0)
serve PandaSetCalibDatasetFull via PUSH; master process binds PULL and consumes.

Sample protocol: torch.save'd tuple (img, true_uvd, dist_uvd, vfp) → bytes.
"""
import io
import zmq
import torch
from torch.utils.data import IterableDataset


class ZmqRemotePullDataset(IterableDataset):
    """One PULL socket bound at construction, yields deserialized samples forever.

    Use num_workers=0 in the DataLoader (single-process), since binding multiple
    times to the same port fails. Combine with a separate local DataLoader if
    you also want CPU-local workers; train loop can interleave batches.
    """
    def __init__(self, bind_url: str = "tcp://*:5555", hwm: int = 16):
        self.bind_url = bind_url
        self.hwm = int(hwm)
        self._sock = None

    def _ensure_sock(self):
        if self._sock is None:
            ctx = zmq.Context.instance()
            self._sock = ctx.socket(zmq.PULL)
            self._sock.set_hwm(self.hwm)
            self._sock.bind(self.bind_url)

    def __iter__(self):
        self._ensure_sock()
        while True:
            data = self._sock.recv()
            batch = torch.load(io.BytesIO(data), weights_only=False)
            # Worker may send a single sample (legacy) or a list of samples (batched).
            if isinstance(batch, list):
                for s in batch:
                    yield s
            else:
                yield batch


def serialize_sample(sample) -> bytes:
    buf = io.BytesIO()
    torch.save(sample, buf)
    return buf.getvalue()
