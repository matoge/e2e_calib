#!/bin/bash
# fire a fake request against /api/calibrate using a random PandaSet inst.
set -euo pipefail
HOST=${HOST:-localhost:5005}

# build a synthetic image + pts npy
python3 -c "
import numpy as np, json, io
from PIL import Image
H, W = 256, 256
img = (np.random.rand(H, W, 3) * 255).astype('uint8')
Image.fromarray(img).save('/tmp/test.jpg', quality=85)
pts = np.random.rand(2000, 4).astype('float32')
pts[:, :3] = pts[:, :3] * 50 + np.array([0, 0, 5])  # z>0
pts[:, 3] = pts[:, 3] * 0.5
np.save('/tmp/test.npy', pts)
print('synth ok')
"
K='[[200,0,128],[0,200,128],[0,0,1]]'

curl -sS -X POST \
  -F "image=@/tmp/test.jpg" \
  -F "pts=@/tmp/test.npy" \
  -F "K=$K" \
  http://$HOST/api/calibrate | python3 -m json.tool | head -30
