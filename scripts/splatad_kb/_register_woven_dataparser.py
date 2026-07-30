"""Patch nerfstudio.configs.dataparser_configs to add WovenDataParserConfig.

Run this before ns-train; it edits the loaded dict in-place so the tyro
subcommand union will include `woven-data` next time the configs module is
imported. We do the rewrite by writing a tiny extra import + entry into the
file at /workspace/neurad-studio/nerfstudio/configs/dataparser_configs.py.
"""
from __future__ import annotations
import re
from pathlib import Path

CFG = Path('/workspace/neurad-studio/nerfstudio/configs/dataparser_configs.py')
src = CFG.read_text()

import_line = ('from woven_dataparser import WovenDataParserConfig  '
               '# injected by _register_woven_dataparser')
entry_line = '    "woven-data": WovenDataParserConfig(),'

if import_line in src:
    print('[register] already injected, skipping')
else:
    # add import after `from nerfstudio.data.dataparsers.zod_dataparser ...`
    src = src.replace(
        'from nerfstudio.data.dataparsers.zod_dataparser import ZodDataParserConfig',
        'from nerfstudio.data.dataparsers.zod_dataparser import ZodDataParserConfig\n'
        + import_line,
    )
    # add entry inside the `dataparsers = {...}` dict, right after the
    # pandaset-data entry. The pandaset entry ends with a comma already, so
    # we just insert our new entry before the closing brace.
    m = re.search(r'("pandaset-data":\s*PandaSetDataParserConfig\(\),)', src)
    if m is None:
        raise SystemExit('could not find pandaset-data entry')
    src = src.replace(m.group(1), m.group(1) + '\n' + entry_line)
    CFG.write_text(src)
    print(f'[register] injected woven-data into {CFG}')

# sanity: import it
import importlib, sys
sys.path.insert(0, '/host_woven')
mod = importlib.import_module('nerfstudio.configs.dataparser_configs')
assert 'woven-data' in mod.dataparsers, list(mod.dataparsers)
print('[register] woven-data registered:', mod.dataparsers['woven-data'])
