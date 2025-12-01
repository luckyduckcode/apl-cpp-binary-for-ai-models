#!/usr/bin/env python3
"""Basic test for `manifest_to_apl.py` — ensures file is generated and contains expected symbols."""
from pathlib import Path
import subprocess
import json

manifest = 'student_quantized_manifest.json'
out_apl = 'apl/generated_manifest.apl'

subprocess.check_call(['python3', 'scripts/manifest_to_apl.py', '--manifest', manifest, '--out_apl', out_apl])
text = Path(out_apl).read_text(encoding='utf-8')
print('Generated length:', len(text))

# Check basic declarations
assert 'MODEL_FAMILY' in text
assert 'WEIGHT_NAMES' in text
assert 'fc_weight_packed' in text or 'fc.weight' in text

# Check architecture metadata variables
assert 'HIDDEN_SIZE' in text
assert 'NUM_LAYERS' in text
assert 'NUM_HEADS' in text
assert 'KV_GROUPS' in text
assert 'HEAD_DIM' in text
assert 'ATTENTION_VARIANT' in text
assert 'ACTIVATION' in text
assert 'NORM_TYPE' in text
assert 'ROPE_BASE' in text
assert 'ROPE_SCALE' in text

print('manifest_to_apl basic smoke test passed')
