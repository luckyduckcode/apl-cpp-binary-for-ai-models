MANIFEST_FILE ← '/tmp/pytest-of-runner/pytest-4/test_manifest_to_apl_and_call_0/test_manifest.json'
MODEL_FAMILY ← 'llama'

⍝ === Architecture Metadata ===
HIDDEN_SIZE ← 8
INTERMEDIATE_SIZE ← 32
NUM_LAYERS ← 1
VOCAB_SIZE ← 0
CONTEXT_LENGTH ← 0
NUM_HEADS ← 0
KV_GROUPS ← 0
HEAD_DIM ← 0
ATTENTION_VARIANT ← 'full'
ACTIVATION ← 'relu'
NORM_TYPE ← 'layernorm'
ROPE_BASE ← 10000.0
ROPE_SCALE ← 1.0

⍝ === Weight Names ===
WEIGHT_NAMES ← 'fc.weight'

⍝ === Per-Weight Paths ===
fc_weight_packed ← 'fc.weight_1bit.bin'
fc_weight_scales_txt ← 'fc.weight_scales.npy'
fc_weight_shape ← 8 8
