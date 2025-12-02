MANIFEST_FILE ← '/tmp/pytest-of-runner/pytest-4/test_easy_run_with_bits_20/out/_tmp_pytest-of-runner_pytest-4_test_easy_run_with_bits_20_tiny_hf_manifest.json'
MODEL_FAMILY ← 'llama'

⍝ === Architecture Metadata ===
HIDDEN_SIZE ← 0
INTERMEDIATE_SIZE ← 0
NUM_LAYERS ← 1
VOCAB_SIZE ← 64
CONTEXT_LENGTH ← 4096
NUM_HEADS ← 0
KV_GROUPS ← 0
HEAD_DIM ← 0
ATTENTION_VARIANT ← 'full'
ACTIVATION ← 'swiglu'
NORM_TYPE ← 'rmsnorm'
ROPE_BASE ← 10000.0
ROPE_SCALE ← 1.0

⍝ === Weight Names ===
WEIGHT_NAMES ← 'embedding.weight' 'lm_head.weight' 'transformer.layers.0.input_layernorm.weight' 'transformer.layers.0.mlp.gate_proj.weight' 'transformer.layers.0.mlp.up_proj.weight' 'transformer.layers.0.post_attention_layernorm.weight'

⍝ === Per-Weight Paths ===
embedding_weight_q ← 'embedding.weight_q2.npy'
embedding_weight_q_scales ← 'embedding.weight_q2_scales.npy'
embedding_weight_q_zero_point ← 'embedding.weight_q2_zero_point.npy'
embedding_weight_shape ← 64 32
transformer_layers_0_mlp_gate_proj_weight_q ← 'transformer.layers.0.mlp.gate_proj.weight_q2.npy'
transformer_layers_0_mlp_gate_proj_weight_q_scales ← 'transformer.layers.0.mlp.gate_proj.weight_q2_scales.npy'
transformer_layers_0_mlp_gate_proj_weight_q_zero_point ← 'transformer.layers.0.mlp.gate_proj.weight_q2_zero_point.npy'
transformer_layers_0_mlp_gate_proj_weight_shape ← 32 128
transformer_layers_0_mlp_up_proj_weight_q ← 'transformer.layers.0.mlp.up_proj.weight_q2.npy'
transformer_layers_0_mlp_up_proj_weight_q_scales ← 'transformer.layers.0.mlp.up_proj.weight_q2_scales.npy'
transformer_layers_0_mlp_up_proj_weight_q_zero_point ← 'transformer.layers.0.mlp.up_proj.weight_q2_zero_point.npy'
transformer_layers_0_mlp_up_proj_weight_shape ← 128 32
transformer_layers_0_input_layernorm_weight_fp32 ← 'transformer.layers.0.input_layernorm.weight_fp32.npy'
transformer_layers_0_post_attention_layernorm_weight_fp32 ← 'transformer.layers.0.post_attention_layernorm.weight_fp32.npy'
lm_head_weight_q ← 'lm_head.weight_q2.npy'
lm_head_weight_q_scales ← 'lm_head.weight_q2_scales.npy'
lm_head_weight_q_zero_point ← 'lm_head.weight_q2_zero_point.npy'
lm_head_weight_shape ← 64 32
