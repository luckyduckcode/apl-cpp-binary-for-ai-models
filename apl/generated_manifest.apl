MANIFEST_FILE ← 'student_quantized_manifest.json'
MODEL_FAMILY ← 'llama'

⍝ === Architecture Metadata ===
HIDDEN_SIZE ← 64
INTERMEDIATE_SIZE ← 2048
NUM_LAYERS ← 1
VOCAB_SIZE ← 1000
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
WEIGHT_NAMES ← 'embedding.weight' 'fc.bias' 'fc.weight' 'transformer.layers.0.linear1.bias' 'transformer.layers.0.linear1.weight' 'transformer.layers.0.linear2.bias' 'transformer.layers.0.linear2.weight' 'transformer.layers.0.norm1.bias' 'transformer.layers.0.norm1.weight' 'transformer.layers.0.norm2.bias' 'transformer.layers.0.norm2.weight' 'transformer.layers.0.self_attn.in_proj_bias' 'transformer.layers.0.self_attn.in_proj_weight' 'transformer.layers.0.self_attn.out_proj.bias' 'transformer.layers.0.self_attn.out_proj.weight'

⍝ === Per-Weight Paths ===
embedding_weight_packed ← 'embedding.weight_1bit.bin'
embedding_weight_scales_txt ← 'embedding.weight_scales.txt'
embedding_weight_shape ← 1000 64
transformer_layers_0_self_attn_in_proj_weight_packed ← 'transformer.layers.0.self_attn.in_proj_weight_1bit.bin'
transformer_layers_0_self_attn_in_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.in_proj_weight_scales.txt'
transformer_layers_0_self_attn_in_proj_weight_shape ← 192 64
transformer_layers_0_self_attn_out_proj_weight_packed ← 'transformer.layers.0.self_attn.out_proj.weight_1bit.bin'
transformer_layers_0_self_attn_out_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.out_proj.weight_scales.txt'
transformer_layers_0_self_attn_out_proj_weight_shape ← 64 64
transformer_layers_0_linear1_weight_packed ← 'transformer.layers.0.linear1.weight_1bit.bin'
transformer_layers_0_linear1_weight_scales_txt ← 'transformer.layers.0.linear1.weight_scales.txt'
transformer_layers_0_linear1_weight_shape ← 2048 64
transformer_layers_0_linear2_weight_packed ← 'transformer.layers.0.linear2.weight_1bit.bin'
transformer_layers_0_linear2_weight_scales_txt ← 'transformer.layers.0.linear2.weight_scales.txt'
transformer_layers_0_linear2_weight_shape ← 64 2048
fc_weight_packed ← 'fc.weight_1bit.bin'
fc_weight_scales_txt ← 'fc.weight_scales.txt'
fc_weight_shape ← 1000 64
