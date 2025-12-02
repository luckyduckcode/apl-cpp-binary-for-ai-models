MANIFEST_FILE ← 'models\tinyllama_manifest.json'
MODEL_FAMILY ← 'llama'

⍝ === Architecture Metadata ===
HIDDEN_SIZE ← 0
INTERMEDIATE_SIZE ← 0
NUM_LAYERS ← 22
VOCAB_SIZE ← 32000
CONTEXT_LENGTH ← 2048
NUM_HEADS ← 0
KV_GROUPS ← 0
HEAD_DIM ← 0
ATTENTION_VARIANT ← 'full'
ACTIVATION ← 'swiglu'
NORM_TYPE ← 'rmsnorm'
ROPE_BASE ← 10000.0
ROPE_SCALE ← 1.0

⍝ === Weight Names ===
WEIGHT_NAMES ← 'embedding.weight' 'lm_head.weight' 'transformer.layers.0.input_layernorm.weight' 'transformer.layers.0.mlp.down_proj.weight' 'transformer.layers.0.mlp.gate_proj.weight' 'transformer.layers.0.mlp.up_proj.weight' 'transformer.layers.0.post_attention_layernorm.weight' 'transformer.layers.0.self_attn.k_proj.weight' 'transformer.layers.0.self_attn.o_proj.weight' 'transformer.layers.0.self_attn.q_proj.weight' 'transformer.layers.0.self_attn.v_proj.weight' 'transformer.layers.1.input_layernorm.weight' 'transformer.layers.1.mlp.down_proj.weight' 'transformer.layers.1.mlp.gate_proj.weight' 'transformer.layers.1.mlp.up_proj.weight' 'transformer.layers.1.post_attention_layernorm.weight' 'transformer.layers.1.self_attn.k_proj.weight' 'transformer.layers.1.self_attn.o_proj.weight' 'transformer.layers.1.self_attn.q_proj.weight' 'transformer.layers.1.self_attn.v_proj.weight' 'transformer.layers.10.input_layernorm.weight' 'transformer.layers.10.mlp.down_proj.weight' 'transformer.layers.10.mlp.gate_proj.weight' 'transformer.layers.10.mlp.up_proj.weight' 'transformer.layers.10.post_attention_layernorm.weight' 'transformer.layers.10.self_attn.k_proj.weight' 'transformer.layers.10.self_attn.o_proj.weight' 'transformer.layers.10.self_attn.q_proj.weight' 'transformer.layers.10.self_attn.v_proj.weight' 'transformer.layers.11.input_layernorm.weight' 'transformer.layers.11.mlp.down_proj.weight' 'transformer.layers.11.mlp.gate_proj.weight' 'transformer.layers.11.mlp.up_proj.weight' 'transformer.layers.11.post_attention_layernorm.weight' 'transformer.layers.11.self_attn.k_proj.weight' 'transformer.layers.11.self_attn.o_proj.weight' 'transformer.layers.11.self_attn.q_proj.weight' 'transformer.layers.11.self_attn.v_proj.weight' 'transformer.layers.12.input_layernorm.weight' 'transformer.layers.12.mlp.down_proj.weight' 'transformer.layers.12.mlp.gate_proj.weight' 'transformer.layers.12.mlp.up_proj.weight' 'transformer.layers.12.post_attention_layernorm.weight' 'transformer.layers.12.self_attn.k_proj.weight' 'transformer.layers.12.self_attn.o_proj.weight' 'transformer.layers.12.self_attn.q_proj.weight' 'transformer.layers.12.self_attn.v_proj.weight' 'transformer.layers.13.input_layernorm.weight' 'transformer.layers.13.mlp.down_proj.weight' 'transformer.layers.13.mlp.gate_proj.weight' 'transformer.layers.13.mlp.up_proj.weight' 'transformer.layers.13.post_attention_layernorm.weight' 'transformer.layers.13.self_attn.k_proj.weight' 'transformer.layers.13.self_attn.o_proj.weight' 'transformer.layers.13.self_attn.q_proj.weight' 'transformer.layers.13.self_attn.v_proj.weight' 'transformer.layers.14.input_layernorm.weight' 'transformer.layers.14.mlp.down_proj.weight' 'transformer.layers.14.mlp.gate_proj.weight' 'transformer.layers.14.mlp.up_proj.weight' 'transformer.layers.14.post_attention_layernorm.weight' 'transformer.layers.14.self_attn.k_proj.weight' 'transformer.layers.14.self_attn.o_proj.weight' 'transformer.layers.14.self_attn.q_proj.weight' 'transformer.layers.14.self_attn.v_proj.weight' 'transformer.layers.15.input_layernorm.weight' 'transformer.layers.15.mlp.down_proj.weight' 'transformer.layers.15.mlp.gate_proj.weight' 'transformer.layers.15.mlp.up_proj.weight' 'transformer.layers.15.post_attention_layernorm.weight' 'transformer.layers.15.self_attn.k_proj.weight' 'transformer.layers.15.self_attn.o_proj.weight' 'transformer.layers.15.self_attn.q_proj.weight' 'transformer.layers.15.self_attn.v_proj.weight' 'transformer.layers.16.input_layernorm.weight' 'transformer.layers.16.mlp.down_proj.weight' 'transformer.layers.16.mlp.gate_proj.weight' 'transformer.layers.16.mlp.up_proj.weight' 'transformer.layers.16.post_attention_layernorm.weight' 'transformer.layers.16.self_attn.k_proj.weight' 'transformer.layers.16.self_attn.o_proj.weight' 'transformer.layers.16.self_attn.q_proj.weight' 'transformer.layers.16.self_attn.v_proj.weight' 'transformer.layers.17.input_layernorm.weight' 'transformer.layers.17.mlp.down_proj.weight' 'transformer.layers.17.mlp.gate_proj.weight' 'transformer.layers.17.mlp.up_proj.weight' 'transformer.layers.17.post_attention_layernorm.weight' 'transformer.layers.17.self_attn.k_proj.weight' 'transformer.layers.17.self_attn.o_proj.weight' 'transformer.layers.17.self_attn.q_proj.weight' 'transformer.layers.17.self_attn.v_proj.weight' 'transformer.layers.18.input_layernorm.weight' 'transformer.layers.18.mlp.down_proj.weight' 'transformer.layers.18.mlp.gate_proj.weight' 'transformer.layers.18.mlp.up_proj.weight' 'transformer.layers.18.post_attention_layernorm.weight' 'transformer.layers.18.self_attn.k_proj.weight' 'transformer.layers.18.self_attn.o_proj.weight' 'transformer.layers.18.self_attn.q_proj.weight' 'transformer.layers.18.self_attn.v_proj.weight' 'transformer.layers.19.input_layernorm.weight' 'transformer.layers.19.mlp.down_proj.weight' 'transformer.layers.19.mlp.gate_proj.weight' 'transformer.layers.19.mlp.up_proj.weight' 'transformer.layers.19.post_attention_layernorm.weight' 'transformer.layers.19.self_attn.k_proj.weight' 'transformer.layers.19.self_attn.o_proj.weight' 'transformer.layers.19.self_attn.q_proj.weight' 'transformer.layers.19.self_attn.v_proj.weight' 'transformer.layers.2.input_layernorm.weight' 'transformer.layers.2.mlp.down_proj.weight' 'transformer.layers.2.mlp.gate_proj.weight' 'transformer.layers.2.mlp.up_proj.weight' 'transformer.layers.2.post_attention_layernorm.weight' 'transformer.layers.2.self_attn.k_proj.weight' 'transformer.layers.2.self_attn.o_proj.weight' 'transformer.layers.2.self_attn.q_proj.weight' 'transformer.layers.2.self_attn.v_proj.weight' 'transformer.layers.20.input_layernorm.weight' 'transformer.layers.20.mlp.down_proj.weight' 'transformer.layers.20.mlp.gate_proj.weight' 'transformer.layers.20.mlp.up_proj.weight' 'transformer.layers.20.post_attention_layernorm.weight' 'transformer.layers.20.self_attn.k_proj.weight' 'transformer.layers.20.self_attn.o_proj.weight' 'transformer.layers.20.self_attn.q_proj.weight' 'transformer.layers.20.self_attn.v_proj.weight' 'transformer.layers.21.input_layernorm.weight' 'transformer.layers.21.mlp.down_proj.weight' 'transformer.layers.21.mlp.gate_proj.weight' 'transformer.layers.21.mlp.up_proj.weight' 'transformer.layers.21.post_attention_layernorm.weight' 'transformer.layers.21.self_attn.k_proj.weight' 'transformer.layers.21.self_attn.o_proj.weight' 'transformer.layers.21.self_attn.q_proj.weight' 'transformer.layers.21.self_attn.v_proj.weight' 'transformer.layers.3.input_layernorm.weight' 'transformer.layers.3.mlp.down_proj.weight' 'transformer.layers.3.mlp.gate_proj.weight' 'transformer.layers.3.mlp.up_proj.weight' 'transformer.layers.3.post_attention_layernorm.weight' 'transformer.layers.3.self_attn.k_proj.weight' 'transformer.layers.3.self_attn.o_proj.weight' 'transformer.layers.3.self_attn.q_proj.weight' 'transformer.layers.3.self_attn.v_proj.weight' 'transformer.layers.4.input_layernorm.weight' 'transformer.layers.4.mlp.down_proj.weight' 'transformer.layers.4.mlp.gate_proj.weight' 'transformer.layers.4.mlp.up_proj.weight' 'transformer.layers.4.post_attention_layernorm.weight' 'transformer.layers.4.self_attn.k_proj.weight' 'transformer.layers.4.self_attn.o_proj.weight' 'transformer.layers.4.self_attn.q_proj.weight' 'transformer.layers.4.self_attn.v_proj.weight' 'transformer.layers.5.input_layernorm.weight' 'transformer.layers.5.mlp.down_proj.weight' 'transformer.layers.5.mlp.gate_proj.weight' 'transformer.layers.5.mlp.up_proj.weight' 'transformer.layers.5.post_attention_layernorm.weight' 'transformer.layers.5.self_attn.k_proj.weight' 'transformer.layers.5.self_attn.o_proj.weight' 'transformer.layers.5.self_attn.q_proj.weight' 'transformer.layers.5.self_attn.v_proj.weight' 'transformer.layers.6.input_layernorm.weight' 'transformer.layers.6.mlp.down_proj.weight' 'transformer.layers.6.mlp.gate_proj.weight' 'transformer.layers.6.mlp.up_proj.weight' 'transformer.layers.6.post_attention_layernorm.weight' 'transformer.layers.6.self_attn.k_proj.weight' 'transformer.layers.6.self_attn.o_proj.weight' 'transformer.layers.6.self_attn.q_proj.weight' 'transformer.layers.6.self_attn.v_proj.weight' 'transformer.layers.7.input_layernorm.weight' 'transformer.layers.7.mlp.down_proj.weight' 'transformer.layers.7.mlp.gate_proj.weight' 'transformer.layers.7.mlp.up_proj.weight' 'transformer.layers.7.post_attention_layernorm.weight' 'transformer.layers.7.self_attn.k_proj.weight' 'transformer.layers.7.self_attn.o_proj.weight' 'transformer.layers.7.self_attn.q_proj.weight' 'transformer.layers.7.self_attn.v_proj.weight' 'transformer.layers.8.input_layernorm.weight' 'transformer.layers.8.mlp.down_proj.weight' 'transformer.layers.8.mlp.gate_proj.weight' 'transformer.layers.8.mlp.up_proj.weight' 'transformer.layers.8.post_attention_layernorm.weight' 'transformer.layers.8.self_attn.k_proj.weight' 'transformer.layers.8.self_attn.o_proj.weight' 'transformer.layers.8.self_attn.q_proj.weight' 'transformer.layers.8.self_attn.v_proj.weight' 'transformer.layers.9.input_layernorm.weight' 'transformer.layers.9.mlp.down_proj.weight' 'transformer.layers.9.mlp.gate_proj.weight' 'transformer.layers.9.mlp.up_proj.weight' 'transformer.layers.9.post_attention_layernorm.weight' 'transformer.layers.9.self_attn.k_proj.weight' 'transformer.layers.9.self_attn.o_proj.weight' 'transformer.layers.9.self_attn.q_proj.weight' 'transformer.layers.9.self_attn.v_proj.weight'

⍝ === Per-Weight Paths ===
embedding_weight_packed ← 'embedding.weight_1bit.bin'
embedding_weight_scales_txt ← 'embedding.weight_scales.txt'
embedding_weight_shape ← 32000 2048
transformer_layers_0_self_attn_q_proj_weight_packed ← 'transformer.layers.0.self_attn.q_proj.weight_1bit.bin'
transformer_layers_0_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.q_proj.weight_scales.txt'
transformer_layers_0_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_0_self_attn_k_proj_weight_packed ← 'transformer.layers.0.self_attn.k_proj.weight_1bit.bin'
transformer_layers_0_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.k_proj.weight_scales.txt'
transformer_layers_0_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_0_self_attn_v_proj_weight_packed ← 'transformer.layers.0.self_attn.v_proj.weight_1bit.bin'
transformer_layers_0_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.v_proj.weight_scales.txt'
transformer_layers_0_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_0_self_attn_o_proj_weight_packed ← 'transformer.layers.0.self_attn.o_proj.weight_1bit.bin'
transformer_layers_0_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.0.self_attn.o_proj.weight_scales.txt'
transformer_layers_0_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_0_mlp_gate_proj_weight_packed ← 'transformer.layers.0.mlp.gate_proj.weight_1bit.bin'
transformer_layers_0_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.0.mlp.gate_proj.weight_scales.txt'
transformer_layers_0_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_0_mlp_up_proj_weight_packed ← 'transformer.layers.0.mlp.up_proj.weight_1bit.bin'
transformer_layers_0_mlp_up_proj_weight_scales_txt ← 'transformer.layers.0.mlp.up_proj.weight_scales.txt'
transformer_layers_0_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_0_mlp_down_proj_weight_packed ← 'transformer.layers.0.mlp.down_proj.weight_1bit.bin'
transformer_layers_0_mlp_down_proj_weight_scales_txt ← 'transformer.layers.0.mlp.down_proj.weight_scales.txt'
transformer_layers_0_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_1_self_attn_q_proj_weight_packed ← 'transformer.layers.1.self_attn.q_proj.weight_1bit.bin'
transformer_layers_1_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.1.self_attn.q_proj.weight_scales.txt'
transformer_layers_1_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_1_self_attn_k_proj_weight_packed ← 'transformer.layers.1.self_attn.k_proj.weight_1bit.bin'
transformer_layers_1_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.1.self_attn.k_proj.weight_scales.txt'
transformer_layers_1_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_1_self_attn_v_proj_weight_packed ← 'transformer.layers.1.self_attn.v_proj.weight_1bit.bin'
transformer_layers_1_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.1.self_attn.v_proj.weight_scales.txt'
transformer_layers_1_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_1_self_attn_o_proj_weight_packed ← 'transformer.layers.1.self_attn.o_proj.weight_1bit.bin'
transformer_layers_1_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.1.self_attn.o_proj.weight_scales.txt'
transformer_layers_1_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_1_mlp_gate_proj_weight_packed ← 'transformer.layers.1.mlp.gate_proj.weight_1bit.bin'
transformer_layers_1_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.1.mlp.gate_proj.weight_scales.txt'
transformer_layers_1_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_1_mlp_up_proj_weight_packed ← 'transformer.layers.1.mlp.up_proj.weight_1bit.bin'
transformer_layers_1_mlp_up_proj_weight_scales_txt ← 'transformer.layers.1.mlp.up_proj.weight_scales.txt'
transformer_layers_1_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_1_mlp_down_proj_weight_packed ← 'transformer.layers.1.mlp.down_proj.weight_1bit.bin'
transformer_layers_1_mlp_down_proj_weight_scales_txt ← 'transformer.layers.1.mlp.down_proj.weight_scales.txt'
transformer_layers_1_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_2_self_attn_q_proj_weight_packed ← 'transformer.layers.2.self_attn.q_proj.weight_1bit.bin'
transformer_layers_2_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.2.self_attn.q_proj.weight_scales.txt'
transformer_layers_2_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_2_self_attn_k_proj_weight_packed ← 'transformer.layers.2.self_attn.k_proj.weight_1bit.bin'
transformer_layers_2_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.2.self_attn.k_proj.weight_scales.txt'
transformer_layers_2_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_2_self_attn_v_proj_weight_packed ← 'transformer.layers.2.self_attn.v_proj.weight_1bit.bin'
transformer_layers_2_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.2.self_attn.v_proj.weight_scales.txt'
transformer_layers_2_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_2_self_attn_o_proj_weight_packed ← 'transformer.layers.2.self_attn.o_proj.weight_1bit.bin'
transformer_layers_2_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.2.self_attn.o_proj.weight_scales.txt'
transformer_layers_2_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_2_mlp_gate_proj_weight_packed ← 'transformer.layers.2.mlp.gate_proj.weight_1bit.bin'
transformer_layers_2_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.2.mlp.gate_proj.weight_scales.txt'
transformer_layers_2_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_2_mlp_up_proj_weight_packed ← 'transformer.layers.2.mlp.up_proj.weight_1bit.bin'
transformer_layers_2_mlp_up_proj_weight_scales_txt ← 'transformer.layers.2.mlp.up_proj.weight_scales.txt'
transformer_layers_2_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_2_mlp_down_proj_weight_packed ← 'transformer.layers.2.mlp.down_proj.weight_1bit.bin'
transformer_layers_2_mlp_down_proj_weight_scales_txt ← 'transformer.layers.2.mlp.down_proj.weight_scales.txt'
transformer_layers_2_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_3_self_attn_q_proj_weight_packed ← 'transformer.layers.3.self_attn.q_proj.weight_1bit.bin'
transformer_layers_3_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.3.self_attn.q_proj.weight_scales.txt'
transformer_layers_3_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_3_self_attn_k_proj_weight_packed ← 'transformer.layers.3.self_attn.k_proj.weight_1bit.bin'
transformer_layers_3_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.3.self_attn.k_proj.weight_scales.txt'
transformer_layers_3_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_3_self_attn_v_proj_weight_packed ← 'transformer.layers.3.self_attn.v_proj.weight_1bit.bin'
transformer_layers_3_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.3.self_attn.v_proj.weight_scales.txt'
transformer_layers_3_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_3_self_attn_o_proj_weight_packed ← 'transformer.layers.3.self_attn.o_proj.weight_1bit.bin'
transformer_layers_3_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.3.self_attn.o_proj.weight_scales.txt'
transformer_layers_3_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_3_mlp_gate_proj_weight_packed ← 'transformer.layers.3.mlp.gate_proj.weight_1bit.bin'
transformer_layers_3_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.3.mlp.gate_proj.weight_scales.txt'
transformer_layers_3_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_3_mlp_up_proj_weight_packed ← 'transformer.layers.3.mlp.up_proj.weight_1bit.bin'
transformer_layers_3_mlp_up_proj_weight_scales_txt ← 'transformer.layers.3.mlp.up_proj.weight_scales.txt'
transformer_layers_3_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_3_mlp_down_proj_weight_packed ← 'transformer.layers.3.mlp.down_proj.weight_1bit.bin'
transformer_layers_3_mlp_down_proj_weight_scales_txt ← 'transformer.layers.3.mlp.down_proj.weight_scales.txt'
transformer_layers_3_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_4_self_attn_q_proj_weight_packed ← 'transformer.layers.4.self_attn.q_proj.weight_1bit.bin'
transformer_layers_4_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.4.self_attn.q_proj.weight_scales.txt'
transformer_layers_4_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_4_self_attn_k_proj_weight_packed ← 'transformer.layers.4.self_attn.k_proj.weight_1bit.bin'
transformer_layers_4_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.4.self_attn.k_proj.weight_scales.txt'
transformer_layers_4_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_4_self_attn_v_proj_weight_packed ← 'transformer.layers.4.self_attn.v_proj.weight_1bit.bin'
transformer_layers_4_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.4.self_attn.v_proj.weight_scales.txt'
transformer_layers_4_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_4_self_attn_o_proj_weight_packed ← 'transformer.layers.4.self_attn.o_proj.weight_1bit.bin'
transformer_layers_4_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.4.self_attn.o_proj.weight_scales.txt'
transformer_layers_4_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_4_mlp_gate_proj_weight_packed ← 'transformer.layers.4.mlp.gate_proj.weight_1bit.bin'
transformer_layers_4_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.4.mlp.gate_proj.weight_scales.txt'
transformer_layers_4_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_4_mlp_up_proj_weight_packed ← 'transformer.layers.4.mlp.up_proj.weight_1bit.bin'
transformer_layers_4_mlp_up_proj_weight_scales_txt ← 'transformer.layers.4.mlp.up_proj.weight_scales.txt'
transformer_layers_4_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_4_mlp_down_proj_weight_packed ← 'transformer.layers.4.mlp.down_proj.weight_1bit.bin'
transformer_layers_4_mlp_down_proj_weight_scales_txt ← 'transformer.layers.4.mlp.down_proj.weight_scales.txt'
transformer_layers_4_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_5_self_attn_q_proj_weight_packed ← 'transformer.layers.5.self_attn.q_proj.weight_1bit.bin'
transformer_layers_5_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.5.self_attn.q_proj.weight_scales.txt'
transformer_layers_5_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_5_self_attn_k_proj_weight_packed ← 'transformer.layers.5.self_attn.k_proj.weight_1bit.bin'
transformer_layers_5_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.5.self_attn.k_proj.weight_scales.txt'
transformer_layers_5_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_5_self_attn_v_proj_weight_packed ← 'transformer.layers.5.self_attn.v_proj.weight_1bit.bin'
transformer_layers_5_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.5.self_attn.v_proj.weight_scales.txt'
transformer_layers_5_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_5_self_attn_o_proj_weight_packed ← 'transformer.layers.5.self_attn.o_proj.weight_1bit.bin'
transformer_layers_5_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.5.self_attn.o_proj.weight_scales.txt'
transformer_layers_5_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_5_mlp_gate_proj_weight_packed ← 'transformer.layers.5.mlp.gate_proj.weight_1bit.bin'
transformer_layers_5_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.5.mlp.gate_proj.weight_scales.txt'
transformer_layers_5_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_5_mlp_up_proj_weight_packed ← 'transformer.layers.5.mlp.up_proj.weight_1bit.bin'
transformer_layers_5_mlp_up_proj_weight_scales_txt ← 'transformer.layers.5.mlp.up_proj.weight_scales.txt'
transformer_layers_5_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_5_mlp_down_proj_weight_packed ← 'transformer.layers.5.mlp.down_proj.weight_1bit.bin'
transformer_layers_5_mlp_down_proj_weight_scales_txt ← 'transformer.layers.5.mlp.down_proj.weight_scales.txt'
transformer_layers_5_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_6_self_attn_q_proj_weight_packed ← 'transformer.layers.6.self_attn.q_proj.weight_1bit.bin'
transformer_layers_6_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.6.self_attn.q_proj.weight_scales.txt'
transformer_layers_6_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_6_self_attn_k_proj_weight_packed ← 'transformer.layers.6.self_attn.k_proj.weight_1bit.bin'
transformer_layers_6_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.6.self_attn.k_proj.weight_scales.txt'
transformer_layers_6_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_6_self_attn_v_proj_weight_packed ← 'transformer.layers.6.self_attn.v_proj.weight_1bit.bin'
transformer_layers_6_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.6.self_attn.v_proj.weight_scales.txt'
transformer_layers_6_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_6_self_attn_o_proj_weight_packed ← 'transformer.layers.6.self_attn.o_proj.weight_1bit.bin'
transformer_layers_6_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.6.self_attn.o_proj.weight_scales.txt'
transformer_layers_6_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_6_mlp_gate_proj_weight_packed ← 'transformer.layers.6.mlp.gate_proj.weight_1bit.bin'
transformer_layers_6_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.6.mlp.gate_proj.weight_scales.txt'
transformer_layers_6_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_6_mlp_up_proj_weight_packed ← 'transformer.layers.6.mlp.up_proj.weight_1bit.bin'
transformer_layers_6_mlp_up_proj_weight_scales_txt ← 'transformer.layers.6.mlp.up_proj.weight_scales.txt'
transformer_layers_6_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_6_mlp_down_proj_weight_packed ← 'transformer.layers.6.mlp.down_proj.weight_1bit.bin'
transformer_layers_6_mlp_down_proj_weight_scales_txt ← 'transformer.layers.6.mlp.down_proj.weight_scales.txt'
transformer_layers_6_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_7_self_attn_q_proj_weight_packed ← 'transformer.layers.7.self_attn.q_proj.weight_1bit.bin'
transformer_layers_7_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.7.self_attn.q_proj.weight_scales.txt'
transformer_layers_7_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_7_self_attn_k_proj_weight_packed ← 'transformer.layers.7.self_attn.k_proj.weight_1bit.bin'
transformer_layers_7_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.7.self_attn.k_proj.weight_scales.txt'
transformer_layers_7_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_7_self_attn_v_proj_weight_packed ← 'transformer.layers.7.self_attn.v_proj.weight_1bit.bin'
transformer_layers_7_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.7.self_attn.v_proj.weight_scales.txt'
transformer_layers_7_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_7_self_attn_o_proj_weight_packed ← 'transformer.layers.7.self_attn.o_proj.weight_1bit.bin'
transformer_layers_7_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.7.self_attn.o_proj.weight_scales.txt'
transformer_layers_7_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_7_mlp_gate_proj_weight_packed ← 'transformer.layers.7.mlp.gate_proj.weight_1bit.bin'
transformer_layers_7_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.7.mlp.gate_proj.weight_scales.txt'
transformer_layers_7_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_7_mlp_up_proj_weight_packed ← 'transformer.layers.7.mlp.up_proj.weight_1bit.bin'
transformer_layers_7_mlp_up_proj_weight_scales_txt ← 'transformer.layers.7.mlp.up_proj.weight_scales.txt'
transformer_layers_7_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_7_mlp_down_proj_weight_packed ← 'transformer.layers.7.mlp.down_proj.weight_1bit.bin'
transformer_layers_7_mlp_down_proj_weight_scales_txt ← 'transformer.layers.7.mlp.down_proj.weight_scales.txt'
transformer_layers_7_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_8_self_attn_q_proj_weight_packed ← 'transformer.layers.8.self_attn.q_proj.weight_1bit.bin'
transformer_layers_8_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.8.self_attn.q_proj.weight_scales.txt'
transformer_layers_8_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_8_self_attn_k_proj_weight_packed ← 'transformer.layers.8.self_attn.k_proj.weight_1bit.bin'
transformer_layers_8_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.8.self_attn.k_proj.weight_scales.txt'
transformer_layers_8_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_8_self_attn_v_proj_weight_packed ← 'transformer.layers.8.self_attn.v_proj.weight_1bit.bin'
transformer_layers_8_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.8.self_attn.v_proj.weight_scales.txt'
transformer_layers_8_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_8_self_attn_o_proj_weight_packed ← 'transformer.layers.8.self_attn.o_proj.weight_1bit.bin'
transformer_layers_8_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.8.self_attn.o_proj.weight_scales.txt'
transformer_layers_8_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_8_mlp_gate_proj_weight_packed ← 'transformer.layers.8.mlp.gate_proj.weight_1bit.bin'
transformer_layers_8_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.8.mlp.gate_proj.weight_scales.txt'
transformer_layers_8_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_8_mlp_up_proj_weight_packed ← 'transformer.layers.8.mlp.up_proj.weight_1bit.bin'
transformer_layers_8_mlp_up_proj_weight_scales_txt ← 'transformer.layers.8.mlp.up_proj.weight_scales.txt'
transformer_layers_8_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_8_mlp_down_proj_weight_packed ← 'transformer.layers.8.mlp.down_proj.weight_1bit.bin'
transformer_layers_8_mlp_down_proj_weight_scales_txt ← 'transformer.layers.8.mlp.down_proj.weight_scales.txt'
transformer_layers_8_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_9_self_attn_q_proj_weight_packed ← 'transformer.layers.9.self_attn.q_proj.weight_1bit.bin'
transformer_layers_9_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.9.self_attn.q_proj.weight_scales.txt'
transformer_layers_9_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_9_self_attn_k_proj_weight_packed ← 'transformer.layers.9.self_attn.k_proj.weight_1bit.bin'
transformer_layers_9_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.9.self_attn.k_proj.weight_scales.txt'
transformer_layers_9_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_9_self_attn_v_proj_weight_packed ← 'transformer.layers.9.self_attn.v_proj.weight_1bit.bin'
transformer_layers_9_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.9.self_attn.v_proj.weight_scales.txt'
transformer_layers_9_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_9_self_attn_o_proj_weight_packed ← 'transformer.layers.9.self_attn.o_proj.weight_1bit.bin'
transformer_layers_9_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.9.self_attn.o_proj.weight_scales.txt'
transformer_layers_9_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_9_mlp_gate_proj_weight_packed ← 'transformer.layers.9.mlp.gate_proj.weight_1bit.bin'
transformer_layers_9_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.9.mlp.gate_proj.weight_scales.txt'
transformer_layers_9_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_9_mlp_up_proj_weight_packed ← 'transformer.layers.9.mlp.up_proj.weight_1bit.bin'
transformer_layers_9_mlp_up_proj_weight_scales_txt ← 'transformer.layers.9.mlp.up_proj.weight_scales.txt'
transformer_layers_9_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_9_mlp_down_proj_weight_packed ← 'transformer.layers.9.mlp.down_proj.weight_1bit.bin'
transformer_layers_9_mlp_down_proj_weight_scales_txt ← 'transformer.layers.9.mlp.down_proj.weight_scales.txt'
transformer_layers_9_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_10_self_attn_q_proj_weight_packed ← 'transformer.layers.10.self_attn.q_proj.weight_1bit.bin'
transformer_layers_10_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.10.self_attn.q_proj.weight_scales.txt'
transformer_layers_10_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_10_self_attn_k_proj_weight_packed ← 'transformer.layers.10.self_attn.k_proj.weight_1bit.bin'
transformer_layers_10_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.10.self_attn.k_proj.weight_scales.txt'
transformer_layers_10_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_10_self_attn_v_proj_weight_packed ← 'transformer.layers.10.self_attn.v_proj.weight_1bit.bin'
transformer_layers_10_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.10.self_attn.v_proj.weight_scales.txt'
transformer_layers_10_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_10_self_attn_o_proj_weight_packed ← 'transformer.layers.10.self_attn.o_proj.weight_1bit.bin'
transformer_layers_10_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.10.self_attn.o_proj.weight_scales.txt'
transformer_layers_10_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_10_mlp_gate_proj_weight_packed ← 'transformer.layers.10.mlp.gate_proj.weight_1bit.bin'
transformer_layers_10_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.10.mlp.gate_proj.weight_scales.txt'
transformer_layers_10_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_10_mlp_up_proj_weight_packed ← 'transformer.layers.10.mlp.up_proj.weight_1bit.bin'
transformer_layers_10_mlp_up_proj_weight_scales_txt ← 'transformer.layers.10.mlp.up_proj.weight_scales.txt'
transformer_layers_10_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_10_mlp_down_proj_weight_packed ← 'transformer.layers.10.mlp.down_proj.weight_1bit.bin'
transformer_layers_10_mlp_down_proj_weight_scales_txt ← 'transformer.layers.10.mlp.down_proj.weight_scales.txt'
transformer_layers_10_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_11_self_attn_q_proj_weight_packed ← 'transformer.layers.11.self_attn.q_proj.weight_1bit.bin'
transformer_layers_11_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.11.self_attn.q_proj.weight_scales.txt'
transformer_layers_11_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_11_self_attn_k_proj_weight_packed ← 'transformer.layers.11.self_attn.k_proj.weight_1bit.bin'
transformer_layers_11_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.11.self_attn.k_proj.weight_scales.txt'
transformer_layers_11_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_11_self_attn_v_proj_weight_packed ← 'transformer.layers.11.self_attn.v_proj.weight_1bit.bin'
transformer_layers_11_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.11.self_attn.v_proj.weight_scales.txt'
transformer_layers_11_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_11_self_attn_o_proj_weight_packed ← 'transformer.layers.11.self_attn.o_proj.weight_1bit.bin'
transformer_layers_11_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.11.self_attn.o_proj.weight_scales.txt'
transformer_layers_11_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_11_mlp_gate_proj_weight_packed ← 'transformer.layers.11.mlp.gate_proj.weight_1bit.bin'
transformer_layers_11_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.11.mlp.gate_proj.weight_scales.txt'
transformer_layers_11_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_11_mlp_up_proj_weight_packed ← 'transformer.layers.11.mlp.up_proj.weight_1bit.bin'
transformer_layers_11_mlp_up_proj_weight_scales_txt ← 'transformer.layers.11.mlp.up_proj.weight_scales.txt'
transformer_layers_11_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_11_mlp_down_proj_weight_packed ← 'transformer.layers.11.mlp.down_proj.weight_1bit.bin'
transformer_layers_11_mlp_down_proj_weight_scales_txt ← 'transformer.layers.11.mlp.down_proj.weight_scales.txt'
transformer_layers_11_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_12_self_attn_q_proj_weight_packed ← 'transformer.layers.12.self_attn.q_proj.weight_1bit.bin'
transformer_layers_12_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.12.self_attn.q_proj.weight_scales.txt'
transformer_layers_12_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_12_self_attn_k_proj_weight_packed ← 'transformer.layers.12.self_attn.k_proj.weight_1bit.bin'
transformer_layers_12_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.12.self_attn.k_proj.weight_scales.txt'
transformer_layers_12_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_12_self_attn_v_proj_weight_packed ← 'transformer.layers.12.self_attn.v_proj.weight_1bit.bin'
transformer_layers_12_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.12.self_attn.v_proj.weight_scales.txt'
transformer_layers_12_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_12_self_attn_o_proj_weight_packed ← 'transformer.layers.12.self_attn.o_proj.weight_1bit.bin'
transformer_layers_12_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.12.self_attn.o_proj.weight_scales.txt'
transformer_layers_12_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_12_mlp_gate_proj_weight_packed ← 'transformer.layers.12.mlp.gate_proj.weight_1bit.bin'
transformer_layers_12_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.12.mlp.gate_proj.weight_scales.txt'
transformer_layers_12_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_12_mlp_up_proj_weight_packed ← 'transformer.layers.12.mlp.up_proj.weight_1bit.bin'
transformer_layers_12_mlp_up_proj_weight_scales_txt ← 'transformer.layers.12.mlp.up_proj.weight_scales.txt'
transformer_layers_12_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_12_mlp_down_proj_weight_packed ← 'transformer.layers.12.mlp.down_proj.weight_1bit.bin'
transformer_layers_12_mlp_down_proj_weight_scales_txt ← 'transformer.layers.12.mlp.down_proj.weight_scales.txt'
transformer_layers_12_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_13_self_attn_q_proj_weight_packed ← 'transformer.layers.13.self_attn.q_proj.weight_1bit.bin'
transformer_layers_13_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.13.self_attn.q_proj.weight_scales.txt'
transformer_layers_13_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_13_self_attn_k_proj_weight_packed ← 'transformer.layers.13.self_attn.k_proj.weight_1bit.bin'
transformer_layers_13_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.13.self_attn.k_proj.weight_scales.txt'
transformer_layers_13_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_13_self_attn_v_proj_weight_packed ← 'transformer.layers.13.self_attn.v_proj.weight_1bit.bin'
transformer_layers_13_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.13.self_attn.v_proj.weight_scales.txt'
transformer_layers_13_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_13_self_attn_o_proj_weight_packed ← 'transformer.layers.13.self_attn.o_proj.weight_1bit.bin'
transformer_layers_13_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.13.self_attn.o_proj.weight_scales.txt'
transformer_layers_13_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_13_mlp_gate_proj_weight_packed ← 'transformer.layers.13.mlp.gate_proj.weight_1bit.bin'
transformer_layers_13_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.13.mlp.gate_proj.weight_scales.txt'
transformer_layers_13_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_13_mlp_up_proj_weight_packed ← 'transformer.layers.13.mlp.up_proj.weight_1bit.bin'
transformer_layers_13_mlp_up_proj_weight_scales_txt ← 'transformer.layers.13.mlp.up_proj.weight_scales.txt'
transformer_layers_13_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_13_mlp_down_proj_weight_packed ← 'transformer.layers.13.mlp.down_proj.weight_1bit.bin'
transformer_layers_13_mlp_down_proj_weight_scales_txt ← 'transformer.layers.13.mlp.down_proj.weight_scales.txt'
transformer_layers_13_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_14_self_attn_q_proj_weight_packed ← 'transformer.layers.14.self_attn.q_proj.weight_1bit.bin'
transformer_layers_14_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.14.self_attn.q_proj.weight_scales.txt'
transformer_layers_14_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_14_self_attn_k_proj_weight_packed ← 'transformer.layers.14.self_attn.k_proj.weight_1bit.bin'
transformer_layers_14_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.14.self_attn.k_proj.weight_scales.txt'
transformer_layers_14_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_14_self_attn_v_proj_weight_packed ← 'transformer.layers.14.self_attn.v_proj.weight_1bit.bin'
transformer_layers_14_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.14.self_attn.v_proj.weight_scales.txt'
transformer_layers_14_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_14_self_attn_o_proj_weight_packed ← 'transformer.layers.14.self_attn.o_proj.weight_1bit.bin'
transformer_layers_14_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.14.self_attn.o_proj.weight_scales.txt'
transformer_layers_14_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_14_mlp_gate_proj_weight_packed ← 'transformer.layers.14.mlp.gate_proj.weight_1bit.bin'
transformer_layers_14_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.14.mlp.gate_proj.weight_scales.txt'
transformer_layers_14_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_14_mlp_up_proj_weight_packed ← 'transformer.layers.14.mlp.up_proj.weight_1bit.bin'
transformer_layers_14_mlp_up_proj_weight_scales_txt ← 'transformer.layers.14.mlp.up_proj.weight_scales.txt'
transformer_layers_14_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_14_mlp_down_proj_weight_packed ← 'transformer.layers.14.mlp.down_proj.weight_1bit.bin'
transformer_layers_14_mlp_down_proj_weight_scales_txt ← 'transformer.layers.14.mlp.down_proj.weight_scales.txt'
transformer_layers_14_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_15_self_attn_q_proj_weight_packed ← 'transformer.layers.15.self_attn.q_proj.weight_1bit.bin'
transformer_layers_15_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.15.self_attn.q_proj.weight_scales.txt'
transformer_layers_15_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_15_self_attn_k_proj_weight_packed ← 'transformer.layers.15.self_attn.k_proj.weight_1bit.bin'
transformer_layers_15_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.15.self_attn.k_proj.weight_scales.txt'
transformer_layers_15_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_15_self_attn_v_proj_weight_packed ← 'transformer.layers.15.self_attn.v_proj.weight_1bit.bin'
transformer_layers_15_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.15.self_attn.v_proj.weight_scales.txt'
transformer_layers_15_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_15_self_attn_o_proj_weight_packed ← 'transformer.layers.15.self_attn.o_proj.weight_1bit.bin'
transformer_layers_15_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.15.self_attn.o_proj.weight_scales.txt'
transformer_layers_15_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_15_mlp_gate_proj_weight_packed ← 'transformer.layers.15.mlp.gate_proj.weight_1bit.bin'
transformer_layers_15_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.15.mlp.gate_proj.weight_scales.txt'
transformer_layers_15_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_15_mlp_up_proj_weight_packed ← 'transformer.layers.15.mlp.up_proj.weight_1bit.bin'
transformer_layers_15_mlp_up_proj_weight_scales_txt ← 'transformer.layers.15.mlp.up_proj.weight_scales.txt'
transformer_layers_15_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_15_mlp_down_proj_weight_packed ← 'transformer.layers.15.mlp.down_proj.weight_1bit.bin'
transformer_layers_15_mlp_down_proj_weight_scales_txt ← 'transformer.layers.15.mlp.down_proj.weight_scales.txt'
transformer_layers_15_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_16_self_attn_q_proj_weight_packed ← 'transformer.layers.16.self_attn.q_proj.weight_1bit.bin'
transformer_layers_16_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.16.self_attn.q_proj.weight_scales.txt'
transformer_layers_16_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_16_self_attn_k_proj_weight_packed ← 'transformer.layers.16.self_attn.k_proj.weight_1bit.bin'
transformer_layers_16_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.16.self_attn.k_proj.weight_scales.txt'
transformer_layers_16_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_16_self_attn_v_proj_weight_packed ← 'transformer.layers.16.self_attn.v_proj.weight_1bit.bin'
transformer_layers_16_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.16.self_attn.v_proj.weight_scales.txt'
transformer_layers_16_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_16_self_attn_o_proj_weight_packed ← 'transformer.layers.16.self_attn.o_proj.weight_1bit.bin'
transformer_layers_16_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.16.self_attn.o_proj.weight_scales.txt'
transformer_layers_16_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_16_mlp_gate_proj_weight_packed ← 'transformer.layers.16.mlp.gate_proj.weight_1bit.bin'
transformer_layers_16_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.16.mlp.gate_proj.weight_scales.txt'
transformer_layers_16_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_16_mlp_up_proj_weight_packed ← 'transformer.layers.16.mlp.up_proj.weight_1bit.bin'
transformer_layers_16_mlp_up_proj_weight_scales_txt ← 'transformer.layers.16.mlp.up_proj.weight_scales.txt'
transformer_layers_16_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_16_mlp_down_proj_weight_packed ← 'transformer.layers.16.mlp.down_proj.weight_1bit.bin'
transformer_layers_16_mlp_down_proj_weight_scales_txt ← 'transformer.layers.16.mlp.down_proj.weight_scales.txt'
transformer_layers_16_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_17_self_attn_q_proj_weight_packed ← 'transformer.layers.17.self_attn.q_proj.weight_1bit.bin'
transformer_layers_17_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.17.self_attn.q_proj.weight_scales.txt'
transformer_layers_17_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_17_self_attn_k_proj_weight_packed ← 'transformer.layers.17.self_attn.k_proj.weight_1bit.bin'
transformer_layers_17_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.17.self_attn.k_proj.weight_scales.txt'
transformer_layers_17_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_17_self_attn_v_proj_weight_packed ← 'transformer.layers.17.self_attn.v_proj.weight_1bit.bin'
transformer_layers_17_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.17.self_attn.v_proj.weight_scales.txt'
transformer_layers_17_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_17_self_attn_o_proj_weight_packed ← 'transformer.layers.17.self_attn.o_proj.weight_1bit.bin'
transformer_layers_17_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.17.self_attn.o_proj.weight_scales.txt'
transformer_layers_17_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_17_mlp_gate_proj_weight_packed ← 'transformer.layers.17.mlp.gate_proj.weight_1bit.bin'
transformer_layers_17_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.17.mlp.gate_proj.weight_scales.txt'
transformer_layers_17_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_17_mlp_up_proj_weight_packed ← 'transformer.layers.17.mlp.up_proj.weight_1bit.bin'
transformer_layers_17_mlp_up_proj_weight_scales_txt ← 'transformer.layers.17.mlp.up_proj.weight_scales.txt'
transformer_layers_17_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_17_mlp_down_proj_weight_packed ← 'transformer.layers.17.mlp.down_proj.weight_1bit.bin'
transformer_layers_17_mlp_down_proj_weight_scales_txt ← 'transformer.layers.17.mlp.down_proj.weight_scales.txt'
transformer_layers_17_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_18_self_attn_q_proj_weight_packed ← 'transformer.layers.18.self_attn.q_proj.weight_1bit.bin'
transformer_layers_18_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.18.self_attn.q_proj.weight_scales.txt'
transformer_layers_18_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_18_self_attn_k_proj_weight_packed ← 'transformer.layers.18.self_attn.k_proj.weight_1bit.bin'
transformer_layers_18_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.18.self_attn.k_proj.weight_scales.txt'
transformer_layers_18_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_18_self_attn_v_proj_weight_packed ← 'transformer.layers.18.self_attn.v_proj.weight_1bit.bin'
transformer_layers_18_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.18.self_attn.v_proj.weight_scales.txt'
transformer_layers_18_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_18_self_attn_o_proj_weight_packed ← 'transformer.layers.18.self_attn.o_proj.weight_1bit.bin'
transformer_layers_18_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.18.self_attn.o_proj.weight_scales.txt'
transformer_layers_18_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_18_mlp_gate_proj_weight_packed ← 'transformer.layers.18.mlp.gate_proj.weight_1bit.bin'
transformer_layers_18_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.18.mlp.gate_proj.weight_scales.txt'
transformer_layers_18_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_18_mlp_up_proj_weight_packed ← 'transformer.layers.18.mlp.up_proj.weight_1bit.bin'
transformer_layers_18_mlp_up_proj_weight_scales_txt ← 'transformer.layers.18.mlp.up_proj.weight_scales.txt'
transformer_layers_18_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_18_mlp_down_proj_weight_packed ← 'transformer.layers.18.mlp.down_proj.weight_1bit.bin'
transformer_layers_18_mlp_down_proj_weight_scales_txt ← 'transformer.layers.18.mlp.down_proj.weight_scales.txt'
transformer_layers_18_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_19_self_attn_q_proj_weight_packed ← 'transformer.layers.19.self_attn.q_proj.weight_1bit.bin'
transformer_layers_19_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.19.self_attn.q_proj.weight_scales.txt'
transformer_layers_19_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_19_self_attn_k_proj_weight_packed ← 'transformer.layers.19.self_attn.k_proj.weight_1bit.bin'
transformer_layers_19_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.19.self_attn.k_proj.weight_scales.txt'
transformer_layers_19_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_19_self_attn_v_proj_weight_packed ← 'transformer.layers.19.self_attn.v_proj.weight_1bit.bin'
transformer_layers_19_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.19.self_attn.v_proj.weight_scales.txt'
transformer_layers_19_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_19_self_attn_o_proj_weight_packed ← 'transformer.layers.19.self_attn.o_proj.weight_1bit.bin'
transformer_layers_19_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.19.self_attn.o_proj.weight_scales.txt'
transformer_layers_19_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_19_mlp_gate_proj_weight_packed ← 'transformer.layers.19.mlp.gate_proj.weight_1bit.bin'
transformer_layers_19_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.19.mlp.gate_proj.weight_scales.txt'
transformer_layers_19_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_19_mlp_up_proj_weight_packed ← 'transformer.layers.19.mlp.up_proj.weight_1bit.bin'
transformer_layers_19_mlp_up_proj_weight_scales_txt ← 'transformer.layers.19.mlp.up_proj.weight_scales.txt'
transformer_layers_19_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_19_mlp_down_proj_weight_packed ← 'transformer.layers.19.mlp.down_proj.weight_1bit.bin'
transformer_layers_19_mlp_down_proj_weight_scales_txt ← 'transformer.layers.19.mlp.down_proj.weight_scales.txt'
transformer_layers_19_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_20_self_attn_q_proj_weight_packed ← 'transformer.layers.20.self_attn.q_proj.weight_1bit.bin'
transformer_layers_20_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.20.self_attn.q_proj.weight_scales.txt'
transformer_layers_20_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_20_self_attn_k_proj_weight_packed ← 'transformer.layers.20.self_attn.k_proj.weight_1bit.bin'
transformer_layers_20_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.20.self_attn.k_proj.weight_scales.txt'
transformer_layers_20_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_20_self_attn_v_proj_weight_packed ← 'transformer.layers.20.self_attn.v_proj.weight_1bit.bin'
transformer_layers_20_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.20.self_attn.v_proj.weight_scales.txt'
transformer_layers_20_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_20_self_attn_o_proj_weight_packed ← 'transformer.layers.20.self_attn.o_proj.weight_1bit.bin'
transformer_layers_20_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.20.self_attn.o_proj.weight_scales.txt'
transformer_layers_20_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_20_mlp_gate_proj_weight_packed ← 'transformer.layers.20.mlp.gate_proj.weight_1bit.bin'
transformer_layers_20_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.20.mlp.gate_proj.weight_scales.txt'
transformer_layers_20_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_20_mlp_up_proj_weight_packed ← 'transformer.layers.20.mlp.up_proj.weight_1bit.bin'
transformer_layers_20_mlp_up_proj_weight_scales_txt ← 'transformer.layers.20.mlp.up_proj.weight_scales.txt'
transformer_layers_20_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_20_mlp_down_proj_weight_packed ← 'transformer.layers.20.mlp.down_proj.weight_1bit.bin'
transformer_layers_20_mlp_down_proj_weight_scales_txt ← 'transformer.layers.20.mlp.down_proj.weight_scales.txt'
transformer_layers_20_mlp_down_proj_weight_shape ← 2048 5632
transformer_layers_21_self_attn_q_proj_weight_packed ← 'transformer.layers.21.self_attn.q_proj.weight_1bit.bin'
transformer_layers_21_self_attn_q_proj_weight_scales_txt ← 'transformer.layers.21.self_attn.q_proj.weight_scales.txt'
transformer_layers_21_self_attn_q_proj_weight_shape ← 2048 2048
transformer_layers_21_self_attn_k_proj_weight_packed ← 'transformer.layers.21.self_attn.k_proj.weight_1bit.bin'
transformer_layers_21_self_attn_k_proj_weight_scales_txt ← 'transformer.layers.21.self_attn.k_proj.weight_scales.txt'
transformer_layers_21_self_attn_k_proj_weight_shape ← 256 2048
transformer_layers_21_self_attn_v_proj_weight_packed ← 'transformer.layers.21.self_attn.v_proj.weight_1bit.bin'
transformer_layers_21_self_attn_v_proj_weight_scales_txt ← 'transformer.layers.21.self_attn.v_proj.weight_scales.txt'
transformer_layers_21_self_attn_v_proj_weight_shape ← 256 2048
transformer_layers_21_self_attn_o_proj_weight_packed ← 'transformer.layers.21.self_attn.o_proj.weight_1bit.bin'
transformer_layers_21_self_attn_o_proj_weight_scales_txt ← 'transformer.layers.21.self_attn.o_proj.weight_scales.txt'
transformer_layers_21_self_attn_o_proj_weight_shape ← 2048 2048
transformer_layers_21_mlp_gate_proj_weight_packed ← 'transformer.layers.21.mlp.gate_proj.weight_1bit.bin'
transformer_layers_21_mlp_gate_proj_weight_scales_txt ← 'transformer.layers.21.mlp.gate_proj.weight_scales.txt'
transformer_layers_21_mlp_gate_proj_weight_shape ← 5632 2048
transformer_layers_21_mlp_up_proj_weight_packed ← 'transformer.layers.21.mlp.up_proj.weight_1bit.bin'
transformer_layers_21_mlp_up_proj_weight_scales_txt ← 'transformer.layers.21.mlp.up_proj.weight_scales.txt'
transformer_layers_21_mlp_up_proj_weight_shape ← 5632 2048
transformer_layers_21_mlp_down_proj_weight_packed ← 'transformer.layers.21.mlp.down_proj.weight_1bit.bin'
transformer_layers_21_mlp_down_proj_weight_scales_txt ← 'transformer.layers.21.mlp.down_proj.weight_scales.txt'
transformer_layers_21_mlp_down_proj_weight_shape ← 2048 5632
lm_head_weight_packed ← 'lm_head.weight_1bit.bin'
lm_head_weight_scales_txt ← 'lm_head.weight_scales.txt'
lm_head_weight_shape ← 32000 2048
