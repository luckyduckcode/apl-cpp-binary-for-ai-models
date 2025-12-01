⍝ test.apl - Test the Llama APL implementation

∇ Z ← TOKENS EMBEDDING EMBED_MATRIX
  Z ← EMBED_MATRIX[TOKENS;]
  
∇

∇ Z ← POSITIONS POS_EMBEDDING POS_MATRIX
  Z ← POS_MATRIX[POSITIONS;]
  
∇

∇ Z ← Q SELF_ATT KV
  ⍝ Accept boxed pair (K V) in KV; unbox into K and V numeric matrices
  K ← ⊃KV[1]
  V ← ⊃KV[2]
  scores ← Q +.× ⍉ K
  attn ← scores
  Z ← attn +.× V
∇

∇ Z ← X FFN dummy
  
  tmp1 ← X +.× W1
  
  ⍝ Broadcast bias B1 (length 4*d) across rows
  ones_col_seq ← seq_len 1 ⍴ 1
  B1_row ← 1 (4×d_model) ⍴ B1
  bias1_mat ← ones_col_seq +.× B1_row
  tmp1_bias ← tmp1 + bias1_mat
  H ← 0⌈ tmp1_bias
  
  tmp2 ← H +.× W2
  
  B2_row ← 1 d ⍴ B2
  bias2_mat ← ones_col_seq +.× B2_row
  tmp2_bias ← tmp2 + bias2_mat
  Z ← tmp2_bias
∇

∇ Z ← X LAYER_NORM dummy
  EPS ← 1E¯6
  dims ← ⍴X
  seq_len ← dims[1]
  d ← dims[2]
  ⍝ Skip empty-shape guard (GNU APL defn style doesn't accept colon Indentals in this defn)
  ⍝ Sum across last axis (columns) to get per-row means
  ⍝ Compute row-wise mean and variance
  sum_vec ← +/[2] X
  
  mean_vec ← sum_vec ÷ d
  mean_reshaped ← (seq_len,1) ⍴ mean_vec
  ⍝ Expand mean column into full matrix (seq_len x d)
  ones_row ← 1 d ⍴ 1
  mean_mat ← mean_reshaped +.× ones_row
  centered ← X - mean_mat
  var_vec ← (+/[2] (centered * 2)) ÷ d
  var_reshaped ← (seq_len,1) ⍴ var_vec
  var_mat ← var_reshaped +.× ones_row
  std_mat ← (var_mat + EPS) * 0.5
  X_norm ← centered ÷ std_mat
  ⍝ Expand GAMMA and BETA to full matrices for element-wise ops
  ones_col_seq ← seq_len 1 ⍴ 1
  gamma_row ← 1 d ⍴ GAMMA
  gamma_mat ← ones_col_seq +.× gamma_row
  beta_row ← 1 d ⍴ BETA
  beta_mat ← ones_col_seq +.× beta_row
  Z ← gamma_mat × X_norm + beta_mat
∇

∇ Z ← TOKENS LLAMA_MODEL WEIGHTS
  seq_len ← ⍴TOKENS
  X ← TOKENS EMBEDDING EMBED_MAT
  
  pos ← (⍳seq_len) POS_EMBEDDING POS_MAT
  
  X ← X + pos
  
  X ← X TRANSFORMER_BLOCK 1
  
  X ← X TRANSFORMER_BLOCK 2
  
  logits ← X +.× OUTPUT_PROJ
  Z ← logits
∇

∇ Z ← X TRANSFORMER_BLOCK LAYER_NUM
  
  ⍝ Extract per-layer matrices from 3D arrays; select the `LAYER_NUM` slice
  WQ ← WQ_LAYERS[LAYER_NUM; ; ]
  WK ← WK_LAYERS[LAYER_NUM; ; ]
  WV ← WV_LAYERS[LAYER_NUM; ; ]
  WO ← WO_LAYERS[LAYER_NUM; ; ]
  W1 ← W1_LAYERS[LAYER_NUM; ; ]
  B1 ← B1_LAYERS[LAYER_NUM; ]
  W2 ← W2_LAYERS[LAYER_NUM; ; ]
  B2 ← B2_LAYERS[LAYER_NUM; ]
  GAMMA1 ← GAMMA1_LAYERS[LAYER_NUM; ]
  BETA1 ← BETA1_LAYERS[LAYER_NUM; ]
  GAMMA2 ← GAMMA2_LAYERS[LAYER_NUM; ]
  BETA2 ← BETA2_LAYERS[LAYER_NUM; ]
  ⍝ Per-layer scalars (already slices above for GAMMA/BETA)
  
  ⍝ Attention
  GAMMA ← GAMMA1
  BETA ← BETA1
    GAMMA ← GAMMA1
    BETA ← BETA1
    temp ← X LAYER_NORM 0
   
  
  Q ← (temp) +.× (WQ)
  K ← (temp) +.× (WK)
  V ← (temp) +.× (WV)
  
  ATT_OUT ← Q SELF_ATT (K V)
  
  
  ATT_OUT ← ATT_OUT +.× WO
  
  ⍝ Ensure both are plain 2D numeric arrays before adding
  X ← (seq_len d) ⍴ ,X
  ATT_OUT ← (seq_len d) ⍴ ,ATT_OUT
  
  X ← X + ATT_OUT
  
  ⍝ FFN
  GAMMA ← GAMMA2
  BETA ← BETA2
  temp ← X LAYER_NORM 0
  FFN_OUT ← temp FFN 0
  Z ← X + FFN_OUT
∇

⍝ Dummy data
vocab_size ← 10
d_model ← 8
seq_len ← 3
n_heads ← 1
d_head ← d_model ÷ n_heads

⍝ Dummy weights (random floats for testing)
embed_size ← vocab_size × d_model
embed_rand ← (? embed_size ⍴ 100) ÷ 100
embed_mat ← (vocab_size, d_model) ⍴ embed_rand

pos_size ← seq_len × d_model
pos_rand ← (? pos_size ⍴ 100) ÷ 100
pos_mat ← (seq_len, d_model) ⍴ pos_rand

output_size ← d_model × vocab_size
output_rand ← (? output_size ⍴ 100) ÷ 100
output_proj ← (d_model, vocab_size) ⍴ output_rand

⍝ Global weights
EMBED_MAT ← embed_mat
POS_MAT ← pos_mat
OUTPUT_PROJ ← output_proj

⍝ Layer parameters
NUM_LAYERS ← 2
⍝ Layer 1
wq_size ← d_model × d_model
wq_rand ← (? wq_size ⍴ 100) ÷ 100
LAYER1_WQ ← (d_model, d_model) ⍴ wq_rand

wk_size ← d_model × d_model
wk_rand ← (? wk_size ⍴ 100) ÷ 100
LAYER1_WK ← (d_model, d_model) ⍴ wk_rand

wv_size ← d_model × d_model
wv_rand ← (? wv_size ⍴ 100) ÷ 100
LAYER1_WV ← (d_model, d_model) ⍴ wv_rand

wo_size ← d_model × d_model
wo_rand ← (? wo_size ⍴ 100) ÷ 100
LAYER1_WO ← (d_model, d_model) ⍴ wo_rand

w1_size ← d_model × 4×d_model
w1_rand ← (? w1_size ⍴ 100) ÷ 100
LAYER1_W1 ← (d_model, 4×d_model) ⍴ w1_rand

LAYER1_B1 ← 4×d_model ⍴ 0

w2_size ← 4×d_model × d_model
w2_rand ← (? w2_size ⍴ 100) ÷ 100
LAYER1_W2 ← (4×d_model, d_model) ⍴ w2_rand

LAYER1_B2 ← d_model ⍴ 0
LAYER1_GAMMA1 ← d_model ⍴ 1
LAYER1_BETA1 ← d_model ⍴ 0
LAYER1_GAMMA2 ← d_model ⍴ 1
LAYER1_BETA2 ← d_model ⍴ 0

⍝ Layer 2 (copy of layer 1 for simplicity)
LAYER2_WQ ← LAYER1_WQ
LAYER2_WK ← LAYER1_WK
LAYER2_WV ← LAYER1_WV
LAYER2_WO ← LAYER1_WO
LAYER2_W1 ← LAYER1_W1
LAYER2_B1 ← LAYER1_B1
LAYER2_W2 ← LAYER1_W2
LAYER2_B2 ← LAYER1_B2
LAYER2_GAMMA1 ← LAYER1_GAMMA1
LAYER2_BETA1 ← LAYER1_BETA1
LAYER2_GAMMA2 ← LAYER1_GAMMA2
LAYER2_BETA2 ← LAYER1_BETA2

⍝ Weight arrays as numeric 3D arrays (num_layers x rows x cols)
WQ_LAYERS ← NUM_LAYERS d_model d_model ⍴ LAYER1_WQ , LAYER2_WQ
WK_LAYERS ← NUM_LAYERS d_model d_model ⍴ LAYER1_WK , LAYER2_WK
WV_LAYERS ← NUM_LAYERS d_model d_model ⍴ LAYER1_WV , LAYER2_WV
WO_LAYERS ← NUM_LAYERS d_model d_model ⍴ LAYER1_WO , LAYER2_WO
W1_LAYERS ← NUM_LAYERS d_model (4×d_model) ⍴ LAYER1_W1 , LAYER2_W1
B1_LAYERS ← NUM_LAYERS (4×d_model) ⍴ LAYER1_B1 , LAYER2_B1
W2_LAYERS ← NUM_LAYERS (4×d_model) d_model ⍴ LAYER1_W2 , LAYER2_W2
B2_LAYERS ← NUM_LAYERS d_model ⍴ LAYER1_B2 , LAYER2_B2
GAMMA1_LAYERS ← NUM_LAYERS d_model ⍴ LAYER1_GAMMA1 , LAYER2_GAMMA1
BETA1_LAYERS ← NUM_LAYERS d_model ⍴ LAYER1_BETA1 , LAYER2_BETA1
GAMMA2_LAYERS ← NUM_LAYERS d_model ⍴ LAYER1_GAMMA2 , LAYER2_GAMMA2
BETA2_LAYERS ← NUM_LAYERS d_model ⍴ LAYER1_BETA2 , LAYER2_BETA2

⍝ Dummy tokens
TOKENS ← ? seq_len ⍴ vocab_size

⍝ Small unit tests
⍝ Test LAYER_NORM consistency
TEST_X ← 2 4 ⍴ 1 2 3 4 5 6 7 8
GAMMA ← 4 ⍴ 1
BETA ← 4 ⍴ 0
LN_RESULT ← TEST_X LAYER_NORM 0
⎕ ← 'LAYER_NORM test shape:', ⍴LN_RESULT

⍝ Test SELF_ATT shapes
QTEST ← 2 8 ⍴ ? 16 ⍴ 100
KTEST ← 2 8 ⍴ ? 16 ⍴ 100
VTEST ← 2 8 ⍴ ? 16 ⍴ 100
KVTEST ← (KTEST VTEST)
ATT_TEST ← QTEST SELF_ATT KVTEST
⎕ ← 'SELF_ATT test shape:', ⍴ATT_TEST

⍝ Run the model
RESULT ← TOKENS LLAMA_MODEL 0

⍝ Output shape
⎕ ← 'Output shape:', ⍴RESULT
⎕ ← 'RESULT rank:', ⍴⍴RESULT
⎕ ← 'RESULT first row shape (⍴RESULT[1;]):', ⍴RESULT[1;]
⎕ ← 'Sample output (first 3 rows, first 3 cols):'
⎕ ← 3 3 ↑ RESULT

)OFF