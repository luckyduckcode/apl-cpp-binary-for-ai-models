⍝ Llama.apl - Full Llama implementation

∇ Z ← TOKENS LLAMA_MODEL WEIGHTS
  ⍝ Full Llama model
  ⍝ TOKENS: vector of token indices
  ⍝ WEIGHTS: structure with all weights
  
  seq_len ← ⍴TOKENS
  embed_mat ← WEIGHTS.embed
  pos_mat ← WEIGHTS.pos
  layers ← WEIGHTS.layers  ⍝ Array of layer weights
  output_proj ← WEIGHTS.output
  
  ⍝ Embeddings
  X ← TOKENS EMBEDDING embed_mat
  pos ← (⍳seq_len) POS_EMBEDDING pos_mat
  X ← X + pos
  
  ⍝ Transformer layers (iterate over configured layers)
  n_layers ← ⍴layers
  :For i :In (⍳ n_layers)
    X ← X TRANSFORMER_BLOCK layers[i;]
  :EndFor
  
  ⍝ Output projection
  logits ← X +.× output_proj
  Z ← logits
∇

∇ Z ← X TRANSFORMER_BLOCK LAYER_WEIGHTS
  ⍝ One transformer block
  WQ ← LAYER_WEIGHTS.WQ
  WK ← LAYER_WEIGHTS.WK
  WV ← LAYER_WEIGHTS.WV
  WO ← LAYER_WEIGHTS.WO
  W1 ← LAYER_WEIGHTS.W1
  B1 ← LAYER_WEIGHTS.B1
  W2 ← LAYER_WEIGHTS.W2
  B2 ← LAYER_WEIGHTS.B2
  GAMMA1 ← LAYER_WEIGHTS.GAMMA1
  BETA1 ← LAYER_WEIGHTS.BETA1
  GAMMA2 ← LAYER_WEIGHTS.GAMMA2
  BETA2 ← LAYER_WEIGHTS.BETA2
  
  Q ← X +.× WQ
  K ← X +.× WK
  V ← X +.× WV
  
  ⍝ Pass explicit number of heads if available in layer weights; default to 8
  H ← (⍬)  ⍝ default empty
  ⍝ If the weight structure includes 'NHEADS', use that field.
  :If (⍴LAYER_WEIGHTS) > 0
    :Try
      H ← LAYER_WEIGHTS.NHEADS
    :Catch
      H ← 8
    :EndTry
  :Else
    H ← 8
  :EndIf
  ATT_OUT ← Q SELF_ATT K V H
  ATT_OUT ← ATT_OUT +.× WO
  
  X ← X LAYER_NORM GAMMA1 BETA1 + ATT_OUT
  
  FFN_OUT ← X FFN W1 B1 W2 B2
  
  Z ← X LAYER_NORM GAMMA2 BETA2 + FFN_OUT
∇