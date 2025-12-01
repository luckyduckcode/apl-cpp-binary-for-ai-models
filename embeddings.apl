∇ Z ← TOKENS EMBEDDING EMBED_MATRIX
  ⍝ Token Embedding in APL
  ⍝ TOKENS: vector of token indices
  ⍝ EMBED_MATRIX: vocab_size x d_model
  
  Z ← EMBED_MATRIX[TOKENS;]  ⍝ Select rows
∇

∇ Z ← POSITIONS POS_EMBEDDING POS_MATRIX
  ⍝ Positional Embedding in APL
  ⍝ POSITIONS: vector of positions (0 to seq_len-1)
  ⍝ POS_MATRIX: max_len x d_model
  
  Z ← POS_MATRIX[POSITIONS;]
∇