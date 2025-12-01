∇ Z ← Q SELF_ATT K;V;H;d;h
  ⍝ Multi-Head Self-Attention in APL
  ⍝ Q, K, V: seq_len x (n_heads * d_head)
  ⍝ Assumes n_heads divides the last dim
  
  dims ← ⍴Q
  seq_len ← ⊃dims
  total_d ← ⊃⌽dims
  ⍝ The caller should pass the number of heads H (e.g. 8). We assume H is provided.
  h ← H
  d ← total_d ÷ h  ⍝ d_head
  
  ⍝ Split into heads: seq_len x h x d
  Q_heads ← (seq_len, h, d) ⍴ Q
  K_heads ← (seq_len, h, d) ⍴ K
  V_heads ← (seq_len, h, d) ⍴ V
  
  ⍝ Attention per head
  scores ← (Q_heads +.×[3] ⍉[2] K_heads) ÷ √d  ⍝ seq_len x h x seq_len
  attn ← {⍵ ÷ +/[2] ⍵}¨ ↓[2] scores  ⍝ Softmax over seq_len per head
  attn ← ⊃attn  ⍝ Reshape back
  
  ⍝ Weighted sum
  head_out ← attn +.×[3] V_heads  ⍝ seq_len x h x d
  
  ⍝ Concatenate heads
  Z ← (seq_len, total_d) ⍴ head_out
∇