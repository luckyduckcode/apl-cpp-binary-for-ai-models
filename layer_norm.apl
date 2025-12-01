∇ Z ← X LAYER_NORM GAMMA;BETA;EPS
  ⍝ Layer Normalization in APL
  ⍝ X: input matrix (batch x seq x d or seq x d)
  ⍝ GAMMA, BETA: scale and shift (vectors of size d)
  
  EPS ← 1e¯6
  dims ← ⍴X
  d ← ⊃⌽dims  ⍝ Last dimension (features)
  mean ← (+/[⌈/⍳⍴dims] X) ÷ d  ⍝ Mean over last axis
  var ← (+/[⌈/⍳⍴dims] (X - mean) * 2) ÷ d  ⍝ Variance over last axis
  Z ← GAMMA × (X - mean) ÷ √(var + EPS) + BETA
∇