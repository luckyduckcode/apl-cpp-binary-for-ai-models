∇ Z ← X FFN W1;B1;W2;B2
  ⍝ Feed-Forward Network in APL
  ⍝ X: input matrix
  ⍝ W1, B1: weights and bias for first layer
  ⍝ W2, B2: for second layer
  
  H ← 0⌈ (X +.× W1) + B1  ⍝ ReLU activation
  Z ← (H +.× W2) + B2
∇