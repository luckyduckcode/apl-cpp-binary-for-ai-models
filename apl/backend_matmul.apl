⍝ backend_matmul.apl - APL helper to call the Python backend wrapper
⍝ Usage: set MANIFEST ← 'models/tinyllama_manifest.json', then: 10 CALL-MATMUL 'embedding.weight'
⍝ This script simply shells out to the Python CLI and prints JSON result; parsing JSON in APL is non-trivial,
⍝ so the demo prints a result the user can capture or process in an external script.

MANIFEST ← 'student_quantized_manifest.json' ⍝ default; change to your manifest path
CALLBACK ← 'python cpp/call_backend.py'

∇ Z ← WEIGHT
  cmd ← CALLBACK , ' --manifest ' , MANIFEST , ' --weight ' , WEIGHT , ' --out-json'
  ⍝ call shell and print JSON
  ] sh cmd
  Z ← 0
∇

⍝ Helper to write output to a file instead of JSON on stdout
∇ Z ← WEIGHT; OUTFILE
  cmd ← CALLBACK , ' --manifest ' , MANIFEST , ' --weight ' , WEIGHT , ' --out-file ' , OUTFILE
  ] sh cmd
  Z ← OUTFILE
∇
