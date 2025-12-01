⍝ loader_demo.apl - Show how to load a quantized model and call runtime backend

⍝ This example runs a Python wrapper that loads a manifest and calls backend
⍝ script `cpp/call_backend.py` with a sample manifest in the repo.

manifest ← 'student_quantized_manifest.json'
input_path ← 'test_input.txt'

⍝ Invoke the Python wrapper which calls the shared library backend_1bit
cmd ← 'python3 cpp/call_backend.py --manifest ' , manifest , ' --input ' , input_path
⍝ Print command then invoke it
cmd
'] sh cmd
