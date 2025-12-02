⍝ loader_demo.apl - Show how to load a quantized model and call runtime backend

⍝ This example runs a Python wrapper that loads a manifest and calls backend
⍝ script `cpp/call_backend.py` with a sample manifest in the repo.

manifest ← 'student_quantized_manifest.json'
input_path ← 'test_input.txt'

⍝ Generate an APL-friendly snippet from the manifest so demos can use simple variables
cmd_gen ← 'python scripts/manifest_to_apl.py --manifest ' , manifest , ' --out_apl apl/generated_manifest.apl'
] sh cmd_gen

⍝ Show generated APL variables for quick debugging
⍝ For POSIX: `cat`; for Windows PowerShell: `type`.
cmd_cat ← 'if [ -f apl/generated_manifest.apl ]; then cat apl/generated_manifest.apl; else type apl\generated_manifest.apl; fi'
] sh cmd_cat

⍝ Invoke the Python wrapper which calls the shared library backend_1bit
cmd ← 'python cpp/call_backend.py --manifest ' , manifest , ' --weight embedding.weight --out-json'
cmd
] sh cmd
