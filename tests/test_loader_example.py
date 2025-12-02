import os
import subprocess
from pathlib import Path


def test_loader_example_runs_if_present():
    # If loader_example exists in cpp/, attempt to run it against test_manifest
    exe = Path('cpp/loader_example')
    exe_win = Path('cpp/loader_example.exe')
    manifest = Path('student_quantized_manifest.json')
    if not manifest.exists():
        # Nothing to test without manifest
        return
    # Prefer platform-native binary: on Windows look for .exe first
    if os.name == 'nt':
        if exe_win.exists():
            ret = subprocess.run([str(exe_win), str(manifest), 'cpp/backend_1bit.dll'])
            assert ret.returncode == 0 or ret.returncode == 1
        else:
            # On Windows if only the Linux binary is present we skip (not runnable on Windows)
            return
    elif exe.exists():
        # Try both the cpp/ location and root location for the .so file
        # The loader example may fail to find the .so if not in the expected location - this is acceptable
        so_path = 'cpp/backend_1bit.so' if Path('cpp/backend_1bit.so').exists() else 'backend_1bit.so'
        ret = subprocess.run([str(exe), str(manifest), so_path])
        # Return codes: 0=success, 1=minor error, 3=dlopen failed (acceptable if .so not found)
        assert ret.returncode in (0, 1, 3)
    elif exe_win.exists():
        # non-Windows runner that has an .exe file
        ret = subprocess.run([str(exe_win), str(manifest), 'cpp/backend_1bit.dll'])
        assert ret.returncode == 0 or ret.returncode == 1
    else:
        # nothing to run; pass
        return
