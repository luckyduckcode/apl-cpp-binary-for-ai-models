<#
Build backend_1bit for Windows (PowerShell script)

This script tries to compile the backend using the Visual Studio (MSVC) toolchain
or MinGW-w64 (g++). If neither is present, it will instruct the user to use WSL.
#>

param(
    [string]$Output = "backend_1bit.dll"
)

# Helper: Write message
function Log($msg) { Write-Host "[build_backend_windows] $msg" }

Log "Checking environment for compiler support..."

# Check for g++ (MinGW) on PATH
$gpp = Get-Command g++ -ErrorAction SilentlyContinue
if ($gpp) {
    Log "Found g++ at $($gpp.Path); compiling with g++ (MinGW)"
    # If using MinGW, compile a DLL (shared library) with -shared
    $srcList = @("cpp/backend_1bit.cpp","cpp/bitmatmul_xnor_avx2.cpp")
    $src = ($srcList -join ' ')
    & g++ -O3 -std=c++17 -m64 -fopenmp -shared -static-libgcc -static-libstdc++ -o $Output $src
    # Try to compile loader_example (executable)
    & g++ -O3 -std=c++17 -m64 -o cpp\loader_example.exe cpp\loader_example.cpp || Write-Host "Could not compile loader_example with g++"
    if ($LASTEXITCODE -ne 0) { throw "g++ compile failed with exit code $LASTEXITCODE" }
    Log "Build succeeded: $Output"
    exit 0
}

# Check for VS where cl.exe is available
$cl = Get-Command cl -ErrorAction SilentlyContinue
if ($cl) {
    Log "Found cl.exe at $($cl.Path); compiling with MSVC"
    # Basic MSVC compile steps; user must open Developer Command Prompt for VS (vcvarsall)
    # Note: This is a minimal command; you may need to adjust include/opt flags.
    $includes = ""
    $sources = @("cpp\\backend_1bit.cpp","cpp\\bitmatmul_xnor_avx2.cpp")
    $objFiles = @()
    foreach ($s in $sources) {
        $obj = [System.IO.Path]::ChangeExtension($s, ".obj")
        $objFiles += $obj
        cl /nologo /O2 /EHsc /MD /W3 /c $s
    }
    # Link into a DLL
    $libFiles = $objFiles -join ' '
    link /DLL /OUT:$Output $libFiles
    # Build loader_example with MSVC
    cl /nologo /O2 /EHsc /MD /W3 /Fe:cpp\loader_example.exe cpp\loader_example.cpp
    if ($LASTEXITCODE -ne 0) { throw "MSVC link failed with exit code $LASTEXITCODE" }
    Log "Build succeeded: $Output"
    exit 0
}

# Neither compiler found; recommend using WSL
Log "Neither g++ nor MSVC found on PATH. If you have WSL, run the build script inside WSL: bash scripts/build_backend.sh"
Log "Alternatively, install MinGW-w64 and add it to PATH, or open Visual Studio Developer Command Prompt and try again."
exit 1
