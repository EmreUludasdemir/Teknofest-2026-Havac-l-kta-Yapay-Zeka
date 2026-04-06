$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$gpuPython = 'C:\Users\Emre\.venvs\teknofest-gpu\Scripts\python.exe'
if (Test-Path $gpuPython) {
  & $gpuPython "$scriptDir\..\tools\run_competition_runtime.py" --config "$scriptDir\config\runtime.toml"
} else {
  python "$scriptDir\..\tools\run_competition_runtime.py" --config "$scriptDir\config\runtime.toml"
}
