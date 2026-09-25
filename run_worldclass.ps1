param(
  [switch]$Pilot,
  [switch]$Full,
  [switch]$NoInstall,
  [int]$MaxSamples = 0,
  [switch]$AllowMissingSplit,
  [switch]$NoLpips
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $root 'run_worldclass.py'
if (-not (Get-Command py -ErrorAction SilentlyContinue) -and -not (Get-Command python -ErrorAction SilentlyContinue)) {
  throw 'Python 3.10+ is required. Install Python and rerun this launcher.'
}
$python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
$args = @($runner)
if ($Pilot) { $args += '--pilot' } elseif ($Full) { $args += '--full' }
if ($NoInstall) { $args += '--no-install' }
if ($MaxSamples -gt 0) { $args += @('--max-samples', "$MaxSamples") }
if ($AllowMissingSplit) { $args += '--allow-missing-split' }
if ($NoLpips) { $args += '--no-lpips' }
& $python @args
exit $LASTEXITCODE
