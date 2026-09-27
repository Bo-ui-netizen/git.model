param(
  [string]$Src = "datasets/CPLID_ROI",
  [string]$Device = "0",
  [int]$Epochs = 120,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Invoke-Stage2([string]$name) {
  & $python (Join-Path $root "train_stage2.py") --data (Join-Path $root "data\cplid_defect_roi.yaml") --model (Join-Path $root "yolo11n.pt") --imgsz 768 --epochs $Epochs --batch 2 --device $Device --name $name
}

Write-Host "[1/3] Baseline: no synthetic augmentation"
Invoke-Stage2 "ablation_no_synth"

Write-Host "[2/3] Ordinary random synthesis"
& $python (Join-Path $root "scripts\synth_defects.py") --src $Src --split train --ratio 0.5 --mode ordinary --seed $Seed
Invoke-Stage2 "ablation_ordinary_synth"

Write-Host "[3/3] Constrained/statistics-aligned synthesis"
& $python (Join-Path $root "scripts\synth_defects.py") --src $Src --split train --ratio 0.5 --mode constrained --stats-split train --seed $Seed --position-jitter 0.03 --size-jitter 0.12 --appearance dark
Invoke-Stage2 "ablation_constrained_stats"

Write-Host "Results are under runs/ablation_no_synth, runs/ablation_ordinary_synth and runs/ablation_constrained_stats"
