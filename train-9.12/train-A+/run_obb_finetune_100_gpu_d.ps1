$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.yolo_config'
$env:TORCH_HOME = Join-Path $ProjectRoot '.torch_cache'
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot '.cache_d'
$env:TEMP = Join-Path $ProjectRoot '.tmp'
$env:TMP = Join-Path $ProjectRoot '.tmp'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR,$env:TORCH_HOME,$env:XDG_CACHE_HOME,$env:TEMP | Out-Null
Set-Location $ProjectRoot
& '..\..\.venv\Scripts\python.exe' train_obb.py --model 'runs\stage2_defect_obb_finetune50\weights\best.pt' --data data/cplid_defect_obb.yaml --epochs 100 --batch 2 --imgsz 1024 --device 0 --name stage2_defect_obb_finetune100
