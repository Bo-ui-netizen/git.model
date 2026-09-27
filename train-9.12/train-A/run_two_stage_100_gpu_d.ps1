$ErrorActionPreference = 'Stop'
$env:YOLO_CONFIG_DIR = 'D:\GeoROI-YOLO\.yolo_config'
$env:TORCH_HOME = 'D:\GeoROI-YOLO\.torch_cache'
$env:XDG_CACHE_HOME = 'D:\GeoROI-YOLO\.cache_d'
$env:TEMP = 'D:\GeoROI-YOLO\.tmp'
$env:TMP = 'D:\GeoROI-YOLO\.tmp'
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR,$env:TORCH_HOME,$env:XDG_CACHE_HOME,$env:TEMP | Out-Null
Set-Location 'D:\GeoROI-YOLO'
& '.\.venv\Scripts\python.exe' train_stage1.py --model 'runs\stage1_original_gpu100\weights\best.pt' --data data/cplid_insulator_windows.yaml --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 0 --name stage1_original_finetune100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.\.venv\Scripts\python.exe' train_obb.py --model 'runs\stage2_defect_obb_finetune100\weights\best.pt' --data data/cplid_defect_obb.yaml --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 0 --name stage2_defect_obb_final100
