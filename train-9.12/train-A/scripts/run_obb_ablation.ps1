param([string]$Device='0',[int]$Epochs=100,[int]$Batch=4)
$py=Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'; $root=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
function Train([string]$data,[string]$name){& $py (Join-Path $root 'train_obb.py') --data $data --model (Join-Path $root 'yolo11n-obb.pt') --imgsz 1024 --epochs $Epochs --batch $Batch --device $Device --name $name --workers 0}
Train (Join-Path $root 'data\cplid_defect_obb.yaml') 'ablation_no_synth'
Copy-Item (Join-Path $root 'data\cplid_defect_obb.yaml') (Join-Path $root 'data\ablation_ordinary.yaml') -Force
(Get-Content (Join-Path $root 'data\ablation_ordinary.yaml')) -replace 'obb_dataset','obb_dataset_synth_ordinary' | Set-Content (Join-Path $root 'data\ablation_ordinary.yaml')
Train (Join-Path $root 'data\ablation_ordinary.yaml') 'ablation_ordinary_synth'
Copy-Item (Join-Path $root 'data\cplid_defect_obb.yaml') (Join-Path $root 'data\ablation_constrained.yaml') -Force
(Get-Content (Join-Path $root 'data\ablation_constrained.yaml')) -replace 'obb_dataset','obb_dataset_synth_constrained' | Set-Content (Join-Path $root 'data\ablation_constrained.yaml')
Train (Join-Path $root 'data\ablation_constrained.yaml') 'ablation_constrained_stats'
Write-Host 'Done: runs/ablation_no_synth, runs/ablation_ordinary_synth, runs/ablation_constrained_stats'
