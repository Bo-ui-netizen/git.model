param(
  [string]$Device = '0',
  [int]$Epochs = 100,
  [int]$Batch = 4,
  [int]$Seed = 42
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'

@"
path: $root/obb_dataset_synth_ordinary
train: images/train
val: images/val
test: images/test
names:
  0: defect
"@ | Set-Content (Join-Path $root 'data\ablation_ordinary.yaml') -Encoding utf8

@"
path: $root/obb_dataset_synth_constrained
train: images/train
val: images/val
test: images/test
names:
  0: defect
"@ | Set-Content (Join-Path $root 'data\ablation_constrained.yaml') -Encoding utf8

function Train([string]$data, [string]$name) {
  & $python (Join-Path $root 'train_obb.py') --data $data --model (Join-Path $root 'yolo11n-obb.pt') --imgsz 1024 --epochs $Epochs --batch $Batch --device $Device --name $name --workers 0 --seed $Seed
  if ($LASTEXITCODE -ne 0) { throw "Training failed: $name" }
}

Train (Join-Path $root 'data\ablation_ordinary.yaml') 'ablation_ordinary_synth'
Train (Join-Path $root 'data\ablation_constrained.yaml') 'ablation_constrained_stats'
