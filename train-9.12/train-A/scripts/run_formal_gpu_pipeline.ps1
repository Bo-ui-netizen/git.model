$ErrorActionPreference = 'Stop'

$projectRoot = 'D:\GeoROI-YOLO'
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$yoloExe = Join-Path $projectRoot '.venv\Scripts\yolo.exe'
$runsRoot = Join-Path $projectRoot 'runs\detect\runs'
$logsRoot = Join-Path $projectRoot 'outputs\training_logs'
$stage1Dir = Join-Path $runsRoot 'stage1_insulator_rtx4060'
$stage2Dir = Join-Path $runsRoot 'stage2_defect_roi_rtx4060'
$stage1Weights = Join-Path $stage1Dir 'weights\best.pt'
$stage2Weights = Join-Path $stage2Dir 'weights\best.pt'
$finalOut = Join-Path $projectRoot 'outputs\final_infer_rtx4060'
$imagePath = 'D:\A-learn\weichat\xwechat_files\wxid_qqa6ttn73nlc22_158f\msg\file\2026-07\CPLID_YOLO\CPLID_YOLO\full\images\val\0058.jpg'

New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null

function Wait-ForTrainingToFinish([string] $resultsPath, [string] $weightsPath) {
    # YOLO updates results.csv after each epoch. A stable file indicates training has exited.
    while (-not (Test-Path -LiteralPath $weightsPath)) {
        Start-Sleep -Seconds 30
    }

    $lastWrite = (Get-Item -LiteralPath $resultsPath).LastWriteTimeUtc
    while ($true) {
        Start-Sleep -Seconds 45
        $currentWrite = (Get-Item -LiteralPath $resultsPath).LastWriteTimeUtc
        if ($currentWrite -eq $lastWrite) {
            break
        }
        $lastWrite = $currentWrite
    }
}

function Convert-PngsToJpg([string] $directory) {
    & $pythonExe -c "import cv2; from pathlib import Path; p=Path(r'$directory'); [cv2.imwrite(str(f.with_suffix('.jpg')), cv2.imread(str(f))) for f in p.glob('*.png')]"
}

$stage1Results = Join-Path $stage1Dir 'results.csv'
Wait-ForTrainingToFinish $stage1Results $stage1Weights
Convert-PngsToJpg $stage1Dir

$stage2OutLog = Join-Path $logsRoot 'stage2_rtx4060.out.log'
$stage2ErrLog = Join-Path $logsRoot 'stage2_rtx4060.err.log'
$stage2 = Start-Process -FilePath $yoloExe -ArgumentList @(
    'detect', 'train', 'model=yolo11n.pt', 'data=data\cplid_defect_roi_windows.yaml',
    'imgsz=768', 'epochs=120', 'batch=4', 'device=0', 'project=runs',
    'name=stage2_defect_roi_rtx4060', 'exist_ok=True', 'patience=30', 'close_mosaic=10'
) -WorkingDirectory $projectRoot -RedirectStandardOutput $stage2OutLog -RedirectStandardError $stage2ErrLog -WindowStyle Hidden -PassThru
$stage2 | Wait-Process

if (-not (Test-Path -LiteralPath $stage2Weights)) {
    throw 'Stage-two training did not produce best.pt.'
}

Convert-PngsToJpg $stage2Dir
& $pythonExe (Join-Path $projectRoot 'infer_pipeline.py') --image $imagePath --stage1 $stage1Weights --stage2 $stage2Weights --out $finalOut --device 0
