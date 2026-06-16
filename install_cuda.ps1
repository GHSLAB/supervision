# =============================================================================
# supervision .venv CUDA 加速安装脚本 (PowerShell)
# 自动生成于 2026-06-16,与 install_cuda.txt 配套使用
# =============================================================================
# 使用方法:右键 → "使用 PowerShell 运行",或在项目根目录下执行:
#     powershell -ExecutionPolicy Bypass -File .\install_cuda.ps1
# =============================================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($n, $title) {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "[Step $n] $title" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
}

function Pause($msg = "按 Enter 继续,Ctrl+C 取消...") {
    Write-Host $msg -ForegroundColor Yellow
    Read-Host | Out-Null
}

# -----------------------------------------------------------------------------
# 第 0 步:环境探测
# -----------------------------------------------------------------------------
Step 0 "环境探测"
Write-Host "Python:" -ForegroundColor Green
& .\.venv\Scripts\python.exe -c "import sys; print('  ', sys.executable); print('  ', sys.version)"

Write-Host "`nGPU:" -ForegroundColor Green
$nsmi = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null)
if ($nsmi) { $nsmi | ForEach-Object { Write-Host "   $_" } } else { Write-Host "  (nvidia-smi 不可用,GPU 加速可能失败)" -ForegroundColor Red }

Write-Host "`n当前 torch:" -ForegroundColor Green
& .\.venv\Scripts\python.exe -c "import torch; print('  version:', torch.__version__); print('  cuda available:', torch.cuda.is_available())"

Pause

# -----------------------------------------------------------------------------
# 第 1 步:替换 torch 为 CUDA 12.6 版
# -----------------------------------------------------------------------------
Step 1 "装 CUDA 版 torch + torchvision (约 1.2 GB)"
uv pip install `
    "torch==2.12.0+cu126" `
    "torchvision==0.27.0+cu126" `
    --extra-index-url https://download.pytorch.org/whl/cu126 `
    --python .venv `
    --reinstall
if ($LASTEXITCODE -ne 0) { throw "Step 1 failed" }
Pause

# -----------------------------------------------------------------------------
# 第 2 步:锁住 pydeprecate,避免被 inference 拉低
# -----------------------------------------------------------------------------
Step 2 "锁住 pydeprecate >= 0.9.0"
uv pip install "pydeprecate>=0.9.0" --python .venv
if ($LASTEXITCODE -ne 0) { throw "Step 2 failed" }
Pause

# -----------------------------------------------------------------------------
# 第 3 步:重装其他 example 依赖 (supervision 仍保持本地 editable)
# -----------------------------------------------------------------------------
Step 3 "装 examples_cn 所需依赖 (9 个包)"
uv pip install `
    gdown `
    inference `
    "jsonargparse[signatures]" `
    pytubefix `
    requests `
    rfdetr `
    tqdm `
    ultralytics `
    yt_dlp `
    --python .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Step 3 had warnings (transitive conflicts), continuing" -ForegroundColor Yellow }
Pause

# -----------------------------------------------------------------------------
# 第 4 步:验证 CUDA
# -----------------------------------------------------------------------------
Step 4 "验证 CUDA 可用性"
& .\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if ($LASTEXITCODE -ne 0) { throw "Step 4 failed" }

# 如果 CUDA 不可用,询问是否继续
$cudaCheck = & .\.venv\Scripts\python.exe -c "import torch; print('OK' if torch.cuda.is_available() else 'NO')"
if ($cudaCheck -ne "OK") {
    Write-Host "`nCUDA 不可用,可能驱动/PyTorch 不匹配" -ForegroundColor Red
    Pause "仍然继续跑例子吗? (按 Enter 继续,Ctrl+C 取消)"
}

# -----------------------------------------------------------------------------
# 第 5 步:跑 traffic_analysis 例子
# -----------------------------------------------------------------------------
Step 5 "跑 traffic_analysis/ultralytics_example.py"
if (-not (Test-Path "examples_cn\traffic_analysis\data\traffic_analysis.pt")) {
    Write-Host "找不到 examples_cn\traffic_analysis\data\traffic_analysis.pt" -ForegroundColor Red
    Write-Host "请先运行 setup.sh (PowerShell 版):" -ForegroundColor Yellow
    Write-Host '  $env:PATH = "C:\uv_project\supervision\.venv\Scripts;$env:PATH"' -ForegroundColor Yellow
    Write-Host '  gdown -O "examples_cn\traffic_analysis\data\traffic_analysis.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"' -ForegroundColor Yellow
    Write-Host '  gdown -O "examples_cn\traffic_analysis\data\traffic_analysis.pt"  "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"' -ForegroundColor Yellow
    throw "data files missing"
}
& .\.venv\Scripts\python.exe examples_cn\traffic_analysis\ultralytics_example.py `
    --source_weights_path "examples_cn\traffic_analysis\data\traffic_analysis.pt" `
    --source_video_path   "examples_cn\traffic_analysis\data\traffic_analysis.mov" `
    --target_video_path   "examples_cn\traffic_analysis\data\output.mp4"
if ($LASTEXITCODE -ne 0) { throw "Step 5 failed" }

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "完成!输出文件: examples_cn\traffic_analysis\data\output.mp4" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
