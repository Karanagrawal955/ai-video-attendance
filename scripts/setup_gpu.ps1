# Swap insightface's CPU/GUI dependencies for the GPU + headless wheels AND
# make sure the CUDA 12 libraries ORT needs are actually loadable.
#
#   .\scripts\setup_gpu.ps1                            # onnxruntime-gpu 1.26.0 (CUDA 12)
#   $env:ORT_GPU_VERSION="1.27.0"; .\scripts\setup_gpu.ps1   # driver >= 580
#   $env:SKIP_NVIDIA_WHEELS="1"; .\scripts\setup_gpu.ps1     # use a system CUDA/cuDNN install
#
# ORT's GPU wheel does NOT bundle CUDA/cuDNN.  Without either a system CUDA
# install or the pip nvidia-* packages below, session creation silently drops
# CUDA and runs on CPU -- while get_available_providers() still LIST CUDA.
# The final check therefore creates a real session and inspects it.
$ErrorActionPreference = "Stop"

$ortVersion = if ($env:ORT_GPU_VERSION) { $env:ORT_GPU_VERSION } else { "1.26.0" }

Write-Host "-> removing CPU onnxruntime + GUI opencv installed by insightface ..."
pip uninstall -y onnxruntime opencv-python 2>$null

Write-Host "-> installing onnxruntime-gpu==$ortVersion + opencv-python-headless ..."
pip install --no-cache-dir "onnxruntime-gpu==$ortVersion" "opencv-python-headless"

if ($env:SKIP_NVIDIA_WHEELS -ne "1") {
    Write-Host "-> installing CUDA 12 runtime libraries (nvidia-* wheels, ~1.5 GB) ..."
    pip install --no-cache-dir nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 `
        nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
}

# Make the wheel DLLs visible to onnxruntime for THIS shell (persist by adding
# the same directories to your user PATH if you want them outside this session).
$siteNvidia = python -c "import pathlib, site; print(pathlib.Path(site.getsitepackages()[0]) / 'nvidia')"
$dirs = @("cublas\bin", "cudnn\bin", "cuda_runtime\bin", "cufft\bin", "curand\bin") |
    ForEach-Object { Join-Path $siteNvidia $_ } | Where-Object { Test-Path $_ }
if ($dirs) {
    $env:PATH = ($dirs -join ';') + ';' + $env:PATH
    Write-Host "-> added to PATH for this session:"
    $dirs | ForEach-Object { Write-Host "     $_" }
}

Write-Host "-> verifying with a REAL session (provider lists can be misleading) ..."
python -c @'
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto as TP

node = helper.make_node("Add", ["X", "Y"], ["Z"])
graph = helper.make_graph(
    [node], "probe",
    [helper.make_tensor_value_info("X", TP.FLOAT, [1]),
     helper.make_tensor_value_info("Y", TP.FLOAT, [1])],
    [helper.make_tensor_value_info("Z", TP.FLOAT, [1])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 10
sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
active = sess.get_providers()
print(f"onnxruntime {ort.__version__} active providers: {active}")
if active and active[0] == "CUDAExecutionProvider":
    print("OK: CUDA session established")
else:
    raise SystemExit(
        "FAIL: session fell back to CPUExecutionProvider - CUDA/cuDNN DLLs "
        "not loadable (check nvidia-* wheels + PATH)"
    )
'@
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
