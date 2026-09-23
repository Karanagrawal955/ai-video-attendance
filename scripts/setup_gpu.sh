#!/usr/bin/env bash
# Swap insightface's CPU/GUI dependencies for the GPU + headless wheels AND
# make sure the CUDA 12 libraries ORT needs are actually loadable.
#
#   ./scripts/setup_gpu.sh                        # onnxruntime-gpu 1.26.0 (CUDA 12)
#   ORT_GPU_VERSION=1.27.0 ./scripts/setup_gpu.sh # driver >= 580
#   SKIP_NVIDIA_WHEELS=1 ./scripts/setup_gpu.sh   # use a system CUDA/cuDNN install
#
# Inside Docker the CUDA base image already provides every library; this script
# is for bare-metal/Linux dev boxes.  ORT's GPU wheel does NOT bundle CUDA/cuDNN:
# without them session creation silently drops CUDA and runs on CPU while
# get_available_providers() still LIST CUDA -- so the final check creates a real
# session and inspects it.
set -euo pipefail

ORT_VERSION="${ORT_GPU_VERSION:-1.26.0}"

echo "-> removing CPU onnxruntime + GUI opencv installed by insightface ..."
pip uninstall -y onnxruntime opencv-python 2>/dev/null || true

echo "-> installing onnxruntime-gpu==${ORT_VERSION} + opencv-python-headless ..."
pip install --no-cache-dir "onnxruntime-gpu==${ORT_VERSION}" "opencv-python-headless"

if [ "${SKIP_NVIDIA_WHEELS:-0}" != "1" ]; then
  echo "-> installing CUDA 12 runtime libraries (nvidia-* wheels) - set SKIP_NVIDIA_WHEELS=1 to skip ..."
  pip install --no-cache-dir nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
      nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12

  # Expose the wheel libs to the dynamic linker for this shell.
  NVIDIA_LIB_DIRS="$(python - <<'PY'
import pathlib
import site

root = pathlib.Path(site.getsitepackages()[0]) / "nvidia"
dirs = [
    root / "cublas" / "lib",
    root / "cudnn" / "lib",
    root / "cuda_runtime" / "lib",
    root / "cufft" / "lib",
    root / "curand" / "lib",
]
print(":".join(str(d) for d in dirs if d.exists()))
PY
)"
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS:+${NVIDIA_LIB_DIRS}:}${LD_LIBRARY_PATH:-}"
  echo "   LD_LIBRARY_PATH += ${NVIDIA_LIB_DIRS}"
fi

echo "-> verifying with a REAL session (provider lists can be misleading) ..."
python - <<'PY'
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
        "FAIL: session fell back to CPUExecutionProvider - CUDA/cuDNN "
        "libraries not loadable (nvidia-* wheels + LD_LIBRARY_PATH, or system CUDA)"
    )
PY
