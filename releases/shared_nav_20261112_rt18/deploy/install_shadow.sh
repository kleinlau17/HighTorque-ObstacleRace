#!/usr/bin/env bash
# Isolated user-space installation; does not start any robot/ROS controller.
set -euo pipefail
nav_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$nav_root"
python3 - <<'PY'
from pathlib import Path
from navigation.release import verify_release
verify_release(Path.cwd())
print('Release source manifest verified')
PY
test -f /usr/local/include/librealsense2/rs.hpp
test -f /opt/ros/noetic/setup.bash
wheel_dir="${NAV_WHEEL_DIR:-$nav_root/wheels}"
pip_options=()
if [ -d "$wheel_dir" ]; then
  pip_options=(--no-index --find-links "$wheel_dir")
fi
python3 -m venv --without-pip --system-site-packages .venv
if .venv/bin/python - <<'PY'
import cv2,numpy as np
from navigation.environment import verify_dependencies
verify_dependencies()
np.random.default_rng(1)
image=np.zeros((12,12,3),np.uint8)
cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.connectedComponents(image[:,:,0])
print('Vendor dependency preflight passed:',np.__version__,cv2.__version__)
PY
then
  printf 'Reusing validated vendor NumPy/OpenCV; no dependency changes.\n'
else
if ! .venv/bin/python -c 'import sys,pip; assert pip.__file__.startswith(sys.prefix+"/")' 2>/dev/null; then
  # Vendor Ubuntu images may omit ensurepip. Bootstrap a venv-local pip wheel.
  mkdir -p .bootstrap
  if [ -f "$wheel_dir/pip-24.3.1-py3-none-any.whl" ]; then
    cp "$wheel_dir/pip-24.3.1-py3-none-any.whl" .bootstrap/
  else
    python3 -m pip download --only-binary=:all: --no-deps --dest .bootstrap pip==24.3.1
  fi
  PYTHONPATH="$nav_root/.bootstrap/pip-24.3.1-py3-none-any.whl" \
    .venv/bin/python -m pip install --no-index --ignore-installed .bootstrap/pip-24.3.1-py3-none-any.whl
fi
.venv/bin/python -m pip install "${pip_options[@]}" --only-binary=:all: \
  -r deploy/requirements-navigation.txt
fi
g++ -O2 -std=c++14 deploy/realsense_stream.cpp \
  -I/usr/local/include -L/usr/local/lib -Wl,-rpath,/usr/local/lib \
  -lrealsense2 -o deploy/realsense_stream
.venv/bin/python -m py_compile navigation/*.py deploy/shared_navigation.py deploy/shadow.py deploy/geometry.py deploy/line_trial.py deploy/navigation_buttons.py deploy/navigation_cli.py deploy/head_step.py
chmod +x deploy/nav
.venv/bin/python -c 'import cv2,numpy; print("Runtime dependencies:",cv2.__version__,numpy.__version__)'
printf 'Installed shared navigation in %s; standalone entry is shadow-only\n' "$nav_root"
printf 'No robot control nodes were started. See deploy/README.md for running the sensor check.\n'
