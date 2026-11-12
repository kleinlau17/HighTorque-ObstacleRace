#!/usr/bin/env bash
set -eo pipefail
nav_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$nav_root"
source /opt/ros/noetic/setup.bash
source /home/nvidia/sim2real/install/setup.bash
exec .venv/bin/python deploy/navigation_buttons.py "$@"
