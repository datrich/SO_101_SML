#!/usr/bin/env bash
# Xuất SO_101_SML từ Onshape -> description/SO_101_SML.urdf (+ bổ sung vật lý qua scripts/so101_physics.py)
# Dùng: ./export.sh          (cần file .env chứa Onshape API key, xem .env.example)
# Env python có onshape-to-robot: mặc định conda env "onshape", đổi bằng ONSHAPE_ENV=/path/to/env
set -e
cd "$(dirname "$0")"
ONSHAPE_ENV="${ONSHAPE_ENV:-$HOME/miniconda3/envs/onshape}"
export PATH="$ONSHAPE_ENV/bin:$PATH"
[ -f .env ] || [ -n "$ONSHAPE_ACCESS_KEY" ] || { echo "Thiếu API key — chạy: cp .env.example .env rồi điền key"; exit 1; }
exec onshape-to-robot description
