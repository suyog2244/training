#!/bin/bash
# =============================================================================
# Piper TTS — Complete Standalone Training Setup Script
# Based on: https://github.com/rhasspy/piper/blob/master/TRAINING.md
#
# KNOWN ISSUES HANDLED BY THIS SCRIPT:
#
#   1. pip >=24.1 refuses yanked packages (pytorch-lightning==1.7.7 was yanked).
#      FIX: pin pip to 24.0 before running `pip install -e .`
#      REF: https://github.com/rhasspy/piper/issues/558
#
#   2. `cythonize: command not found` when running build_monotonic_align.sh.
#      FIX: install Cython>=0.29.0 before running the build script.
#      REF: https://github.com/rhasspy/piper/issues/689
#
#   3. monotonic_align build must happen AFTER `pip install -e .` so that
#      the piper_train package path exists for the .so to land in.
#
# PYTHON REQUIREMENT: Python 3.10
#   piper-phonemize binary wheels exist only for 3.9 / 3.10 / 3.11.
#
# Usage:
#   chmod +x setup_piper_train.sh
#   ./setup_piper_train.sh
#
# All options can be overridden as environment variables (see CONFIG below).
# =============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (override via env vars or edit here)
# ──────────────────────────────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-$HOME/test_dataset}"
TRAINING_DIR="${TRAINING_DIR:-$HOME/training_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/piper_output}"

LANGUAGE="${LANGUAGE:-en}"

SAMPLE_RATE="${SAMPLE_RATE:-22050}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
QUALITY="${QUALITY:-high}"

CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
FINETUNE_CKPT="${FINETUNE_CKPT:-}"
S3_DATASET_URI="${S3_BUCKET_URI:-s3://ai-labs-5497-ml-bucket/test_dataset/}"
S3_DATASET_ZIP="${S3_DATASET_ZIP:-https://ai-labs-5497-ml-bucket.s3.ap-south-1.amazonaws.com/test_dataset.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAoaCmFwLXNvdXRoLTEiRzBFAiAGvsyjJlZox54Fy1u8ZGvtgejO4%2Bj7npG3s183wSE8DgIhAKV%2Bs6pgN6VaHr7xJeeJviEGh5Sa5dT6TugoubO9K7D7KtsDCNP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQABoMNTQ5NzEwNDE2NTQzIgxl1LyRZqNs3XRg6oYqrwO64qKA1UtH516f%2Fm68LZjUbLRcBjjfw%2FX3FY%2BydzACqqdLC7tV9N7ECFTlTaj5Yhfc4o3wskQjqRa%2BESXD%2F3gB7xa2IMbZToOUaeQYBrT2kmwNrUiyx6fUWT4OER%2BKVdn2EbHjZtWMXb7VaKS3kEXzLde1Dm3lxT%2BWnmUXdMlt2%2Ff7Pm75M%2BUhVc8KhxeC5BEObYrjebLJRRhVv6aIreZnpOdgXxG1Cs8%2B2zU04U1OJkWSX4EojmKMPhaos9bJ4gsjiAl7ge6bNQagMC1tIFNV4B2fm5CD5%2FuUJd7aAnCcclrw4nbk2tQkk9x0gT9egCn7o%2BvZAw2oRsFNViL4%2BMgPBEpbogInSqCnhQNdaWiHyH8RMTid2AeQooS1h8NXD1sokeTsMicfaJmtxqylcLa%2BARxb5wHMESfwgIGRZcIgl4RV7K2ib%2FR5UAAphKjqpxSgR1QIXmi57uh%2FB1Z3lvlpdSax2YqnVfaBcgog5aVNNBg4o%2FEYAPZ1DfSqMcmoWWgwcyvt3D0g5PwucVgjzSTRESEix4iv2Wai4n2bZBCa9vvyVMouG%2Faku6BNdA8%2B1zDEsd7NBjreAvDuIlB74clxKlOE5I%2F41d7eymjhdpEOmrARjzhmdzGtBhfA0z0%2Ba%2FN8RIou1MZq%2FBNGKHcs2vb3%2FMjQ25E7%2FTsU2LVtwynTxhkpbfZlo%2FDdTIKoA2sS9cwgpN8wLIQv%2FTqPZ%2B1Y4aAu0JIIIB78ZB56LFwIK3brCuMrtfh3vokCkaTWOHrdUi7jNV8Wlq356lCoixaWuDqX7dpexmXr7Klp5gcJ8kDh6WW8SDBMXAS86FxN0MDm8Wq0%2BsPu%2BI8Kda8ul249mqnyJEAGOG4sYcj%2B5zlA1nQ0PuT%2FDfl1tA9TAtF%2Fn%2BiU2hCF%2FsWv5riUaPEkh6cWJW1T%2Bpw6J2%2B5ilmu962hqbcxd71vGo9DxkLK5%2B6NqVej3BCTCixj7LXiPzDxP54YvNdkyRa2h8pCoHquVTwgBJHsGJsA4eiKIFgZcR8Mw4tf1sLHIUlsZO2pzBztWbZeYKkKDVbwgfG2&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAX77KLJKPRA23HHMQ%2F20260316%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20260316T100556Z&X-Amz-Expires=43200&X-Amz-SignedHeaders=host&X-Amz-Signature=35dbdca3775ff1ec6d7a2d2b30d8e363c27f725fe22d57fec6b2436ad1a22f0b}"
S3_BUCKET="${S3_BUCKET:-s3://ai-labs-5497-ml-bucket/}"
S3_PREFIX="${S3_PREFIX:-piper-checkpoints}"

UPLOAD_ONNX="${UPLOAD_ONNX:-true}"
MAX_WORKERS="${MAX_WORKERS:-4}"

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"

PRECISION="${PRECISION:-32}"

PIPER_DIR="$HOME/piper"
PYTHON_BIN=""   # resolved after Python 3.10 install

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
warn() { echo -e "\033[1;33mWARN: $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — System dependencies + Python 3.10
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 1 — Installing system dependencies"
sudo apt-get install -y \
  software-properties-common \
  build-essential \
  git \
  ffmpeg \
  curl \
  unzip \
  espeak-ng \
  espeak-ng-data \
  libespeak-ng-dev

# Python 3.10 required — piper-phonemize wheels only exist for 3.9/3.10/3.11
if command -v python3.10 &>/dev/null; then
  log "Python 3.10 already present: $(python3.10 --version)"
else
  log "Installing Python 3.10 via deadsnakes PPA"
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -y
  sudo apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi
PYTHON_BIN="$(command -v python3.10)"
log "Using: $PYTHON_BIN  ($(${PYTHON_BIN} --version))"

# AWS CLI — only if S3 upload is requested
if [[ -n "$S3_BUCKET" ]] && ! command -v aws &>/dev/null; then
  log "Installing AWS CLI v2"
  cd /tmp
  curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
  unzip -q awscliv2.zip
  sudo ./aws/install --update
  rm -rf awscliv2.zip aws/
  cd "$HOME"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Clone / update Piper
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 2 — Cloning / updating Piper repository"
if [[ -d "$PIPER_DIR/.git" ]]; then
  log "Piper already cloned — pulling latest changes"
  git -C "$PIPER_DIR" pull
else
  git clone https://github.com/rhasspy/piper.git "$PIPER_DIR"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Python 3.10 virtual environment
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 3 — Creating Python 3.10 virtual environment"
VENV_DIR="$PIPER_DIR/src/python/.venv"
cd "$PIPER_DIR/src/python"

# Recreate venv if it was built with the wrong Python
if [[ -d "$VENV_DIR" ]]; then
  VENV_PY_VER=$("$VENV_DIR/bin/python3" --version 2>/dev/null | awk '{print $2}' || echo "")
  if [[ "$VENV_PY_VER" != 3.10.* ]]; then
    warn "Existing venv uses Python $VENV_PY_VER (need 3.10) — recreating"
    rm -rf "$VENV_DIR"
  fi
fi

[[ -d "$VENV_DIR" ]] || "$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Install packages
#
# ORDER IS CRITICAL:
#   a) pin pip to 24.0  — prevents refusal of yanked pytorch-lightning==1.7.7
#   b) install Cython   — required by build_monotonic_align.sh (cythonize cmd)
#   c) install piper-train deps manually (avoids pip re-resolving the yanked pkg)
#   d) pip install -e . — installs piper_train into the venv package tree
#   e) build monotonic_align — MUST come after -e . so the package path exists
#   f) install remaining audio/utility deps
#   g) install torchmetrics 0.11.4 — required by pytorch-lightning 1.x
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 4a — Pinning pip to 24.0 (workaround for yanked pytorch-lightning)"
pip install "pip==24.0" setuptools wheel

log "STEP 4b — Installing Cython (required by build_monotonic_align.sh)"
pip install "Cython>=0.29.0"

log "STEP 4c — Installing PyTorch"
pip install \
  "torch>=2.0,<3" \
  torchvision \
  torchaudio

log "STEP 4d — Pre-installing pytorch-lightning 1.9.5 and torchmetrics 0.11.4"
# pytorch-lightning 1.7.7 is yanked; we pre-install 1.9.5 (last stable 1.x).
# torchmetrics 0.11.4 is required by pytorch-lightning 1.x.
pip install \
  "pytorch-lightning==1.9.5" \
  "torchmetrics==0.11.4"

log "STEP 4e — Installing piper-phonemize and phonemizer"
pip install \
  "piper-phonemize==1.1.0" \
  "phonemizer>=3.2,<4"

log "STEP 4f — Installing piper_train in editable mode (--no-deps skips yanked lightning re-resolve)"
# --no-deps is critical: prevents pip from overwriting our pytorch-lightning
# 1.9.5 with the yanked 1.7.7 that piper-train's setup.py pins.
pip install --no-deps -e .

# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Build monotonic_align (must come AFTER pip install -e .)
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 5 — Building monotonic_align Cython extension"
BUILD_SCRIPT="$PIPER_DIR/src/python/build_monotonic_align.sh"
MONO_DIR="$PIPER_DIR/src/python/piper_train/vits/monotonic_align"

if [[ -f "$BUILD_SCRIPT" ]]; then
  cd "$PIPER_DIR/src/python"
  # Ensure the venv's cythonize is on PATH
  export PATH="$VENV_DIR/bin:$PATH"
  bash "$BUILD_SCRIPT"
elif [[ -f "$MONO_DIR/setup.py" ]]; then
  log "No build script — compiling via setup.py directly"
  cd "$MONO_DIR"
  python3 setup.py build_ext --inplace
  cd "$PIPER_DIR/src/python"
else
  warn "monotonic_align source not found — attempting python -m cython fallback"
  if [[ -f "$MONO_DIR/core.pyx" ]]; then
    cd "$MONO_DIR"
    python3 -m cython core.pyx
    python3 -c "
from distutils.core import setup
from distutils.extension import Extension
setup(ext_modules=[Extension('core', ['core.c'])])
" build_ext --inplace
    cd "$PIPER_DIR/src/python"
  else
    warn "Could not find monotonic_align source — training will fail at import time"
  fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Install remaining dependencies
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 6 — Installing remaining dependencies"
pip install \
  "onnx>=1.13,<2" \
  "onnxruntime>=1.14,<2" \
  "onnxsim" \
  "librosa>=0.10" \
  audioread \
  soundfile \
  numpy \
  pandas \
  scipy \
  matplotlib \
  tqdm \
  PyYAML \
  requests \
  packaging \
  coloredlogs \
  inflect \
  "networkx>=2.8,<4" \
  "numba>=0.57" \
  Jinja2 \
  jsonschema \
  protobuf \
  filelock \
  fsspec \
  regex \
  babel \
  isodate \
  language-tags \
  csvw \
  rdflib \
  pathvalidate \
  msgpack \
  pooch \
  lazy_loader \
  decorator \
  cycler \
  kiwisolver \
  pyparsing \
  python-dateutil \
  Pygments \
  MarkupSafe \
  certifi \
  idna \
  charset-normalizer \
  more-itertools

log "Installation complete."
log "  pytorch-lightning : $(pip show pytorch-lightning | grep '^Version')"
log "  Cython            : $(pip show Cython | grep '^Version')"
log "  torch             : $(pip show torch | grep '^Version')"
# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Download dataset from S3
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 7 — Downloading dataset zip from $S3_DATASET_ZIP"
command -v aws &>/dev/null || die "'aws' CLI not found. It is required to download the dataset."
 
aws sts get-caller-identity &>/dev/null || \
  die "AWS credentials not configured (set AWS_ACCESS_KEY_ID/SECRET or attach an IAM role)."
 
# Install unzip if missing
command -v unzip &>/dev/null || sudo apt-get install -y unzip
 
ZIP_LOCAL="/tmp/test_dataset.zip"
 
log "Downloading s3 zip → $ZIP_LOCAL"
aws s3 cp "$S3_DATASET_ZIP" "$ZIP_LOCAL" \
  --region "$S3_DATASET_REGION"
 
log "Extracting $ZIP_LOCAL → $DATASET_DIR"
mkdir -p "$DATASET_DIR"
unzip -o "$ZIP_LOCAL" -d "$DATASET_DIR"
 
# If the zip contains a single top-level subdirectory (e.g. test_dataset/),
# flatten it so DATASET_DIR directly holds metadata.csv and wav/.
EXTRACTED_SUBDIRS=( "$DATASET_DIR"/*/  )
if [[ ${#EXTRACTED_SUBDIRS[@]} -eq 1 && -d "${EXTRACTED_SUBDIRS[0]}" ]]; then
  SUBDIR="${EXTRACTED_SUBDIRS[0]}"
  log "Flattening single subdirectory: $SUBDIR → $DATASET_DIR"
  shopt -s dotglob
  mv "$SUBDIR"* "$DATASET_DIR/"
  rmdir "$SUBDIR"
  shopt -u dotglob
fi
 
rm -f "$ZIP_LOCAL"
log "Dataset download and extraction complete."


# ──────────────────────────────────────────────────────────────────────────────
# STEP 8 — Validate dataset
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 7 — Validating dataset: $DATASET_DIR"
[[ -d "$DATASET_DIR" ]] || die "Dataset directory '$DATASET_DIR' not found.
Expected LJSpeech layout:
  $DATASET_DIR/
  ├── metadata.csv   (id|transcript  OR  id|normalized|transcript)
  └── wav/
      ├── LJ001-0001.wav
      └── ..."

[[ -f "$DATASET_DIR/metadata.csv" ]] || die "metadata.csv missing in '$DATASET_DIR'"
[[ -d "$DATASET_DIR/wav" ]]          || die "'wav/' directory missing in '$DATASET_DIR'"

WAV_COUNT=$(find "$DATASET_DIR/wav" -name "*.wav" | wc -l)
log "Found $WAV_COUNT WAV file(s)"
[[ $WAV_COUNT -gt 0 ]] || die "No .wav files in '$DATASET_DIR/wav/'"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 9 — Preprocess
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 8 — Preprocessing dataset → $TRAINING_DIR"
mkdir -p "$TRAINING_DIR"
cd "$PIPER_DIR/src/python"

python3 -m piper_train.preprocess \
  --language        "$LANGUAGE" \
  --input-dir       "$DATASET_DIR" \
  --output-dir      "$TRAINING_DIR" \
  --dataset-format  ljspeech \
  --single-speaker \
  --sample-rate     "$SAMPLE_RATE" \
  --max-workers     "$MAX_WORKERS"

log "Preprocessing complete."

# ──────────────────────────────────────────────────────────────────────────────
# STEP 20 — Train
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 9 — Training  (epochs=$MAX_EPOCHS  quality=$QUALITY  batch=$BATCH_SIZE)"

TRAIN_CMD=(
  python3 -m piper_train
    --dataset-dir       "$TRAINING_DIR"
    --accelerator       "$ACCELERATOR"
    --devices           "$DEVICES"
    --batch-size        "$BATCH_SIZE"
    --validation-split  0.10
    --max_epochs        "$MAX_EPOCHS"
    --precision         "$PRECISION"
    --quality           "$QUALITY"
    --checkpoint-epochs "$CHECKPOINT_EPOCHS"
)

if [[ -n "$FINETUNE_CKPT" ]]; then
  if [[ -f "$FINETUNE_CKPT" ]]; then
    log "Fine-tuning from checkpoint: $FINETUNE_CKPT"
    TRAIN_CMD+=(--resume_from_checkpoint "$FINETUNE_CKPT")
  else
    warn "FINETUNE_CKPT='$FINETUNE_CKPT' not found — training from scratch"
  fi
fi

"${TRAIN_CMD[@]}"
log "Training complete!"

# ──────────────────────────────────────────────────────────────────────────────
# STEP 12 — Export to ONNX
# ──────────────────────────────────────────────────────────────────────────────
log "STEP 10 — Locating latest checkpoint"
CKPT_FILE=$(find "$TRAINING_DIR/lightning_logs" -name "*.ckpt" 2>/dev/null \
  | sort -V | tail -n 1 || true)
ONNX_FILE=""

if [[ -z "$CKPT_FILE" ]]; then
  warn "No .ckpt found — skipping ONNX export"
  UPLOAD_ONNX="false"
else
  log "Checkpoint: $CKPT_FILE"
  mkdir -p "$OUTPUT_DIR"
  ONNX_FILE="$OUTPUT_DIR/model.onnx"

  log "Exporting → $ONNX_FILE"
  python3 -m piper_train.export_onnx "$CKPT_FILE" "$ONNX_FILE"

  CONFIG_SRC="$TRAINING_DIR/config.json"
  if [[ -f "$CONFIG_SRC" ]]; then
    cp "$CONFIG_SRC" "${ONNX_FILE}.json"
    log "config.json copied → ${ONNX_FILE}.json"
  else
    warn "config.json not found — copy it manually next to model.onnx"
  fi
  log "ONNX export done."
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 12 — Upload to S3
# ──────────────────────────────────────────────────────────────────────────────
if [[ -n "$S3_BUCKET" ]]; then
  log "STEP 11 — Uploading to s3://$S3_BUCKET/$S3_PREFIX/"

  command -v aws &>/dev/null || die "'aws' CLI not found. Install it or unset S3_BUCKET."
  aws sts get-caller-identity &>/dev/null || \
    die "AWS credentials not configured (set AWS_ACCESS_KEY_ID/SECRET or attach IAM role)."

  TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
  S3_RUN="s3://$S3_BUCKET/$S3_PREFIX/$TIMESTAMP"

  # Checkpoints
  CKPT_DIR=$(find "$TRAINING_DIR/lightning_logs" -type d -name "checkpoints" 2>/dev/null \
    | head -n 1 || true)
  if [[ -n "$CKPT_DIR" ]]; then
    log "Uploading checkpoints → $S3_RUN/checkpoints/"
    aws s3 sync "$CKPT_DIR" "$S3_RUN/checkpoints/" \
      --exclude "*" --include "*.ckpt"
  else
    warn "No checkpoints/ directory found — skipping"
  fi

  # Training config
  if [[ -f "$TRAINING_DIR/config.json" ]]; then
    aws s3 cp "$TRAINING_DIR/config.json" "$S3_RUN/config.json"
    log "config.json uploaded"
  fi

  # ONNX model
  if [[ "$UPLOAD_ONNX" == "true" && -n "$ONNX_FILE" && -f "$ONNX_FILE" ]]; then
    log "Uploading ONNX → $S3_RUN/onnx/"
    aws s3 cp "$ONNX_FILE"          "$S3_RUN/onnx/model.onnx"
    [[ -f "${ONNX_FILE}.json" ]] && \
      aws s3 cp "${ONNX_FILE}.json" "$S3_RUN/onnx/model.onnx.json"
    log "ONNX upload done."
  fi

  log "All uploads complete → $S3_RUN/"
else
  log "STEP 11 — S3_BUCKET not set. Skipping upload."
  echo ""
  echo "  Manual upload commands:"
  echo "  aws s3 sync $TRAINING_DIR/lightning_logs/version_0/checkpoints/ \\"
  echo "    s3://<BUCKET>/piper-checkpoints/ --exclude '*' --include '*.ckpt'"
  if [[ -n "$ONNX_FILE" && -f "$ONNX_FILE" ]]; then
    echo "  aws s3 cp $ONNX_FILE       s3://<BUCKET>/piper-output/model.onnx"
    echo "  aws s3 cp ${ONNX_FILE}.json s3://<BUCKET>/piper-output/model.onnx.json"
  fi
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
log "═══════════════════════════════════════════════════════"
log " Piper training pipeline complete!"
log "═══════════════════════════════════════════════════════"
echo ""
echo "  Python           : $PYTHON_BIN"
echo "  Dataset          : $DATASET_DIR"
echo "  Training dir     : $TRAINING_DIR"
[[ -n "$ONNX_FILE" && -f "$ONNX_FILE" ]] && \
echo "  ONNX model       : $ONNX_FILE"
[[ -n "$S3_BUCKET" ]] && \
echo "  S3 destination   : s3://$S3_BUCKET/$S3_PREFIX/"
echo ""
echo "  Quick test:"
echo "    echo 'Hello world.' | piper -m $OUTPUT_DIR/model.onnx \\"
echo "      --output_file /tmp/test.wav && aplay /tmp/test.wav"
echo ""