#!/usr/bin/env bash
# Creates an isolated Python 3.10 environment in .venv, installs dependencies,
# and downloads Slippi Dolphin (the GameCube emulator) for macOS or Linux if it is missing.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python 3.10
  uv pip install --python .venv/bin/python -r requirements.txt
elif command -v python3.10 >/dev/null 2>&1; then
  python3.10 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
else
  echo "Python 3.10 not found - install uv (https://docs.astral.sh/uv/) or python3.10" >&2
  exit 1
fi

RELEASE=https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4
if [ "$(uname)" = "Darwin" ]; then
  APP="dolphin/Slippi Dolphin.app"
  if [ ! -d "$APP" ]; then
    echo "Downloading Slippi Dolphin 3.6.4 for macOS..."
    mkdir -p dolphin
    curl -L -o dolphin/FM-Slippi-Mac.dmg "$RELEASE/FM-Slippi-3.6.4-Mac.dmg"
    hdiutil attach -nobrowse -readonly -mountpoint dolphin/mnt dolphin/FM-Slippi-Mac.dmg >/dev/null
    cp -R "dolphin/mnt/Slippi Dolphin.app" dolphin/
    hdiutil detach dolphin/mnt >/dev/null
  fi
else
  APP="dolphin/Slippi_Online-x86_64.AppImage"
  if [ ! -f "$APP" ]; then
    echo "Downloading Slippi Dolphin 3.6.4 for Linux..."
    mkdir -p dolphin
    curl -L -o "$APP" "$RELEASE/Slippi_Online-x86_64.AppImage"
    chmod +x "$APP"
  fi
  # Headless build (no window, fast-forward) used for training, from libmelee's author.
  HEADLESS="dolphin/Slippi_Online-x86_64-ExiAI.AppImage"
  if [ ! -f "$HEADLESS" ]; then
    echo "Downloading headless ExiAI Dolphin for training..."
    curl -L -o "$HEADLESS" \
      https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.2.0/Slippi_Online-x86_64-ExiAI.AppImage
    chmod +x "$HEADLESS"
  fi
  # AppImages need FUSE 2 (Ubuntu 22.04+: sudo apt install libfuse2)
fi

if [ ! -f "Super Smash Bros Melee v1.02.iso" ]; then
  echo "WARNING: put your NTSC v1.02 Melee ISO here as 'Super Smash Bros Melee v1.02.iso'" >&2
fi

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then check the game:  python test_env.py"
