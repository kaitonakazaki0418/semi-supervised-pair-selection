#!/bin/bash
set -e

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

if ! command -v $PYTHON &> /dev/null; then
    echo "Python 3 not found."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

python pair_ssl.py "$@"
python visualize.py
