#!/bin/bash
# Activate Python 3.12 virtual environment with qiskit support
#
# Usage:
#   source activate_qiskit.sh
#   OR
#   . activate_qiskit.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv_qiskit"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Creating new environment..."
    /opt/homebrew/bin/python3.12 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install qiskit qiskit-aer numpy matplotlib pandas pytest -q
    echo "✅ Environment created and activated"
else
    source "$VENV_DIR/bin/activate"
    echo "✅ Qiskit environment activated (Python $(python --version | cut -d' ' -f2))"
fi

echo ""
echo "Available commands:"
echo "  python          - Python 3.12.12 with qiskit"
echo "  pytest          - Run tests"
echo "  jupyter lab     - Start Jupyter (if installed)"
echo ""
echo "To run tests:"
echo "  pytest tests/test_unified.py -v"
echo ""
echo "To deactivate:"
echo "  deactivate"
