#!/bin/bash
# QFDP Multi-Asset Setup Script
# Installs dependencies and verifies environment

set -e

echo "=========================================="
echo "QFDP Multi-Asset Setup"
echo "=========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "[2/5] Creating virtual environment..."
    python3 -m venv .venv
else
    echo ""
    echo "[2/5] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/unit/test_factor_model.py -v"
echo ""
