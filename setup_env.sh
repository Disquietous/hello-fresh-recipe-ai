#!/bin/bash

echo "Setting up virtual environment for HelloFresh Recipe AI..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo ""
echo "Virtual environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To deactivate, run: deactivate"