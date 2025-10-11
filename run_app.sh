#!/bin/bash

# Launch script for FEniCS Streamlit Application
# This ensures the app runs in the correct conda environment

echo "======================================================================"
echo "  Starting FEniCS Simulation Assistant"
echo "======================================================================"

# Activate conda environment
source ~/miniconda3/bin/activate fenicsproject

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate fenicsproject environment"
    echo ""
    echo "Please create the environment first:"
    echo "  conda create -n fenicsproject -c conda-forge fenics matplotlib numpy scipy"
    echo "  conda activate fenicsproject"
    echo "  pip install streamlit python-dotenv openai pillow"
    exit 1
fi

echo "✓ FEniCS environment activated"

# Check if fenics is installed
if ! python -c "import fenics" 2>/dev/null; then
    echo "❌ FEniCS is not installed in this environment"
    exit 1
fi

echo "✓ FEniCS is available"

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing Streamlit..."
    pip install streamlit python-dotenv openai pillow
fi

echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================================"
echo ""

# Run streamlit
streamlit run app.py

