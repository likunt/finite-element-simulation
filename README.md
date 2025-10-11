# FEniCS Simulation Assistant

An intelligent web interface that uses AI to help you run finite element simulations. Simply describe your problem in natural language, and the app will select and run the appropriate FEniCS simulation.

## 🚀 Quick Start

### Option 1: Use the Launch Script (Recommended)

```bash
# Navigate to the project directory
cd finite-element-simulation

# Run the launch script
./run_app.sh
```

The script will:
- ✅ Activate the correct conda environment (`fenicsproject`)
- ✅ Check all dependencies (FEniCS, Streamlit, etc.)
- ✅ Install missing dependencies automatically
- ✅ Start the Streamlit app
- 🌐 Open in your browser at http://localhost:8501

### Option 2: Manual Launch

```bash
# Activate the FEniCS environment
conda activate fenicsproject

# Navigate to project directory
cd finite-element-simulation

# Run the app
streamlit run app.py
```

## 📚 Features

### Available Simulations

1. **Heat Transfer**
   - Transient heat diffusion
   - Steady-state heat equation
   - 3D heat transfer

2. **Solid Mechanics**
   - Cantilever beam analysis
   - Plate under tension
   - Compression test

### AI-Powered Simulation Selection

- Describe your problem in natural language
- AI selects the appropriate simulation function


### Example Prompts

- "Simulate heat diffusion in a square plate with hot center"
- "Simulate 3D heat transfer in a cube with hot center"
- "Calculate thermal stresses in a heated aluminum plate"
- "Find steady-state temperature in a plate with heat source"
- "Simulate compression of a rubber cylinder"

## 🖥️ Interface

### Sidebar
- **Model Selection**: Choose GPT-4, GPT-4-turbo, or GPT-3.5-turbo
- **API Key Status**: Shows if API key is loaded
- **Example Problems**: Quick-start templates

### Main Area
- **Problem Description**: Natural language input
- **AI Analysis**: Shows detected problem type
- **Results**: Visualizations, metrics, and downloadable files
- **History**: Track your recent simulations

## 📁 Project Structure

```
finite-element-simulation/
├── app.py                          # Streamlit UI application
├── run_app.sh                      # Launch script
├── .env.example                    # Example environment file
├── README.md                       # This file
└── backend/
   ├── heat_transfer.py           # Heat transfer simulations
   ├── solid_mechanics.py         # Solid mechanics simulations
   ├── requirements.txt           # Python dependencies
   ├── activate_fenics.sh         # Helper script to activate environment
   └── *_results/                 # Output directories
```

