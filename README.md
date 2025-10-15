# FEniCS Simulation Assistant

An intelligent web interface that uses AI to help you run finite element simulations. Simply describe your problem in natural language, and the app will select and run the appropriate FEniCS simulation.

## 🚀 Quick Start

Follow these steps to run simulations via a Dockerized FEniCS API and view results in the Streamlit UI.

### 1) Prerequisites
- Docker Desktop installed and running
- Python 3.10+ with `pip`

### 2) Start the FEniCS Simulation API (Docker)
```bash
cd finite-element-simulation

# Build the API image 
docker build -t fenics-sim-api .

# Run the API on port 8000
docker run --rm -p 8000:8000 fenics-sim-api
```

Verify the API is healthy in a second terminal:
```bash
curl http://localhost:8000/healthz
```
You should see: `{ "status": "ok" }`.

Notes for Apple silicon: if you hit platform issues, try:
```bash
docker buildx build --platform linux/arm64 -t fenics-sim-api .
```

### 3) Configure the Streamlit UI to use the API
Create a `.env` file in the project root (or export the variable) with:
```
FENICS_API_URL=http://localhost:8000
```

### 4) Run the Streamlit UI locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the app in your browser (Streamlit prints the URL, typically `http://localhost:8501`).

### 5) (Optional) Deploy the API remotely
You can deploy the Docker image to any container host (Render, Railway, Fly.io, AWS, etc.). Expose port 8000 and note the public URL, e.g. `https://your-api.example.com`.

### 6) (Optional) Run Streamlit on Streamlit Cloud
Because Streamlit Cloud cannot install FEniCS, it must call your remote API.
- Push this repo to GitHub
- Create an app on Streamlit Cloud pointing to `app.py`
- In the app’s Settings → Secrets, add:
```
FENICS_API_URL=https://your-api.example.com
```
The UI will route simulations to your API and display returned metrics/plots.

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
├── Dockerfile                      # Docker for FEniCS FastAPI service
├── run_app.sh                      # Launch script
├── .env.example                    # Example environment file
├── README.md                       # This file
└── backend/
   ├── heat_transfer.py           # Heat transfer simulations
   ├── solid_mechanics.py         # Solid mechanics simulations
   ├── api.py                     # FastAPI endpoints for simulations
   ├── requirements.txt           # Python dependencies
   ├── activate_fenics.sh         # Helper script to activate environment
   └── *_results/                 # Output directories
```

## 🐳 Deploy FEniCS Simulations via Docker API

Streamlit Cloud cannot install FEniCS. Run simulations in a Dockerized FastAPI and call it from the UI.

### Build and run the API locally

```bash
docker build -t fenics-sim-api .
docker run --rm -p 8000:8000 fenics-sim-api
# Health check
curl http://localhost:8000/healthz
```

### Configure the UI to use the API

Set `FENICS_API_URL` for the Streamlit app (e.g., in `.env`):

```
FENICS_API_URL=http://localhost:8000
```

Then run the UI:

```bash
pip install -r requirements.txt
streamlit run app.py
```

When `FENICS_API_URL` is set, the app offloads simulations to the API and displays returned metrics.

