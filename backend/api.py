"""
FastAPI service that exposes FEniCS simulations via HTTP.

Endpoints:
- POST /simulate
  { "function_name": "heat_transfer.solve_heat_equation", "parameters": { ... } }
  Returns JSON with basic metrics and base64-encoded images (PNG/GIF) when available.
"""

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
import os
import sys

# Ensure imports work when running as a module
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import heat_transfer
import solid_mechanics


class SimulationRequest(BaseModel):
    function_name: str
    parameters: Dict[str, Any] = {}


app = FastAPI(title="FEniCS Simulation API")


def encode_plot_buffer(buf: BytesIO) -> str:
    buf.seek(0)
    data = buf.read()
    return base64.b64encode(data).decode("utf-8")


@app.post("/simulate")
def simulate(req: SimulationRequest) -> Dict[str, Any]:
    try:
        module_name, func_name = req.function_name.rsplit(".", 1)
        if module_name == "heat_transfer":
            module = heat_transfer
        elif module_name == "solid_mechanics":
            module = solid_mechanics
        else:
            raise HTTPException(status_code=400, detail=f"Unknown module: {module_name}")

        func = getattr(module, func_name, None)
        if func is None:
            raise HTTPException(status_code=400, detail=f"Unknown function: {req.function_name}")

        # Execute simulation
        result_data = func(**req.parameters)

        # Minimal metrics payload (avoid sending huge data)
        payload: Dict[str, Any] = {"module": module_name, "function": func_name}
        plots: Dict[str, str] = {}
        plots_interactive: Dict[str, Any] = {}
        output_dir: Optional[str] = None

        # Prepare title and output subfolder using same convention as UI
        title_map = {
            'solve_heat_equation': 'Transient Heat Diffusion',
            'solve_heat_equation_3d': '3D Heat Diffusion',
            'solve_steady_state_heat': 'Steady State Heat Transfer',
            'solve_cantilever_beam': 'Cantilever Beam',
            'solve_plate_with_hole': 'Plate with Tension',
            'solve_compression_test': 'Compression Test'
        }
        plot_title = title_map.get(func_name, func_name.replace('_', ' ').title())

        if module_name == "heat_transfer":
            if func_name in ("solve_heat_equation", "solve_heat_equation_3d"):
                u, mesh, temps, times, solutions = result_data
                payload["num_cells"] = mesh.num_cells()
                payload["num_vertices"] = mesh.num_vertices()
                payload["time_steps"] = len(times)
                # Save plots to disk using backend helpers and read primary PNG back
                if func_name == 'solve_heat_equation':
                    heat_transfer.plot_results(u, mesh, title=plot_title, solutions=solutions, times=times)
                    subfolder = plot_title.lower().replace(' ', '_')
                    output_dir = os.path.join("heat_transfer_results", subfolder)
                    png_path = os.path.join(output_dir, "temperature_plot.png")
                    if os.path.exists(png_path):
                        with open(png_path, 'rb') as f:
                            plots['temperature'] = base64.b64encode(f.read()).decode('utf-8')
                elif func_name == 'solve_heat_equation_3d':
                    heat_transfer.plot_results_3d(u, mesh, title=plot_title, solutions=solutions, times=times)
                    subfolder = plot_title.lower().replace(' ', '_')
                    output_dir = os.path.join("heat_transfer_results", subfolder)
                    png_path = os.path.join(output_dir, "temperature_slices.png")
                    if os.path.exists(png_path):
                        with open(png_path, 'rb') as f:
                            plots['3d_slices'] = base64.b64encode(f.read()).decode('utf-8')
                    # Build interactive 3D scatter using plotly and return as JSON
                    try:
                        import numpy as np
                        import plotly.graph_objects as go
                        temp_values = u.compute_vertex_values(mesh)
                        coords = mesh.coordinates()
                        n_points = len(temp_values)
                        sample_rate = max(1, n_points // 5000)
                        sampled_coords = coords[::sample_rate]
                        sampled_temps = temp_values[::sample_rate]
                        fig_3d = go.Figure(data=[go.Scatter3d(
                            x=sampled_coords[:, 0],
                            y=sampled_coords[:, 1],
                            z=sampled_coords[:, 2],
                            mode='markers',
                            marker=dict(size=3, color=sampled_temps, colorscale='Hot',
                                        colorbar=dict(title="Temperature"), showscale=True),
                            text=[f'T={t:.2f}' for t in sampled_temps],
                            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>'
                        )])
                        fig_3d.update_layout(
                            title='Interactive 3D Temperature Distribution',
                            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
                            width=800, height=700, margin=dict(l=0, r=0, b=0, t=40)
                        )
                        plots_interactive['3d_interactive'] = fig_3d.to_json()
                    except Exception:
                        pass
            else:
                u, mesh = result_data
                payload["num_cells"] = mesh.num_cells()
                # Compute a simple metric
                try:
                    max_temp = max(u.compute_vertex_values(mesh))
                    payload["max_temperature"] = float(max_temp)
                except Exception:
                    pass
                # Save steady-state plot
                heat_transfer.plot_results(u, mesh, title=plot_title)
                subfolder = plot_title.lower().replace(' ', '_')
                output_dir = os.path.join("heat_transfer_results", subfolder)
                png_path = os.path.join(output_dir, "temperature_plot.png")
                if os.path.exists(png_path):
                    with open(png_path, 'rb') as f:
                        plots['temperature'] = base64.b64encode(f.read()).decode('utf-8')
        else:
            u, stress, mesh = result_data
            payload["num_cells"] = mesh.num_cells()
            solid_mechanics.plot_mechanics_results(u, stress, mesh, title=plot_title)
            subfolder = plot_title.lower().replace(' ', '_')
            output_dir = os.path.join("solid_mechanics_results", subfolder)
            png_path = os.path.join(output_dir, "results_plot.png")
            if os.path.exists(png_path):
                with open(png_path, 'rb') as f:
                    plots['results'] = base64.b64encode(f.read()).decode('utf-8')

        return {"success": True, "result": payload, "plots": plots, "plots_interactive": plots_interactive, "output_dir": output_dir}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


