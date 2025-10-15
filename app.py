"""
FEniCS Simulation Assistant - Streamlit UI
===========================================
An intelligent interface that uses LLMs to understand simulation requirements
and automatically select and configure appropriate FEniCS functions.
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
import importlib.util
import requests

# Optional dotenv support (no-op if package is missing)
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="FEniCS Simulation Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔬 FEniCS Simulation Assistant</div>', unsafe_allow_html=True)

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None


# ============================================================================
# HELPER FUNCTIONS (Must be defined before UI code)
# ============================================================================

def analyze_problem_with_llm(description, api_key, model):
    """
    Use LLMs to analyze the problem description and decide which FEniCS function to use
    """
    
    # Available functions catalog
    functions_catalog = {
        "heat_transfer.solve_heat_equation": {
            "description": "Transient heat diffusion with initial conditions (2D)",
            "keywords": ["transient", "heat", "diffusion", "time-dependent", "cooling", "heating", "2d", "plate", "square"],
            "parameters": ["nx", "ny", "T_final", "num_steps", "alpha"]
        },
        "heat_transfer.solve_heat_equation_3d": {
            "description": "3D transient heat diffusion in a cube",
            "keywords": ["3d", "three-dimensional", "cube", "volume", "transient", "heat", "diffusion"],
            "parameters": ["nx", "ny", "nz", "T_final", "num_steps", "alpha"]
        },
        "heat_transfer.solve_steady_state_heat": {
            "description": "Steady-state heat equation with heat source",
            "keywords": ["steady", "equilibrium", "heat source", "constant temperature"],
            "parameters": ["nx", "ny", "boundary_value", "thermal_conductivity"]
        },
        "solid_mechanics.solve_cantilever_beam": {
            "description": "Cantilever beam analysis with fixed end",
            "keywords": ["beam", "cantilever", "bending", "deflection", "fixed"],
            "parameters": []
        },
        "solid_mechanics.solve_plate_with_hole": {
            "description": "Plate under tension (stress concentration)",
            "keywords": ["plate", "tension", "stress", "hole", "concentration"],
            "parameters": []
        },
        "solid_mechanics.solve_compression_test": {
            "description": "Uniaxial compression test",
            "keywords": ["compression", "squeeze", "press", "cylinder"],
            "parameters": []
        }
    }
    
    # Use OpenAI with gpt-oss-20b
    if api_key:
        return analyze_with_openai(description, api_key, model, functions_catalog)
    else:
        st.warning("⚠️ No API key provided. Using fallback rule-based matching.")
        # Fallback to rule-based matching
        description_lower = description.lower()
        
        # Score each function
        scores = {}
        for func_name, func_info in functions_catalog.items():
            score = sum(1 for keyword in func_info['keywords'] 
                       if keyword in description_lower)
            scores[func_name] = score
        
        # Select best match
        best_function = max(scores, key=scores.get)
        func_info = functions_catalog[best_function]
        
        # Determine parameters based on description
        parameters = {}
        if "high resolution" in description_lower or "fine mesh" in description_lower:
            parameters['nx'] = 80
            parameters['ny'] = 80
        
        return {
            'problem_type': func_info['description'],
            'function_name': best_function,
            'parameters': parameters,
            'reasoning': f"Matched keywords: {[k for k in func_info['keywords'] if k in description_lower]}"
        }


def analyze_with_openai(description, api_key, model, functions_catalog):
    """Analyze using OpenAI API (v1.0+)"""
    try:
        from openai import OpenAI
        
        # Initialize client with API key
        client = OpenAI(api_key=api_key)
        
        prompt = f"""Given this simulation problem description:
"{description}"

Available FEniCS functions:
{json.dumps(functions_catalog, indent=2)}

Return a JSON with:
- problem_type: brief description
- function_name: best matching function
- parameters: dict of any custom parameters (can be empty {{}})
- reasoning: why you chose this

Response (JSON only):"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        # Ensure parameters key exists
        if 'parameters' not in result:
            result['parameters'] = {}
        return result
        
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        st.warning("Falling back to rule-based matching...")
        # Fallback to simple matching
        description_lower = description.lower()
        scores = {}
        for func_name, func_info in functions_catalog.items():
            score = sum(1 for keyword in func_info['keywords'] 
                       if keyword in description_lower)
            scores[func_name] = score
        
        best_function = max(scores, key=scores.get)
        func_info = functions_catalog[best_function]
        
        return {
            'problem_type': func_info['description'],
            'function_name': best_function,
            'parameters': {},
            'reasoning': f"Fallback matching (API error)"
        }


def generate_plots_from_data(result_data, module_name, func_name):
    """
    Generate plots directly from simulation data without loading from disk
    Returns dict with plot images as BytesIO objects
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from io import BytesIO
    import numpy as np
    
    plots = {}
    
    try:
        if module_name == "heat_transfer":
            if func_name == 'solve_heat_equation':
                u, mesh, temps, times, solutions = result_data
                
                # Generate temperature plot
                fig = plt.figure(figsize=(10, 8))
                try:
                    from fenics import plot
                    c = plot(u, title='Temperature Distribution')
                    plt.colorbar(c)
                    plt.xlabel('x')
                    plt.ylabel('y')
                except:
                    # Fallback: use triangulation
                    coords = mesh.coordinates()
                    cells = mesh.cells()
                    temp_values = u.compute_vertex_values(mesh)
                    plt.tricontourf(coords[:, 0], coords[:, 1], cells, temp_values, cmap='hot')
                    plt.colorbar(label='Temperature')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('Temperature Distribution')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plots['temperature'] = buf
                plt.close()
                
                # Generate animation if we have solutions
                if solutions and len(solutions) > 1:
                    from matplotlib.animation import FuncAnimation, PillowWriter
                    
                    coords = mesh.coordinates()
                    cells = mesh.cells()
                    x, y = coords[:, 0], coords[:, 1]
                    
                    # Get temperature range
                    all_temps = [sol.compute_vertex_values(mesh) for sol in solutions]
                    vmin, vmax = min(np.min(t) for t in all_temps), max(np.max(t) for t in all_temps)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    def update(frame):
                        ax.clear()
                        temp_values = solutions[frame].compute_vertex_values(mesh)
                        ax.tricontourf(x, y, cells, temp_values, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_title(f'Temperature Evolution - Time: {times[frame]:.3f} s')
                        ax.set_aspect('equal')
                    
                    anim = FuncAnimation(fig, update, frames=len(solutions), interval=200, repeat=True)
                    
                    buf_anim = BytesIO()
                    writer = PillowWriter(fps=5)
                    anim.save(buf_anim, writer=writer, format='gif')
                    buf_anim.seek(0)
                    plots['animation'] = buf_anim
                    plt.close()
            
            elif func_name == 'solve_heat_equation_3d':
                u, mesh, temps, times, solutions = result_data
                
                # Generate 3D slice plots
                fig = plt.figure(figsize=(15, 5))
                
                temp_values = u.compute_vertex_values(mesh)
                coords = mesh.coordinates()
                
                # Create slice plots at different z-levels
                z_slices = [0.25, 0.5, 0.75]
                
                for idx, z_level in enumerate(z_slices):
                    ax = fig.add_subplot(1, 3, idx + 1)
                    
                    # Find nodes close to this z-level
                    tolerance = 0.05
                    slice_mask = np.abs(coords[:, 2] - z_level) < tolerance
                    
                    if np.any(slice_mask):
                        slice_coords = coords[slice_mask]
                        slice_temps = temp_values[slice_mask]
                        
                        scatter = ax.scatter(slice_coords[:, 0], slice_coords[:, 1], 
                                           c=slice_temps, cmap='hot', s=20)
                        plt.colorbar(scatter, ax=ax)
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        ax.set_title(f'Temperature at z={z_level:.2f}')
                        ax.set_aspect('equal')
                
                plt.tight_layout()
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plots['3d_slices'] = buf
                plt.close()
                
                # Create interactive 3D plot with Plotly
                try:
                    import plotly.graph_objects as go
                    
                    # Sample the mesh for visualization (use every nth point for performance)
                    n_points = len(temp_values)
                    sample_rate = max(1, n_points // 5000)  # Limit to ~5000 points
                    
                    sampled_coords = coords[::sample_rate]
                    sampled_temps = temp_values[::sample_rate]
                    
                    # Create 3D scatter plot
                    fig_3d = go.Figure(data=[go.Scatter3d(
                        x=sampled_coords[:, 0],
                        y=sampled_coords[:, 1],
                        z=sampled_coords[:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=sampled_temps,
                            colorscale='Hot',
                            colorbar=dict(title="Temperature"),
                            showscale=True
                        ),
                        text=[f'T={t:.2f}' for t in sampled_temps],
                        hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>%{text}<extra></extra>'
                    )])
                    
                    fig_3d.update_layout(
                        title='Interactive 3D Temperature Distribution',
                        scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z',
                            aspectmode='cube'
                        ),
                        width=800,
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    
                    # Store as plotly figure (not BytesIO)
                    plots['3d_interactive'] = fig_3d
                    
                except Exception as e:
                    print(f"Could not create interactive 3D plot: {e}")
                    
            else:  # steady_state_heat
                u, mesh = result_data
                
                fig = plt.figure(figsize=(10, 8))
                try:
                    from fenics import plot
                    c = plot(u, title='Steady State Temperature')
                    plt.colorbar(c)
                    plt.xlabel('x')
                    plt.ylabel('y')
                except:
                    coords = mesh.coordinates()
                    cells = mesh.cells()
                    temp_values = u.compute_vertex_values(mesh)
                    plt.tricontourf(coords[:, 0], coords[:, 1], cells, temp_values, cmap='hot')
                    plt.colorbar(label='Temperature')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title('Steady State Temperature')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                plots['temperature'] = buf
                plt.close()
                
        else:  # solid_mechanics
            u, stress, mesh = result_data
            
            # Generate results plot
            fig = plt.figure(figsize=(15, 5))
            
            # Displacement magnitude
            plt.subplot(1, 2, 1)
            try:
                from fenics import project, FunctionSpace, sqrt, dot
                u_magnitude = project(sqrt(dot(u, u)), FunctionSpace(mesh, 'P', 1))
                from fenics import plot
                c1 = plot(u_magnitude, title='Displacement Magnitude')
                plt.colorbar(c1)
            except:
                pass
            
            # Von Mises stress
            plt.subplot(1, 2, 2)
            try:
                from fenics import plot
                c2 = plot(stress, title='von Mises Stress')
                plt.colorbar(c2)
            except:
                pass
            
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['results'] = buf
            plt.close()
    
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return plots


def run_simulation(function_name, parameters, mesh_resolution, time_steps, simulation_time, 
                   alpha=None, thermal_conductivity=None):
    """
    Execute the selected FEniCS simulation function with custom parameters
    
    Args:
        function_name: Name of the FEniCS function to execute
        parameters: Dict of parameters from LLM analysis
        mesh_resolution: Mesh resolution (nx, ny)
        time_steps: Number of time steps (for transient)
        simulation_time: Total simulation time (for transient)
        alpha: Thermal diffusivity (for transient heat)
        thermal_conductivity: Thermal conductivity (for steady-state heat)
    """
    import time
    start_time = time.time()
    
    try:
        # If configured, use remote API (FEniCS inside Docker)
        api_url = "https://finite-element-simulation-purple-dew-7118.fly.dev" #os.getenv("FENICS_API_URL")
        if api_url:
            # Prepare parameters similar to local run
            cleaned_params = {}
            for key, value in parameters.items():
                if isinstance(value, str):
                    try:
                        cleaned_params[key] = float(value) if '.' in value else int(value)
                    except (ValueError, AttributeError):
                        continue
                elif isinstance(value, (int, float)):
                    cleaned_params[key] = value

            module_name, func_name = function_name.rsplit('.', 1)
            if func_name in ('solve_heat_equation', 'solve_steady_state_heat'):
                cleaned_params.setdefault('nx', mesh_resolution)
                cleaned_params.setdefault('ny', mesh_resolution)
            if func_name == 'solve_heat_equation':
                cleaned_params.setdefault('T_final', simulation_time)
                cleaned_params.setdefault('num_steps', time_steps)
                if alpha is not None:
                    cleaned_params['alpha'] = alpha
            if func_name == 'solve_steady_state_heat' and thermal_conductivity is not None:
                cleaned_params['thermal_conductivity'] = thermal_conductivity

            payload = {"function_name": function_name, "parameters": cleaned_params}
            resp = requests.post(f"{api_url.rstrip('/')}/simulate", json=payload, timeout=300)
            if resp.status_code != 200:
                return {"success": False, "error": f"API error: {resp.status_code} {resp.text}"}
            data = resp.json()
            if not data.get('success'):
                return {"success": False, "error": data}

            # Decode plots (base64 → BytesIO)
            plots = {}
            try:
                from io import BytesIO
                import base64
                for name, b64 in (data.get('plots') or {}).items():
                    try:
                        plots[name] = BytesIO(base64.b64decode(b64))
                    except Exception:
                        pass
            except Exception:
                pass

            # Interactive plots (Plotly JSON → Figure)
            if data.get('plots_interactive'):
                try:
                    import plotly.io as pio
                    for name, fig_json in data['plots_interactive'].items():
                        try:
                            plots[name] = pio.from_json(fig_json)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Minimal success payload for UI
            result_metrics = data.get('result', {})
            return {
                'success': True,
                'solve_time': 0.0,
                'num_elements': result_metrics.get('num_cells', 'N/A'),
                'metrics': {
                    'Mesh Cells': result_metrics.get('num_cells', 'N/A'),
                    'Mesh Vertices': result_metrics.get('num_vertices', 'N/A'),
                    'Time Steps': result_metrics.get('time_steps', 'N/A'),
                    'Max Temperature': result_metrics.get('max_temperature', 'N/A'),
                },
                'plots': plots,
                'output_dir': data.get('output_dir'),
            }

        # Otherwise run locally (requires fenics installed)
        # Import the appropriate module
        module_name, func_name = function_name.rsplit('.', 1)
        
        # Add backend directory to path
        backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
        if os.path.exists(backend_dir):
            sys.path.insert(0, backend_dir)
        else:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import module
        if module_name == "heat_transfer":
            import heat_transfer as module
        elif module_name == "solid_mechanics":
            import solid_mechanics as module
        else:
            return {'success': False, 'error': f'Unknown module: {module_name}'}
        
        # Get function
        func = getattr(module, func_name)
        
        # Get plotting function
        if module_name == "heat_transfer":
            plot_func = getattr(module, 'plot_results')
        else:  # solid_mechanics
            plot_func = getattr(module, 'plot_mechanics_results')
        
        # Clean and validate parameters - remove invalid values
        cleaned_params = {}
        for key, value in parameters.items():
            # Skip if value is a non-numeric string (like "value", "description", etc.)
            if isinstance(value, str):
                # Try to convert to number
                try:
                    if '.' in value:
                        cleaned_params[key] = float(value)
                    else:
                        cleaned_params[key] = int(value)
                except (ValueError, AttributeError):
                    # Skip invalid string values
                    continue
            elif isinstance(value, (int, float)):
                cleaned_params[key] = value
        
        # Use cleaned parameters
        parameters = cleaned_params
        
        # Add parameters from UI sliders (these are guaranteed to be correct)
        if func_name == 'solve_heat_equation' or func_name == 'solve_steady_state_heat':
            parameters.setdefault('nx', mesh_resolution)
            parameters.setdefault('ny', mesh_resolution)
        
        if func_name == 'solve_heat_equation':
            parameters.setdefault('T_final', simulation_time)
            parameters.setdefault('num_steps', time_steps)
            # Add thermal diffusivity if provided
            if alpha is not None:
                parameters['alpha'] = alpha
        
        if func_name == 'solve_steady_state_heat':
            # Add thermal conductivity if provided
            if thermal_conductivity is not None:
                parameters['thermal_conductivity'] = thermal_conductivity
        
        # Ensure correct types for critical parameters
        if 'nx' in parameters:
            parameters['nx'] = int(parameters['nx'])
        if 'ny' in parameters:
            parameters['ny'] = int(parameters['ny'])
        if 'num_steps' in parameters:
            parameters['num_steps'] = int(parameters['num_steps'])
        if 'T_final' in parameters:
            parameters['T_final'] = float(parameters['T_final'])
        if 'alpha' in parameters:
            parameters['alpha'] = float(parameters['alpha'])
        if 'thermal_conductivity' in parameters:
            parameters['thermal_conductivity'] = float(parameters['thermal_conductivity'])
        
        # Call function with validated parameters
        print("Running simulation...")
        result_data = func(**parameters)
        print("Simulation completed!")
        
        # Generate a title for the plot based on function name
        title_map = {
            'solve_heat_equation': 'Transient Heat Diffusion',
            'solve_heat_equation_3d': '3D Heat Diffusion',
            'solve_steady_state_heat': 'Steady State Heat Transfer',
            'solve_cantilever_beam': 'Cantilever Beam',
            'solve_plate_with_hole': 'Plate with Tension',
            'solve_compression_test': 'Compression Test'
        }
        plot_title = title_map.get(func_name, func_name.replace('_', ' ').title())
        
        # Generate plots directly from simulation data (in memory)
        print("Generating plots from simulation data...")
        direct_plots = generate_plots_from_data(result_data, module_name, func_name)
        print(f"Generated {len(direct_plots)} plot(s) in memory")
        
        # Also call the backend plotting function to save files to disk
        print("Saving results to disk...")
        if module_name == "heat_transfer":
            if func_name == 'solve_heat_equation':
                u, mesh, temps, times, solutions = result_data
                plot_func(u, mesh, title=plot_title, solutions=solutions, times=times)
            elif func_name == 'solve_heat_equation_3d':
                u, mesh, temps, times, solutions = result_data
                # Import 3D plotting function
                plot_results_3d = getattr(module, 'plot_results_3d')
                plot_results_3d(u, mesh, title=plot_title, solutions=solutions, times=times)
            else:
                u, mesh = result_data
                plot_func(u, mesh, title=plot_title)
        else:  # solid_mechanics
            u, stress, mesh = result_data
            plot_func(u, stress, mesh, title=plot_title)
        print("Results saved to disk!")
        
        # Process results
        solve_time = time.time() - start_time
        
        # Find output directory
        if 'heat' in func_name.lower():
            result_dir_name = 'heat_transfer_results'
        else:
            result_dir_name = 'solid_mechanics_results'
        
        subfolder_name = plot_title.lower().replace(' ', '_')
        output_dir = os.path.join(backend_dir, result_dir_name, subfolder_name)
        
        # Extract metrics from the unpacked results
        metrics = {}
        if module_name == "heat_transfer":
            if func_name == 'solve_heat_equation' or func_name == 'solve_heat_equation_3d':
                # Already unpacked: u, mesh, temps, times, solutions
                metrics['Final Max Temperature'] = f"{max(temps[-1]) if temps else 0:.4f}"
                metrics['Mesh Cells'] = mesh.num_cells()
                metrics['Time Steps'] = len(times)
                if func_name == 'solve_heat_equation_3d':
                    metrics['Dimension'] = '3D'
                    metrics['Mesh Vertices'] = mesh.num_vertices()
            else:
                # Already unpacked: u, mesh
                metrics['Mesh Cells'] = mesh.num_cells()
                max_temp = max(u.compute_vertex_values(mesh))
                metrics['Max Temperature'] = f"{max_temp:.4f}"
        else:  # solid_mechanics
            # Already unpacked: u, stress, mesh
            metrics['Mesh Cells'] = mesh.num_cells()
            u_vals = u.compute_vertex_values(mesh).reshape((mesh.num_vertices(), 2))
            max_displacement = max(u_vals[:, 0]**2 + u_vals[:, 1]**2)**0.5
            metrics['Max Displacement'] = f"{max_displacement:.6e} m"
            max_stress = max(stress.compute_vertex_values(mesh))
            metrics['Max von Mises Stress'] = f"{max_stress:.2e} Pa"
        
        return {
            'success': True,
            'solve_time': solve_time,
            'num_elements': metrics.get('Mesh Cells', 'N/A'),
            'metrics': metrics,
            'plots': direct_plots,  # BytesIO plot objects for direct display
            'output_dir': output_dir
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'solve_time': time.time() - start_time
        }


# ============================================================================
# UI CODE STARTS HERE
# ============================================================================

# Load API key from session or environment
api_key = st.session_state.get('api_key') or os.getenv("OPENAI_API_KEY")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_options = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    model = st.selectbox("Select Model", model_options, index=0)
    st.info(f"🤖 Using: {model}")
    
    # API Key status
    if api_key:
        st.success("✅ API Key loaded from .env")
    else:
        st.warning("⚠️ No API key found in .env file")
        st.info("Create a `.env` file with:\n`OPENAI_API_KEY=your-key-here`")
    
    # API Key input (stored only in session)
    st.write("**API Key**")
    api_key_input = st.text_input(
        "Enter OpenAI API Key",
        value=st.session_state.get('api_key', ''),
        type="password",
        placeholder="sk-...",
        help="Stored for this session only (not saved to disk)."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
        api_key = api_key_input
    
    st.divider()
    
    # Quick examples
    st.header("📚 Example Problems")
    examples = {
        "Heat Diffusion": "Simulate heat diffusion in a square plate with hot center",
        "3D Heat Transfer": "Simulate 3D heat transfer in a cube with hot center",
        "Cantilever Beam": "Analyze a steel cantilever beam fixed at one end with a load",
        "Steady Heat": "Find steady-state temperature in a plate with heat source",
        "Compression Test": "Simulate compression of a rubber cylinder"
    }
    
    for name, desc in examples.items():
        if st.button(name, use_container_width=True):
            st.session_state.example_text = desc

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Problem Description")
    
    # Get example text if button was clicked
    default_text = st.session_state.get('example_text', '')
    
    problem_description = st.text_area(
        "Describe your simulation problem:",
        value=default_text,
        height=150,
        placeholder="Example: I want to simulate heat transfer in a 2D square plate..."
    )
    
    # Advanced options - Dynamic based on problem type
    # with st.expander("🔧 Simulation Parameters", expanded=True):
    # Detect problem type if analysis is available
    problem_type = None
    if 'current_analysis' in st.session_state:
        func_name = st.session_state.current_analysis.get('function_name', '')
        if 'heat_equation' in func_name:
            problem_type = 'transient_heat'
        elif 'steady_state' in func_name:
            problem_type = 'steady_heat'
        elif 'mechanics' in func_name or 'beam' in func_name or 'compression' in func_name or 'plate' in func_name:
            problem_type = 'mechanics'
    
    # Run button
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

with col2:
    st.subheader("🤖 AI Analysis")
    
    if run_button and problem_description:
        with st.spinner("Analyzing your problem..."):
            # Call LLM to analyze the problem
            analysis = analyze_problem_with_llm(
                problem_description, 
                api_key,
                model
            )
            
            st.session_state.current_analysis = analysis
            
            # Display analysis
            st.success("✅ Problem analyzed successfully!")
            
            st.write("**Detected Problem Type:**")
            st.info(analysis['problem_type'])
            
            st.write("**Selected Function:**")
            st.code(analysis['function_name'])
            

# Results section
if run_button and problem_description and 'current_analysis' in st.session_state:
    st.divider()
    st.subheader("📊 Simulation Results")
    
    analysis = st.session_state.current_analysis
    
    # Run the simulation
    with st.spinner("Running simulation..."):
        try:
            # Gather optional parameters based on problem type
            alpha_param = None
            thermal_conductivity_param = None
            
            if problem_type == 'transient_heat':
                alpha_param = 0.3
            elif problem_type == 'steady_heat':
                thermal_conductivity_param = 0.1
            
            result = run_simulation(
                analysis['function_name'],
                analysis.get('parameters', {}),
                50,
                1,
                2,
                alpha=alpha_param,
                thermal_conductivity=thermal_conductivity_param
            )
            
            # Display results
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                st.metric("Status", "✅ Complete")
            with col_r2:
                st.metric("Mesh Elements", result.get('num_elements', 'N/A'))
            with col_r3:
                st.metric("Solve Time", f"{result.get('solve_time', 0):.2f}s")
            
            # Show results
            if result.get('success'):
                st.success("Simulation completed successfully!")
                
                # Display metrics
                if 'metrics' in result:
                    st.write("**Key Results:**")
                    for key, value in result['metrics'].items():
                        st.write(f"- {key}: `{value}`")
                
                       
                # Download results
                if True:
                    st.write("**Output Files:**")
                    if result.get('output_dir'):
                        st.info(f"📁 Results saved to: `{result['output_dir']}`")
                        
                        # Show what files were created with details
                        if os.path.exists(result['output_dir']):
                            files = os.listdir(result['output_dir'])
                            if files:
                                # Categorize files
                                png_files = [f for f in files if f.endswith('.png')]
                                gif_files = [f for f in files if f.endswith('.gif')]
                                pvd_files = [f for f in files if f.endswith('.pvd')]
                                vtu_files = [f for f in files if f.endswith('.vtu')]
                                
                                col_f1, col_f2, col_f3 = st.columns(3)
                                
                                with col_f1:
                                    if png_files:
                                        st.metric("📊 PNG Plots", len(png_files))
                                    if gif_files:
                                        st.metric("🎬 Animations", len(gif_files))
                                
                                with col_f2:
                                    if pvd_files:
                                        st.metric("📦 PVD Files", len(pvd_files))
                                        st.caption("For ParaView")
                                
                                with col_f3:
                                    if vtu_files:
                                        st.metric("📄 VTU Files", len(vtu_files))
                                
                                # Show file list in expander
                                with st.expander("📂 View all files"):
                                    for f in sorted(files):
                                        file_path = os.path.join(result['output_dir'], f)
                                        size = os.path.getsize(file_path) / 1024  # KB
                                        if f.endswith('.png'):
                                            st.write(f"📊 `{f}` ({size:.1f} KB)")
                                        elif f.endswith('.gif'):
                                            st.write(f"🎬 `{f}` ({size:.1f} KB)")
                                        elif f.endswith('.pvd'):
                                            st.write(f"📦 `{f}` ({size:.1f} KB) - Open in ParaView")
                                        elif f.endswith('.vtu'):
                                            st.write(f"📄 `{f}` ({size:.1f} KB)")
                                        else:
                                            st.write(f"📄 `{f}` ({size:.1f} KB)")
                            else:
                                st.caption("No files found in output directory")
                    else:
                        st.warning("⚠️ Output directory not found")


                # Display plots directly from memory
                st.divider()
                st.subheader("🎨 Visualization")
                
                if 'plots' in result and result['plots']:
                    plots_dict = result['plots']
                    st.success(f"✅ Generated {len(plots_dict)} visualization(s) from simulation data")
                    
                    # Create tab names based on plot type
                    tab_names = []
                    plot_items = list(plots_dict.items())
                    
                    for plot_name, _ in plot_items:
                        if plot_name == 'animation':
                            tab_names.append("🎬 Animation")
                        elif plot_name == 'temperature':
                            tab_names.append("🌡️ Temperature")
                        elif plot_name == '3d_slices':
                            tab_names.append("🧊 3D Slices")
                        elif plot_name == '3d_interactive':
                            tab_names.append("🌐 Interactive 3D")
                        elif plot_name == 'results':
                            tab_names.append("📐 Results")
                        else:
                            tab_names.append(plot_name.replace('_', ' ').title())
                    
                    # Create tabs
                    tabs = st.tabs(tab_names)
                    
                    for idx, (tab, (plot_name, plot_data)) in enumerate(zip(tabs, plot_items)):
                        with tab:
                            # Check if it's a Plotly figure or BytesIO buffer
                            if plot_name == '3d_interactive':
                                # Display interactive Plotly figure
                                try:
                                    st.plotly_chart(plot_data, use_container_width=True)
                                    st.success("✨ **Interactive 3D Visualization**")
                                    st.info("💡 **Controls:**\n"
                                           "- **Rotate:** Click and drag\n"
                                           "- **Zoom:** Scroll or pinch\n"
                                           "- **Pan:** Right-click and drag\n"
                                           "- **Reset:** Double-click\n"
                                           "- **Hover:** See temperature values")
                                except Exception as e:
                                    st.error(f"Could not display interactive plot: {e}")
                            else:
                                # Display static image from BytesIO buffer
                                plot_data.seek(0)  # Reset buffer position
                                st.image(plot_data, use_container_width=True)
                                
                                # Show tips for special plot types
                                if plot_name == 'animation':
                                    st.info("💡 **Tip:** This animation shows time evolution of the simulation. Right-click to save.")
                                elif plot_name == '3d_slices':
                                    st.info("💡 **Tip:** These are 2D slices through the 3D domain at different z-levels. Check the Interactive 3D tab for full exploration.")
                else:
                    st.warning("⚠️ No visualizations generated.")
                    st.caption("💡 Plots are generated automatically from simulation results.")
         
                
                # Add to history
                st.session_state.simulation_history.append({
                    'description': problem_description,
                    'analysis': analysis,
                    'result': result
                })
            else:
                st.error(f"Simulation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error running simulation: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# History section
if st.session_state.simulation_history:
    st.divider()
    st.subheader("📜 Simulation History")
    
    for i, entry in enumerate(reversed(st.session_state.simulation_history[-5:])):
        with st.expander(f"Simulation {len(st.session_state.simulation_history) - i}: {entry['analysis']['problem_type']}"):
            st.write("**Description:**", entry['description'])
            st.write("**Function:**", entry['analysis']['function_name'])
            if entry['result'].get('success'):
                st.success("✅ Completed")
            else:
                st.error("❌ Failed")


if __name__ == "__main__":
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🔬 FEniCS Simulation Assistant | Powered by AI</p>
        <p><small>Make sure you're in the fenicsproject conda environment</small></p>
    </div>
    """, unsafe_allow_html=True)

