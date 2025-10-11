"""
Heat Transfer Problem using FEniCS
===================================
This script solves the transient heat equation (diffusion equation):
    ∂u/∂t - α∇²u = f

where:
    u = temperature
    α = thermal diffusivity
    f = heat source term

Problem setup: Heat diffusion in a 2D square plate with initial temperature
distribution and boundary conditions.
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def solve_heat_equation(
    nx=50, 
    ny=50,
    initial_condition='100*exp(-50*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))',
    boundary_value=0.0,
    T_final=2.0,
    num_steps=50,
    alpha=0.3,
    heat_source=0.0
):
    """
    Solve 2D transient heat equation with configurable parameters.
    
    Parameters:
    -----------
    nx, ny : int
        Number of mesh elements in x and y directions (default: 50)
    initial_condition : str or Expression
        Initial temperature distribution (default: Gaussian at center)
    boundary_value : float or Constant
        Temperature at boundaries (default: 0.0)
    T_final : float
        Final simulation time (default: 2.0)
    num_steps : int
        Number of time steps (default: 50)
    alpha : float
        Thermal diffusivity (default: 0.3)
    heat_source : float or Expression
        Internal heat generation (default: 0.0)
    
    Returns:
    --------
    u : Function
        Final temperature field
    mesh : Mesh
        Computational mesh
    temperatures : list
        Temperature snapshots at selected times
    times : list
        Corresponding time values
    """
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)  # Linear Lagrange elements
    
    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    
    if isinstance(boundary_value, (int, float)):
        u_D = Constant(boundary_value)
    else:
        u_D = boundary_value
    bc = DirichletBC(V, u_D, boundary)
    
    # Define initial condition
    if isinstance(initial_condition, str):
        u_0 = Expression(initial_condition, degree=2)
    else:
        u_0 = initial_condition
    u_n = interpolate(u_0, V)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    if isinstance(heat_source, (int, float)):
        f = Constant(heat_source)
    else:
        f = heat_source
    
    # Time parameters
    dt = T_final / num_steps  # Time step size
    
    # Bilinear and linear forms (Backward Euler)
    a = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx
    L = (u_n + dt*f)*v*dx
    
    # Create solution function
    u = Function(V)
    
    # Time-stepping
    t = 0
    temperatures = []
    times = []
    solutions = []  # Store Function objects for animation
    
    print("Solving transient heat equation...")
    print(f"Mesh elements: {mesh.num_cells()}")
    print(f"Mesh resolution: {nx} x {ny}")
    print(f"Time step: {dt:.4f}, Final time: {T_final}")
    print(f"Thermal diffusivity: {alpha}")
    
    # Store every Nth solution for animation (to avoid too many frames)
    animation_interval = max(1, num_steps // 20)  # Store ~20 frames
    
    for n in range(num_steps):
        # Update current time
        t += dt
        
        # Solve variational problem
        solve(a == L, u, bc)
        
        # Store solution for animation
        if n % animation_interval == 0:
            u_copy = Function(V)
            u_copy.assign(u)
            solutions.append(u_copy)
            temperatures.append(u.compute_vertex_values(mesh))
            times.append(t)
        
        # Print progress
        if n % 10 == 0:
            max_temp = np.max(u.compute_vertex_values(mesh))
            print(f"Time: {t:.2f}, Max temperature: {max_temp:.4f}")
        
        # Update previous solution
        u_n.assign(u)
    
    print("\nSolution completed!")
    print(f"Stored {len(solutions)} snapshots for animation")
    
    return u, mesh, temperatures, times, solutions


def solve_steady_state_heat(
    nx=40,
    ny=40,
    boundary_value=0.0,
    heat_source='100*exp(-100*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5)))',
    thermal_conductivity=1.0
):
    """
    Solve steady-state heat equation with heat source:
        -k∇²u = f
    
    Parameters:
    -----------
    nx, ny : int
        Number of mesh elements in x and y directions (default: 40)
    boundary_value : float or Constant
        Temperature at boundaries (default: 0.0)
    heat_source : str or Expression
        Internal heat generation rate (default: Gaussian at center)
    thermal_conductivity : float
        Thermal conductivity k (default: 1.0)
    
    Returns:
    --------
    u : Function
        Temperature field
    mesh : Mesh
        Computational mesh
    """
    
    print("\n" + "="*60)
    print("Solving steady-state heat equation with heat source")
    print("="*60)
    
    # Create mesh and function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    
    if isinstance(boundary_value, (int, float)):
        u_D = Constant(boundary_value)
    else:
        u_D = boundary_value
    bc = DirichletBC(V, u_D, boundary)
    
    # Define heat source
    if isinstance(heat_source, str):
        f = Expression(heat_source, degree=2)
    else:
        f = heat_source
    
    print(f"Mesh elements: {mesh.num_cells()}")
    print(f"Mesh resolution: {nx} x {ny}")
    print(f"Thermal conductivity: {thermal_conductivity}")
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = thermal_conductivity * dot(grad(u), grad(v))*dx
    L = f*v*dx
    
    # Solve
    u = Function(V)
    solve(a == L, u, bc)
    
    max_temp = np.max(u.compute_vertex_values(mesh))
    print(f"Maximum temperature: {max_temp:.4f}")
    
    return u, mesh


def solve_heat_equation_3d(
    nx=30,
    ny=30,
    nz=30,
    initial_condition='100*exp(-50*((x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) + (x[2]-0.5)*(x[2]-0.5)))',
    boundary_value=0.0,
    T_final=2.0,
    num_steps=50,
    alpha=0.3,
    heat_source=0.0
):
    """
    Solve 3D transient heat equation with configurable parameters.
    
    Parameters:
    -----------
    nx, ny, nz : int
        Number of mesh elements in x, y, and z directions (default: 30)
    initial_condition : str or Expression
        Initial temperature distribution (default: Gaussian at center)
    boundary_value : float or Constant
        Temperature at boundaries (default: 0.0)
    T_final : float
        Final simulation time (default: 2.0)
    num_steps : int
        Number of time steps (default: 50)
    alpha : float
        Thermal diffusivity (default: 0.3)
    heat_source : float or Expression
        Internal heat generation (default: 0.0)
    
    Returns:
    --------
    u : Function
        Final temperature field
    mesh : Mesh
        Computational mesh (3D)
    temperatures : list
        Temperature snapshots at selected times
    times : list
        Corresponding time values
    """
    
    print("\n" + "="*60)
    print("Solving 3D transient heat equation")
    print("="*60)
    
    # Create 3D mesh
    mesh = UnitCubeMesh(nx, ny, nz)
    V = FunctionSpace(mesh, 'P', 1)  # Linear Lagrange elements
    
    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    
    if isinstance(boundary_value, (int, float)):
        u_D = Constant(boundary_value)
    else:
        u_D = boundary_value
    bc = DirichletBC(V, u_D, boundary)
    
    # Define initial condition
    if isinstance(initial_condition, str):
        u_0 = Expression(initial_condition, degree=2)
    else:
        u_0 = initial_condition
    u_n = interpolate(u_0, V)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    if isinstance(heat_source, (int, float)):
        f = Constant(heat_source)
    else:
        f = heat_source
    
    # Time parameters
    dt = T_final / num_steps  # Time step size
    
    # Bilinear and linear forms (Backward Euler)
    a = u*v*dx + dt*alpha*dot(grad(u), grad(v))*dx
    L = (u_n + dt*f)*v*dx
    
    # Create solution function
    u = Function(V)
    
    # Time-stepping
    t = 0
    temperatures = []
    times = []
    solutions = []  # Store Function objects for later analysis
    
    print(f"Mesh elements: {mesh.num_cells()}")
    print(f"Mesh resolution: {nx} x {ny} x {nz}")
    print(f"Time step: {dt:.4f}, Final time: {T_final}")
    print(f"Thermal diffusivity: {alpha}")
    
    # Store every Nth solution
    animation_interval = max(1, num_steps // 20)  # Store ~20 frames
    
    for n in range(num_steps):
        # Update current time
        t += dt
        
        # Solve variational problem
        solve(a == L, u, bc)
        
        # Store solution
        if n % animation_interval == 0:
            u_copy = Function(V)
            u_copy.assign(u)
            solutions.append(u_copy)
            temperatures.append(u.compute_vertex_values(mesh))
            times.append(t)
        
        # Print progress
        if n % 10 == 0:
            max_temp = np.max(u.compute_vertex_values(mesh))
            print(f"Time: {t:.2f}, Max temperature: {max_temp:.4f}")
        
        # Update previous solution
        u_n.assign(u)
    
    print("\nSolution completed!")
    print(f"Stored {len(solutions)} snapshots")
    
    return u, mesh, temperatures, times, solutions


def create_animation(solutions, times, mesh, title, subfolder_path):
    """
    Create an animation showing temperature evolution over time
    
    Parameters:
    -----------
    solutions : list of Functions
        Temperature field at different time steps
    times : list of floats
        Corresponding time values
    mesh : Mesh
        Computational mesh
    title : str
        Title for the animation
    subfolder_path : str
        Path to save the animation
    """
    try:
        print("Creating animation...")
        
        # Get mesh coordinates for plotting
        coords = mesh.coordinates()
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Get triangulation for contour plot
        cells = mesh.cells()
        
        # Find temperature range for consistent colorbar
        all_temps = [sol.compute_vertex_values(mesh) for sol in solutions]
        vmin = min(np.min(temps) for temps in all_temps)
        vmax = max(np.max(temps) for temps in all_temps)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create initial contour plot
        temp_values = solutions[0].compute_vertex_values(mesh)
        contourf = ax.tricontourf(x, y, cells, temp_values, levels=20, 
                                   cmap='hot', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Temperature')
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - Time Evolution')
        ax.set_aspect('equal')
        
        def update(frame):
            """Update function for animation"""
            ax.clear()
            temp_values = solutions[frame].compute_vertex_values(mesh)
            contourf = ax.tricontourf(x, y, cells, temp_values, levels=20,
                                      cmap='hot', vmin=vmin, vmax=vmax)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f'{title} - Time Evolution')
            ax.set_aspect('equal')
            
            # Update time text
            time_text = ax.text(0.02, 0.95, f'Time: {times[frame]:.3f} s', 
                               transform=ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                               fontsize=12)
            return contourf, time_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(solutions), 
                           interval=200, blit=False, repeat=True)
        
        # Save as GIF
        gif_filename = f'{subfolder_path}/animation.gif'
        writer = PillowWriter(fps=5)
        anim.save(gif_filename, writer=writer)
        print(f"Animation saved to '{gif_filename}'")
        
        plt.close()
        
    except Exception as e:
        print(f"Note: Could not create animation - {e}")
        import traceback
        traceback.print_exc()


def plot_results(u, mesh, title="Results", solutions=None, times=None):
    """
    Plot the solution and save to subfolder based on title
    
    Parameters:
    -----------
    u : Function
        Temperature field to plot
    mesh : Mesh
        Computational mesh
    title : str
        Title for the plots
    solutions : list of Functions, optional
        Temperature fields at different times for animation
    times : list of floats, optional
        Time values corresponding to solutions
    """
    import os
    
    try:
        # Create subfolder based on title
        subfolder_name = title.lower().replace(' ', '_')
        subfolder_path = f"heat_transfer_results/{subfolder_name}"
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Plot final solution
        plt.figure(figsize=(10, 8))
        c = plot(u, title=f'{title} - Temperature Distribution')
        plt.colorbar(c)
        plt.xlabel('x')
        plt.ylabel('y')
        filename = f'{subfolder_path}/temperature_plot.png'
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to '{filename}'")
        plt.close()
        
        # Create animation if transient data is provided
        if solutions is not None and times is not None and len(solutions) > 1:
            create_animation(solutions, times, mesh, title, subfolder_path)
        
        # Save PVD file in the subfolder
        vtkfile = File(f'{subfolder_path}/solution.pvd')
        vtkfile << u
        print(f"PVD file saved to '{subfolder_path}/solution.pvd'")
        
    except Exception as e:
        print(f"Note: Could not create plot - {e}")


def plot_results_3d(u, mesh, title="3D Results", solutions=None, times=None):
    """
    Plot the 3D solution and save to subfolder based on title
    
    Parameters:
    -----------
    u : Function
        Temperature field to plot (3D)
    mesh : Mesh
        Computational mesh (3D)
    title : str
        Title for the plots
    solutions : list of Functions, optional
        Temperature fields at different times
    times : list of floats, optional
        Time values corresponding to solutions
    """
    import os
    
    try:
        # Create subfolder based on title
        subfolder_name = title.lower().replace(' ', '_')
        subfolder_path = f"heat_transfer_results/{subfolder_name}"
        os.makedirs(subfolder_path, exist_ok=True)
        
        # For 3D, create slice plots
        print(f"Creating 3D visualization for '{title}'...")
        
        # Get temperature values
        temp_values = u.compute_vertex_values(mesh)
        coords = mesh.coordinates()
        
        # Create slice plots at different z-levels
        fig = plt.figure(figsize=(15, 5))
        
        # Find nodes at different z slices
        z_slices = [0.25, 0.5, 0.75]
        
        for idx, z_level in enumerate(z_slices):
            ax = fig.add_subplot(1, 3, idx + 1)
            
            # Find nodes close to this z-level
            tolerance = 0.05
            slice_mask = np.abs(coords[:, 2] - z_level) < tolerance
            
            if np.any(slice_mask):
                slice_coords = coords[slice_mask]
                slice_temps = temp_values[slice_mask]
                
                # Create scatter plot
                scatter = ax.scatter(slice_coords[:, 0], slice_coords[:, 1], 
                                    c=slice_temps, cmap='hot', s=20)
                plt.colorbar(scatter, ax=ax)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(f'Temperature at z={z_level:.2f}')
                ax.set_aspect('equal')
        
        plt.tight_layout()
        filename = f'{subfolder_path}/temperature_slices.png'
        plt.savefig(filename, dpi=150)
        print(f"3D slice plot saved to '{filename}'")
        plt.close()
        
        # Save PVD file for ParaView (best for 3D visualization)
        vtkfile = File(f'{subfolder_path}/solution_3d.pvd')
        vtkfile << u
        print(f"3D PVD file saved to '{subfolder_path}/solution_3d.pvd'")
        print("💡 Open the PVD file in ParaView for interactive 3D visualization")
        
        # Save statistics
        with open(f'{subfolder_path}/stats.txt', 'w') as f:
            f.write(f"{title}\n")
            f.write("="*60 + "\n")
            f.write(f"Mesh cells: {mesh.num_cells()}\n")
            f.write(f"Mesh vertices: {mesh.num_vertices()}\n")
            f.write(f"Min temperature: {np.min(temp_values):.4f}\n")
            f.write(f"Max temperature: {np.max(temp_values):.4f}\n")
            f.write(f"Mean temperature: {np.mean(temp_values):.4f}\n")
        print(f"Statistics saved to '{subfolder_path}/stats.txt'")
        
    except Exception as e:
        print(f"Note: Could not create 3D plot - {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Create output directory
    import os
    os.makedirs('heat_transfer_results', exist_ok=True)
    
    # Check for command line arguments
    run_3d = '--3d' in sys.argv or len(sys.argv) > 1 and sys.argv[1] == '3d'
    
    if run_3d:
        # Run 3D example
        print("="*60)
        print("3D TRANSIENT HEAT TRANSFER ANALYSIS")
        print("="*60)
        print("⚠️  Note: 3D simulations may take longer to compute")
        u_3d, mesh_3d, temps_3d, times_3d, solutions_3d = solve_heat_equation_3d(
            nx=20, ny=20, nz=20,  # Smaller mesh for faster computation
            num_steps=30,
            T_final=1.0
        )
        plot_results_3d(u_3d, mesh_3d, "3D Heat Diffusion")
        
        print("\n" + "="*60)
        print("3D simulation completed successfully!")
        print("="*60)
    else:
        # Run default 2D examples
        print("="*60)
        print("TRANSIENT HEAT TRANSFER ANALYSIS (2D)")
        print("="*60)
        u_transient, mesh_transient, temps, times, solutions = solve_heat_equation()
        plot_results(u_transient, mesh_transient, "Transient Heat Diffusion", 
                    solutions=solutions, times=times)
        
        # Solve steady-state problem
        u_steady, mesh_steady = solve_steady_state_heat()
        plot_results(u_steady, mesh_steady, "Steady State Heat Source")
        
        print("\n" + "="*60)
        print("All 2D simulations completed successfully!")
        print("💡 Run with '--3d' or '3d' argument to solve 3D problems")
        print("   Example: python heat_transfer.py --3d")
        print("="*60)
   

