"""
Solid Mechanics Problems using FEniCS
======================================
This script solves linear elasticity problems:
    -div(σ(u)) = f    in Ω
    σ(u) = λ trace(ε(u))I + 2με(u)
    ε(u) = (∇u + (∇u)ᵀ)/2

where:
    u = displacement vector
    σ = stress tensor
    ε = strain tensor
    λ, μ = Lamé parameters
    f = body force
"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

def compute_lame_parameters(E, nu):
    """
    Compute Lamé parameters from Young's modulus and Poisson's ratio
    
    Args:
        E: Young's modulus
        nu: Poisson's ratio
    
    Returns:
        lambda, mu (Lamé parameters)
    """
    lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
    mu = E / (2 * (1 + nu))
    return lmbda, mu


def solve_cantilever_beam():
    """
    Solve cantilever beam problem:
    - Fixed at left end
    - Uniform load at right end
    - Returns displacement and stress fields
    """
    
    print("="*60)
    print("CANTILEVER BEAM ANALYSIS")
    print("="*60)
    
    # Create mesh: rectangular beam
    beam_length = 2.0   # Length
    H = 0.3   # Height
    mesh = RectangleMesh(Point(0, 0), Point(beam_length, H), 60, 20)
    
    # Define function space (vector for displacement)
    V = VectorFunctionSpace(mesh, 'P', 1)
    
    # Define boundary conditions
    # Fixed at left end (x = 0)
    def left_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-10)
    
    # Zero displacement at fixed end
    bc = DirichletBC(V, Constant((0, 0)), left_boundary)
    
    # Material properties (steel-like)
    E = 200e9    # Young's modulus (Pa)
    nu = 0.3     # Poisson's ratio
    lmbda, mu = compute_lame_parameters(E, nu)
    
    print(f"Material properties:")
    print(f"  Young's modulus: {E:.2e} Pa")
    print(f"  Poisson's ratio: {nu}")
    print(f"  Lamé λ: {lmbda:.2e}")
    print(f"  Lamé μ: {mu:.2e}")
    
    # Define strain and stress tensors
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma(u):
        return lmbda*tr(epsilon(u))*Identity(2) + 2*mu*epsilon(u)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    
    # Body force (gravity)
    rho = 7850  # Density (kg/m³) - steel
    g = 9.81    # Gravity (m/s²)
    f = Constant((0, -rho*g))
    
    # Applied traction at right end
    T = Constant((0, -1e8))  # Traction force at free end
    
    # Bilinear form (virtual work)
    a = inner(sigma(u), epsilon(v))*dx
    
    # Linear form
    L = dot(f, v)*dx
    
    # Add traction boundary condition
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], beam_length, 1e-10)
    
    # Mark boundaries
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    RightBoundary().mark(boundaries, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    
    L += dot(T, v)*ds(1)
    
    # Solve
    u = Function(V)
    print("\nSolving linear elasticity problem...")
    solve(a == L, u, bc)
    
    print("Solution completed!")
    
    # Compute stress
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)  # Deviatoric stress
    von_Mises = sqrt(3./2*inner(s, s))
    V_vm = FunctionSpace(mesh, 'P', 1)
    von_Mises_proj = project(von_Mises, V_vm)
    
    # Get max values
    u_vals = u.compute_vertex_values(mesh).reshape((mesh.num_vertices(), 2))
    max_displacement = np.max(np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2))
    max_von_mises = np.max(von_Mises_proj.compute_vertex_values(mesh))
    
    print(f"\nResults:")
    print(f"  Max displacement: {max_displacement:.6e} m")
    print(f"  Max von Mises stress: {max_von_mises:.2e} Pa")
    
    return u, von_Mises_proj, mesh


def solve_plate_with_hole():
    """
    Solve stress concentration problem:
    Plate with circular hole under tension
    """
    
    print("\n" + "="*60)
    print("PLATE WITH HOLE ANALYSIS (Stress Concentration)")
    print("="*60)
    
    # Create mesh (simplified - in practice use gmsh for complex geometry)
    # For this example, we'll use a square with refined mesh
    mesh = UnitSquareMesh(40, 40)
    
    # Define function space
    V = VectorFunctionSpace(mesh, 'P', 2)  # Quadratic elements for better accuracy
    
    # Boundary conditions
    # Left edge: fixed in x-direction
    def left(x, on_boundary):
        return on_boundary and near(x[0], 0, 1e-10)
    
    # Right edge: applied displacement (tension)
    def right(x, on_boundary):
        return on_boundary and near(x[0], 1, 1e-10)
    
    bc_left = DirichletBC(V.sub(0), Constant(0), left)
    bc_right = DirichletBC(V.sub(0), Constant(0.01), right)
    bcs = [bc_left, bc_right]
    
    # Material properties (aluminum)
    E = 70e9     # Young's modulus (Pa)
    nu = 0.33    # Poisson's ratio
    lmbda, mu = compute_lame_parameters(E, nu)
    
    print(f"Material properties (Aluminum):")
    print(f"  Young's modulus: {E:.2e} Pa")
    print(f"  Poisson's ratio: {nu}")
    
    # Define strain and stress
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma(u):
        return lmbda*tr(epsilon(u))*Identity(2) + 2*mu*epsilon(u)
    
    # Variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0, 0))  # No body force
    
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx
    
    # Solve
    u = Function(V)
    print("\nSolving linear elasticity problem...")
    solve(a == L, u, bcs)
    
    print("Solution completed!")
    
    # Compute von Mises stress
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)
    von_Mises = sqrt(3./2*inner(s, s))
    V_vm = FunctionSpace(mesh, 'P', 2)
    von_Mises_proj = project(von_Mises, V_vm)
    
    # Get statistics
    u_vals = u.compute_vertex_values(mesh).reshape((mesh.num_vertices(), 2))
    max_displacement = np.max(np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2))
    max_von_mises = np.max(von_Mises_proj.compute_vertex_values(mesh))
    
    print(f"\nResults:")
    print(f"  Max displacement: {max_displacement:.6e} m")
    print(f"  Max von Mises stress: {max_von_mises:.2e} Pa")
    
    return u, von_Mises_proj, mesh


def solve_compression_test():
    """
    Solve uniaxial compression test of a cylinder
    """
    
    print("\n" + "="*60)
    print("COMPRESSION TEST ANALYSIS")
    print("="*60)
    
    # Create mesh: rectangular block representing cylinder cross-section
    mesh = RectangleMesh(Point(0, 0), Point(0.1, 0.3), 20, 60)
    
    # Function space
    V = VectorFunctionSpace(mesh, 'P', 2)
    
    # Boundary conditions
    # Bottom: fixed
    def bottom(x, on_boundary):
        return on_boundary and near(x[1], 0, 1e-10)
    
    # Top: applied displacement (compression)
    def top(x, on_boundary):
        return on_boundary and near(x[1], 0.3, 1e-10)
    
    bc_bottom = DirichletBC(V, Constant((0, 0)), bottom)
    bc_top = DirichletBC(V.sub(1), Constant(-0.001), top)  # Compress by 1mm
    bcs = [bc_bottom, bc_top]
    
    # Material properties (rubber-like material)
    E = 1e6      # Young's modulus (Pa)
    nu = 0.45    # Poisson's ratio (nearly incompressible)
    lmbda, mu = compute_lame_parameters(E, nu)
    
    print(f"Material properties:")
    print(f"  Young's modulus: {E:.2e} Pa")
    print(f"  Poisson's ratio: {nu}")
    
    # Define strain and stress
    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    
    def sigma(u):
        return lmbda*tr(epsilon(u))*Identity(2) + 2*mu*epsilon(u)
    
    # Variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant((0, 0))
    
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(f, v)*dx
    
    # Solve
    u = Function(V)
    print("\nSolving linear elasticity problem...")
    solve(a == L, u, bcs)
    
    print("Solution completed!")
    
    # Compute strain energy
    strain_energy = assemble(0.5*inner(sigma(u), epsilon(u))*dx)
    
    # Compute von Mises stress
    s = sigma(u) - (1./3)*tr(sigma(u))*Identity(2)
    von_Mises = sqrt(3./2*inner(s, s))
    V_vm = FunctionSpace(mesh, 'P', 2)
    von_Mises_proj = project(von_Mises, V_vm)
    
    print(f"\nResults:")
    print(f"  Strain energy: {strain_energy:.6e} J")
    print(f"  Max von Mises stress: {np.max(von_Mises_proj.compute_vertex_values(mesh)):.2e} Pa")
    
    return u, von_Mises_proj, mesh


def plot_mechanics_results(u, stress, mesh, title="Results"):
    """Plot displacement and stress fields"""
    import os
    
    try:
        # Create subfolder based on title
        subfolder_name = title.lower().replace(' ', '_')
        subfolder_path = f"solid_mechanics_results/{subfolder_name}"
        os.makedirs(subfolder_path, exist_ok=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot displacement magnitude
        plt.subplot(1, 2, 1)
        u_magnitude = project(sqrt(dot(u, u)), FunctionSpace(mesh, 'P', 1))
        c1 = plot(u_magnitude, title=f'{title} - Displacement Magnitude')
        plt.colorbar(c1)
        
        # Plot von Mises stress
        plt.subplot(1, 2, 2)
        c2 = plot(stress, title=f'{title} - von Mises Stress')
        plt.colorbar(c2)
        
        plt.tight_layout()
        filename = f"{subfolder_path}/results_plot.png"
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to '{filename}'")
        plt.close()
        
        # Also save PVD files in the subfolder
        File(f'{subfolder_path}/displacement.pvd') << u
        File(f'{subfolder_path}/von_mises_stress.pvd') << stress
        print(f"PVD files saved to '{subfolder_path}/'")
        
    except Exception as e:
        print(f"Note: Could not create plot - {e}")


if __name__ == '__main__':
    # Create output directory
    import os
    os.makedirs('solid_mechanics_results', exist_ok=True)
    
    # Solve cantilever beam
    u1, stress1, mesh1 = solve_cantilever_beam()
    plot_mechanics_results(u1, stress1, mesh1, "Cantilever Beam")
    
    # Solve plate with hole
    u2, stress2, mesh2 = solve_plate_with_hole()
    plot_mechanics_results(u2, stress2, mesh2, "Plate with Tension")
    
    # Solve compression test
    u3, stress3, mesh3 = solve_compression_test()
    plot_mechanics_results(u3, stress3, mesh3, "Compression Test")
    
    print("\n" + "="*60)
    print("All solid mechanics simulations completed successfully!")
    print("="*60)

