# FEniCS Examples: Heat Transfer and Solid Mechanics

This directory contains Python scripts demonstrating the use of FEniCS for solving partial differential equations (PDEs) in heat transfer and solid mechanics applications.

## Overview

FEniCS is a powerful open-source computing platform for solving PDEs using the finite element method. These examples showcase common engineering problems.

## Files

- **`heat_transfer.py`** - Heat transfer simulations
  - Transient heat equation with Gaussian initial condition
  - Steady-state heat equation with internal heat source
  
- **`solid_mechanics.py`** - Solid mechanics simulations
  - Cantilever beam under load
  - Plate under tension (stress concentration)
  - Compression test simulation
  
- **`requirements.txt`** - Python dependencies

## Installation

### Recommended: Using Conda

```bash
# Create a new conda environment with FEniCS
conda create -n fenicsproject -c conda-forge fenics mshr

# Activate the environment
conda activate fenicsproject

# Install additional dependencies
pip install matplotlib numpy scipy
```

### Alternative: Using Docker

```bash
# Pull the FEniCS Docker image
docker pull quay.io/fenicsproject/stable:latest

# Run the container with your code mounted
docker run -ti -v $(pwd):/home/fenics/shared quay.io/fenicsproject/stable
```

### Using pip (may have issues on some systems)

```bash
pip install -r requirements.txt
```

## Usage

### Heat Transfer Examples

```bash
python heat_transfer.py
```

This script solves:

1. **Transient Heat Equation**: Time-dependent heat diffusion in a 2D plate
   - Initial condition: Gaussian temperature distribution
   - Boundary conditions: Zero temperature at edges
   - Uses backward Euler time-stepping

2. **Steady-State Heat Equation**: Equilibrium temperature with heat source
   - Internal heat generation
   - Fixed boundary temperatures

**Output:**
- Results saved to `heat_transfer_results/` directory
- `.pvd` files for visualization in ParaView
- `.png` plots of temperature distribution

### Solid Mechanics Examples

```bash
python solid_mechanics.py
```

This script solves:

1. **Cantilever Beam**: Classic beam bending problem
   - Fixed at one end
   - Load at free end
   - Computes displacement and von Mises stress

2. **Plate Under Tension**: Stress concentration analysis
   - Uniaxial tension
   - Demonstrates stress distribution

3. **Compression Test**: Uniaxial compression
   - Material deformation under compression
   - Nearly incompressible material (rubber-like)

**Output:**
- Results saved to `solid_mechanics_results/` directory
- `.pvd` files for visualization in ParaView
- `.png` plots of displacement and stress fields

## Visualization

### Using Matplotlib
The scripts automatically generate PNG images showing the results.

### Using ParaView (Recommended for 3D visualization)

1. Install [ParaView](https://www.paraview.org/download/)
2. Open the `.pvd` files from the results directories
3. Apply appropriate color maps and filters

```bash
# Example: Open results in ParaView
paraview heat_transfer_results/solution.pvd
paraview solid_mechanics_results/displacement.pvd
```

## Problem Details

### Heat Transfer

**Governing Equation:**
```
∂u/∂t - α∇²u = f
```

Where:
- `u` = temperature
- `α` = thermal diffusivity
- `f` = heat source term

**Applications:**
- Electronic cooling
- Building thermal analysis
- Material processing

### Solid Mechanics

**Governing Equations:**
```
-div(σ(u)) = f
σ = λ trace(ε)I + 2με
ε = (∇u + (∇u)ᵀ)/2
```

Where:
- `u` = displacement vector
- `σ` = stress tensor
- `ε` = strain tensor
- `λ, μ` = Lamé parameters
- `E` = Young's modulus
- `ν` = Poisson's ratio

**Applications:**
- Structural analysis
- Material testing
- Mechanical design

## Material Properties

### Heat Transfer
- Thermal diffusivity: α = 0.3

### Solid Mechanics
- **Steel** (cantilever beam):
  - Young's modulus: E = 200 GPa
  - Poisson's ratio: ν = 0.3
  - Density: ρ = 7850 kg/m³

- **Aluminum** (plate):
  - Young's modulus: E = 70 GPa
  - Poisson's ratio: ν = 0.33

- **Rubber** (compression):
  - Young's modulus: E = 1 MPa
  - Poisson's ratio: ν = 0.45

## Customization

You can modify the scripts to:
- Change mesh resolution (increase `nx`, `ny` for finer mesh)
- Adjust material properties (E, ν, α, etc.)
- Modify boundary conditions
- Add more complex geometries (use `mshr` or `gmsh`)
- Change time steps and simulation duration
- Add different loads and constraints

## Troubleshooting

### FEniCS Installation Issues

If you encounter installation problems:

1. **Use Docker**: The most reliable method
   ```bash
   docker pull quay.io/fenicsproject/stable:latest
   ```

2. **Use Conda**: Works well on most systems
   ```bash
   conda install -c conda-forge fenics
   ```

3. **Check Python version**: FEniCS 2019.1.0 works with Python 3.6-3.8

### Common Errors

- **"Unable to find fenics"**: Install using conda or docker
- **"PETSc error"**: Check that all dependencies are properly installed
- **"Mesh generation failed"**: Reduce mesh resolution or check geometry

## References

- [FEniCS Project](https://fenicsproject.org/)
- [FEniCS Documentation](https://fenicsproject.org/documentation/)
- [FEniCS Tutorial](https://fenicsproject.org/tutorial/)
- [FEniCS Book](https://fenicsproject.org/book/)

## License

These examples are provided for educational purposes. FEniCS is licensed under LGPL v3.

## Contributing

Feel free to extend these examples with:
- More complex geometries
- Nonlinear problems
- Coupled physics (thermo-mechanics)
- 3D simulations
- Advanced material models
