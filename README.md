# Haldane Chain Simulation

A C++ implementation for calculating ground state energies and energy gaps of spin-1/2 and spin-1 quantum chains using the Lanczos algorithm.

## Overview

This program simulates Heisenberg spin chains and analyzes their energy spectra, particularly focusing on the Haldane conjecture which predicts that integer spin chains have energy gaps while half-integer spin chains are gapless.

## Features

- **Spin-1/2 chains**: Exact diagonalization using bit representation
- **Spin-1 chains**: Exact diagonalization using base-3 representation  
- **Lanczos algorithm**: Efficient calculation of lowest eigenvalues
- **Boundary conditions**: Support for both open (OBC) and periodic (PBC) boundary conditions
- **Scaling analysis**: Calculate energy gaps as function of chain length

## Physics Background

The Hamiltonian for the Heisenberg spin chain is:

```
H = J Σᵢ (Sᵢˣ Sᵢ₊₁ˣ + Sᵢʸ Sᵢ₊₁ʸ + Sᵢᶻ Sᵢ₊₁ᶻ)
```

Where:
- `J = 1.0` (antiferromagnetic coupling)
- `Sᵢᵅ` are spin operators at site i
- Sum runs over nearest neighbors

## SYK Model Simulation

`syk.cpp` performs exact diagonalization of the Majorana SYK (q=4) model
`H = Σ J_abcd χ_a χ_b χ_c χ_d`, `<J²> = 3! J²/N³` (no Eigen needed).

```bash
make syk
./syk <N> [samples=10] [seed=1] [J=1] [dump_spectrum=0]   # N even, 4..24
./syk 16 50 1
```

Add an inverse temperature to also compute the Euclidean Green function
`G(τ) = (1/N) Σ_a <χ_a(τ) χ_a(0)>_β` (full-space diagonalization, N ≤ 16):

```bash
./syk 12 20 1 1 0 10     # N=12, 20 samples, seed=1, J=1, no spectrum dump, beta=10
```

Checks: G(0)=1/2 and G(τ)=G(β-τ). Compare with the large-N conformal result
`G = b (π/(βJ sin(πτ/β)))^(1/2)`, `b=(4π)^(-1/4)`.

Outputs E0/N, gap, bandwidth and the level-spacing ratio <r>
(GOE 0.531 for N%8=0, GUE 0.600 for N%8=2,6, GSE 0.674 for N%8=4).

## Requirements

- C++17 or later
- Eigen3 library for matrix operations
- Standard C++ libraries

## Compilation

### Manual compilation:
```bash
g++ -std=c++17 -O3 -I/path/to/eigen3 haldane.cpp -o haldane
```

### Using Makefile (recommended):
```bash
make                    # Build optimized version
make debug              # Build debug version with symbols
make test               # Run comprehensive test suite
make help               # Show all available options
make version            # Show version information
```

## Usage

```bash
./haldane <spin> <L_max> <lanczos_steps> [periodic] [L_min]
```

### Parameters

- `spin`: Spin value (0.5 or 1)
- `L_max`: Maximum chain length
- `lanczos_steps`: Number of Lanczos iterations
- `periodic`: Boundary conditions (0=OBC, 1=PBC) [optional, default=0]
- `L_min`: Minimum chain length [optional, default=4]

### Examples

```bash
# Spin-1/2 chain, L=4 to 12, open boundary conditions
./haldane 0.5 12 50 0 4

# Spin-1 chain, L=4 to 10, periodic boundary conditions  
./haldane 1 10 100 1 4
```

## Output Format

The program outputs a table with columns:
- `L`: Chain length
- `Ground_E`: Ground state energy
- `1st_excited_E`: First excited state energy
- `Gap`: Energy gap (E₁ - E₀)
- `E_per_site`: Ground state energy per site
- `Gap_per_site`: Energy gap per site

## Expected Results

### Spin-1/2 Chains
- Gapless in thermodynamic limit
- Gap scales as ~1/L for finite systems
- Ground state energy per site → -ln(2) ≈ -0.693

### Spin-1 Chains (Haldane Gap)
- Finite gap in thermodynamic limit (~0.41)
- Exponential decay of correlations
- Ground state energy per site → -1.401

## Implementation Details

- **Spin-1/2**: States encoded as bit strings (2^L dimension)
- **Spin-1**: States encoded in base-3 (3^L dimension)
- **Lanczos**: Symmetric tridiagonal matrix diagonalization with convergence checking
- **Memory**: Optimized for large system sizes
- **Error Handling**: Comprehensive error messages and input validation
- **Constants**: Well-defined numerical thresholds for stability

## References

1. Haldane, F.D.M. "Nonlinear field theory of large-spin Heisenberg antiferromagnets" (1983)
2. White, S.R. "Density matrix formulation for quantum renormalization groups" (1992)
3. Affleck, I. "Quantum spin chains and the Haldane gap" (1989)

## Recent Updates

### Version 1.0.0
- Added convergence checking in Lanczos algorithm
- Improved error handling and user feedback
- Enhanced Makefile with debug build and comprehensive tests
- Added version information and better documentation

## License

MIT License - see LICENSE file for details.
