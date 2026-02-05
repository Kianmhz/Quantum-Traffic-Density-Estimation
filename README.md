# Quantum Traffic Density Estimation

A quantum computing approach to estimating traffic density using Grover's algorithm and Quantum Phase Estimation (QPE). This project demonstrates how quantum counting can achieve a theoretical quadratic speedup over classical counting methods.

## Overview

This system divides a video frame into a grid of regions, identifies which regions contain vehicles (using object detection), and uses **Quantum Counting** to estimate the traffic density — the fraction of regions occupied by cars.

### Key Innovation

Instead of classically counting all N regions (O(N) operations), quantum counting estimates the count M in **O(√N) oracle queries** — a quadratic speedup.

For a city-scale grid with 1,000,000 cells:
- **Classical**: 1,000,000 checks
- **Quantum**: ~1,000 oracle queries

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Video Frame    │ ──▶ │  YOLO Detection  │ ──▶ │  Bounding Boxes     │
│  (1920×1080)    │     │  (Car Detection) │     │  [(x1,y1,x2,y2)...] │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Traffic        │ ◀── │  Quantum         │ ◀── │  Occupancy Grid     │
│  Density: 37.5% │     │  Counting (QPE)  │     │  [1,0,1,1,0,0,...]  │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

## Project Structure

```
Capstone/
├── src/
│   ├── vision/
│   │   ├── grid.py              # Grid division utilities
│   │   └── boxes_to_occupancy.py # Convert bounding boxes to binary grid
│   └── quantum/
│       ├── grover_density_demo.py  # Original Grover search demo
│       └── quantum_counting.py     # Quantum counting implementation
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Run Quantum Counting Demo
```bash
python -m src.quantum.quantum_counting
```

This runs three test scenarios (low, moderate, and high traffic density) and compares quantum estimates with classical ground truth.

### Example Output
```
Configuration:
  Grid size: 4 × 4 = 16 regions
  Search space qubits: 4
  Precision qubits (QPE): 4

Occupancy Grid (■ = occupied, □ = empty):
  ■ □ □ □
  ■ □ ■ □
  □ □ ■ ■
  □ □ □ ■

Classical Count: 6 regions (37.5%)
Quantum Estimate: 5 regions (31.2%)
Error: 1 region (16.7%)
```

## Theoretical Background

### Grover's Operator

The Grover operator G consists of:
1. **Oracle (O)**: Flips the phase of marked states (occupied regions)
2. **Diffusion (D)**: Amplifies marked states via inversion about the mean

$$G = D \cdot O$$

### Eigenvalue Structure

The key insight is that G has eigenvalues:

$$G|\psi_\pm\rangle = e^{\pm 2i\theta}|\psi_\pm\rangle$$

where:

$$\sin(\theta) = \sqrt{\frac{M}{N}}$$

This encodes the count M in the angle θ.

### Quantum Phase Estimation

QPE extracts the phase θ from G's eigenvalues:

1. Prepare uniform superposition over search space
2. Apply controlled-G^(2^k) operations
3. Apply inverse QFT to counting register
4. Measure to get φ ≈ θ/π

### Extracting the Count

From the measured phase φ:

$$M = N \cdot \sin^2(\pi \cdot \phi)$$

### Complexity Analysis

| Method | Oracle Queries | Total Operations |
|--------|---------------|------------------|
| Classical Counting | N | O(N) |
| Quantum Counting | O(√N) | O(√N log N) |

## Important Assumptions

> ⚠️ **Academic Honesty Note**: This project makes the following assumptions that should be clearly stated in any report or presentation.

### The Oracle Assumption

Quantum counting achieves O(√N) speedup **in oracle queries**. However, in our implementation:

1. We construct the oracle **classically** by iterating through all regions
2. This classical preprocessing is O(N)
3. The quantum speedup applies to the **oracle query complexity**, not total runtime

### When Would This Be Practical?

The quantum advantage would be realized if:
- Car detection could be performed by **quantum sensors** directly
- Occupancy data was stored in **quantum memory**
- The oracle was a true **black-box** (not constructed classically)

### What This Project Demonstrates

- Correct implementation of quantum counting algorithm
- How Grover's operator eigenvalues encode solution count
- QPE for phase extraction
- Theoretical foundation for future quantum sensing applications

## Algorithm Details

### Oracle Construction

For each marked index i, the oracle applies a phase flip:

```python
for idx in marked_indices:
    # X gates to convert |idx⟩ to |11...1⟩
    # Multi-controlled Z gate
    # Undo X gates
```

### Controlled Grover Powers

QPE requires controlled-G^(2^k) for each precision qubit k:

```python
for k in range(precision_qubits):
    power = 2 ** k
    for _ in range(power):
        controlled_grover = grover_op.control(1)
        qc.compose(controlled_grover, ...)
```

### Phase Interpretation

Due to ±θ eigenvalues, we observe two peaks at φ and (1-φ). Both correspond to the same M:

```python
# Group symmetric phases
phi_partner = 1.0 - phi
# Both give M = N * sin²(π*φ)
```

## Results

Typical accuracy with 4 precision qubits on 16-region grid:

| Scenario | Classical M | Quantum M | Relative Error |
|----------|-------------|-----------|----------------|
| Low density | 6 | 5-7 | ~15% |
| Moderate density | 10 | 9-11 | ~10% |
| High density | 15 | 14-16 | ~7% |

Accuracy improves with more precision qubits (at cost of deeper circuits).

## Future Work

1. **Video Integration**: Connect YOLO object detection for real-time processing
2. **Visualization**: Overlay grid and density metrics on video frames
3. **Larger Grids**: Scale to 8×8 (64 regions) or larger
4. **Real Hardware**: Run on IBM Quantum or other NISQ devices
5. **Error Mitigation**: Implement techniques for noisy quantum hardware