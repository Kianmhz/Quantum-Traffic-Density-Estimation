# Quantum Traffic Density Estimation

A quantum computing approach to estimating traffic density using Grover's algorithm and Quantum Phase Estimation (QPE). This project demonstrates how quantum counting can achieve a theoretical quadratic speedup over classical counting methods.

## Overview

This system divides a video frame into a grid of regions, identifies which regions contain vehicles (using YOLO object detection), and uses **Quantum Counting** to estimate the traffic density — the fraction of regions occupied by cars.

### Key Innovation

Instead of classically counting all N regions (O(N) operations), quantum counting estimates the count M in **O(√N) oracle queries** — a quadratic speedup.

For a city-scale grid with 1,000,000 cells:
- **Classical**: 1,000,000 checks
- **Quantum**: ~1,000 oracle queries

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Video Frame    │ ──▶ │  YOLO Detection  │ ──▶ │  Bounding Boxes     │
│  (1920×1080)    │     │  (YOLOv8 + GPU)  │     │  [(x1,y1,x2,y2)...] │
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
│   │   ├── grid.py               # Grid division utilities
│   │   ├── boxes_to_occupancy.py # Convert bounding boxes to binary grid
│   │   ├── video_processor.py    # YOLO video processing
│   │   └── visualization.py      # Overlay drawing utilities
│   ├── quantum/
│   │   ├── grover_density_demo.py  # Original Grover search demo
│   │   └── quantum_counting.py     # Quantum counting implementation
│   ├── utils/
│   │   └── logging.py            # CSV logging and statistics
│   └── pipeline.py               # Main CLI entry point
├── logs/                         # Output logs (gitignored)
├── requirements.txt
└── README.md
```

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) GPU Acceleration
For NVIDIA GPU support (recommended for video processing):
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Terminal Demo (No Video Required)
```bash
python -m src.quantum.quantum_counting
```

Runs three test scenarios comparing quantum vs classical counting.

### Video Processing Pipeline
```bash
# Basic usage
python -m src.pipeline --video traffic.mp4

# With output video
python -m src.pipeline --video traffic.mp4 --output result.mp4

# Process single image
python -m src.pipeline --image intersection.jpg --output result.png
```

## CLI Reference

```
python -m src.pipeline [OPTIONS]
```

### Input Options (one required)
| Flag | Description |
|------|-------------|
| `--video PATH` | Path to input video file |
| `--image PATH` | Path to input image file |
| `--webcam [ID]` | Use webcam (optional camera ID, default 0) |

### Output Options
| Flag | Description |
|------|-------------|
| `--output PATH`, `-o` | Output file path (video or image) |
| `--no-preview` | Disable live preview window |

### Grid Options
| Flag | Default | Description |
|------|---------|-------------|
| `--rows N` | 4 | Grid rows |
| `--cols N` | 4 | Grid columns |

> ⚠️ Grid size (rows × cols) must be a power of 2 (e.g., 16, 32, 64)

### Quantum Options
| Flag | Default | Description |
|------|---------|-------------|
| `--no-quantum` | false | Disable quantum counting (classical only, faster) |
| `--precision N` | 6 | QPE precision qubits (more = accurate but slower) |
| `--shots N` | 1024 | Quantum measurement shots (more = accurate) |
| `--quantum-every N` | 5 | Run quantum every N frames (1 = every frame) |

### Processing Options
| Flag | Default | Description |
|------|---------|-------------|
| `--skip-frames N` | 0 | Frames to skip between processing |
| `--max-frames N` | None | Maximum frames to process |
| `--confidence F` | 0.5 | YOLO confidence threshold (0.0-1.0) |
| `--device DEV` | cuda | Device for YOLO: `cpu`, `cuda`, or `mps` |

### Logging Options
| Flag | Default | Description |
|------|---------|-------------|
| `--no-log` | false | Disable CSV logging |
| `--log-dir PATH` | logs | Directory for log files |

### Examples

```bash
# Fast mode (no quantum, classical only)
python -m src.pipeline --video traffic.mp4 --no-quantum

# High accuracy quantum (slower)
python -m src.pipeline --video traffic.mp4 --precision 7 --shots 2048 --quantum-every 1

# Balanced for real-time
python -m src.pipeline --video traffic.mp4 --precision 6 --quantum-every 10

# CPU-only processing
python -m src.pipeline --video traffic.mp4 --device cpu

# Custom grid (8×8 = 64 regions)
python -m src.pipeline --video traffic.mp4 --rows 8 --cols 8 --precision 8

# Process first 100 frames only
python -m src.pipeline --video traffic.mp4 --max-frames 100
```

### Keyboard Controls (during video playback)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `s` | Save screenshot |

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

1. Prepare uniform superposition over search space |s⟩
2. Apply controlled-G^(2^k) operations for each precision qubit k
3. Apply inverse QFT to counting register
4. Measure to get phase φ

### Phase Interpretation (Important!)

When the search register is initialized in uniform superposition |s⟩ (not a pure eigenstate), QPE produces measurement peaks **centered around φ = 0.5**, offset by ±θ/π:

$$\phi_{measured} = 0.5 \pm \frac{\theta}{\pi}$$

Therefore, to extract θ:

$$\theta = \pi \cdot |\phi_{measured} - 0.5|$$

And to get the count M:

$$M = N \cdot \sin^2(\theta)$$

### Precision vs Search Space

| Register | Symbol | Purpose |
|----------|--------|---------|
| **Search qubits** | n | Encode N = 2^n grid regions |
| **Precision qubits** | p | Determine phase measurement resolution (2^p bins) |

**Rule of thumb:** `precision ≥ search_qubits + 2` for accurate results.

| Grid | N | Search (n) | Recommended Precision (p) |
|------|---|------------|---------------------------|
| 4×4 | 16 | 4 | 6 |
| 8×4 | 32 | 5 | 7 |
| 8×8 | 64 | 6 | 8 |

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

Due to the symmetric eigenvalues (±θ), we observe peaks at φ and (1-φ), both symmetric around 0.5. They correspond to the same M value:

```python
# Extract theta from offset from 0.5
theta = math.pi * abs(phi - 0.5)
M = N * math.sin(theta) ** 2
```

## Output Logs

When processing video, the pipeline generates:

- **`logs/density_log_YYYYMMDD_HHMMSS.csv`**: Per-frame data
  - Frame number, timestamp, detections count
  - Classical count/density, quantum count/density
  - Error metrics, processing time

- **`logs/summary_YYYYMMDD_HHMMSS.txt`**: Session summary
  - Configuration used
  - Accuracy statistics
  - Error distribution histogram

## Results

Typical accuracy with precision=6 on 16-region grid:

| Scenario | Classical M | Quantum M | Error |
|----------|-------------|-----------|-------|
| Low density (M=2) | 2 | 2 | 0 |
| Moderate density (M=6) | 6 | 5-6 | ≤1 |
| High density (M=14) | 14 | 14 | 0 |

Average error: **< 1 region** with proper precision settings.

## Future Work

1. ~~**Video Integration**~~: ✅ YOLO + video pipeline complete
2. ~~**Visualization**~~: ✅ Grid overlay and density metrics
3. ~~**Logging**~~: ✅ CSV logging with statistics
4. **Larger Grids**: Scale to 8×8 (64 regions) or larger
5. **Real Hardware**: Run on IBM Quantum or other NISQ devices
6. **Error Mitigation**: Implement techniques for noisy quantum hardware
7. **Webcam Support**: Real-time processing from camera feed