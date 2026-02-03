"""
Grover-based traffic density estimation from a binary occupancy grid.

Uses a multi-solution Grover oracle to phase-flip basis states corresponding
to occupied regions, then estimates density from measurement statistics.
"""

import math
from typing import List, Dict, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from src.vision.grid import make_grid, index_to_rc
from src.vision.boxes_to_occupancy import boxes_to_occupancy


def build_multi_solution_oracle(n_qubits: int, marked_indices: List[int]) -> QuantumCircuit:
    """
    Build a Grover oracle that phase-flips all basis states |i⟩ where i is in marked_indices.
    
    This is a multi-solution oracle: it applies a phase of -1 to multiple marked states.
    
    Args:
        n_qubits: Number of qubits (N = 2^n_qubits total basis states).
        marked_indices: List of indices to mark (phase-flip).
    
    Returns:
        QuantumCircuit implementing the oracle.
    """
    qc = QuantumCircuit(n_qubits, name="Oracle")
    
    for idx in marked_indices:
        # Convert index to binary string (LSB at q0)
        bits = format(idx, f'0{n_qubits}b')[::-1]  # Reverse for LSB-first
        
        # Apply X gates to qubits that should be |0⟩ in this basis state
        for q, b in enumerate(bits):
            if b == '0':
                qc.x(q)
        
        # Apply multi-controlled Z (phase flip for this state)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            # For n > 2, use mcx with an ancilla trick or decompose CZ
            # Using H-MCX-H to implement multi-controlled Z
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        
        # Undo X gates
        for q, b in enumerate(bits):
            if b == '0':
                qc.x(q)
    
    return qc


def build_diffusion(n_qubits: int) -> QuantumCircuit:
    """
    Build the Grover diffusion operator (inversion about the mean) for n qubits.
    
    Implements: 2|s⟩⟨s| - I where |s⟩ is the uniform superposition.
    
    Args:
        n_qubits: Number of qubits.
    
    Returns:
        QuantumCircuit implementing the diffusion operator.
    """
    qc = QuantumCircuit(n_qubits, name="Diffusion")
    
    # H gates
    qc.h(range(n_qubits))
    
    # X gates
    qc.x(range(n_qubits))
    
    # Multi-controlled Z gate (phase flip |00...0⟩)
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.x(0)  # For single qubit, just X since we already applied X
    elif n_qubits == 2:
        qc.cx(0, 1)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    
    # Undo X gates
    qc.x(range(n_qubits))
    
    # H gates
    qc.h(range(n_qubits))
    
    return qc


def estimate_density_quantum(
    occupancy: List[int],
    n_iterations: int = 1,
    shots: int = 1024
) -> Tuple[float, Dict[str, int]]:
    """
    Estimate traffic density using a Grover-style quantum circuit.
    
    The oracle marks occupied regions. After Grover iterations, we measure
    and estimate density by comparing probability mass on occupied indices
    vs total measurements.
    
    Args:
        occupancy: Binary list where 1 = occupied region.
        n_iterations: Number of Grover iterations (oracle + diffusion pairs).
        shots: Number of measurement shots.
    
    Returns:
        Tuple of (estimated_density, measurement_counts).
    """
    N = len(occupancy)
    n_qubits = int(math.log2(N))
    
    if 2 ** n_qubits != N:
        raise ValueError(f"Occupancy length {N} must be a power of 2")
    
    # Find marked (occupied) indices
    marked_indices = [i for i, occ in enumerate(occupancy) if occ == 1]
    
    if not marked_indices:
        # No occupied regions - return zero density
        return 0.0, {"0" * n_qubits: shots}
    
    # Build the circuit
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize uniform superposition
    qc.h(range(n_qubits))
    
    # Build oracle and diffusion
    oracle = build_multi_solution_oracle(n_qubits, marked_indices)
    diffusion = build_diffusion(n_qubits)
    
    # Apply Grover iterations
    for _ in range(n_iterations):
        qc.compose(oracle, range(n_qubits), inplace=True)
        qc.compose(diffusion, range(n_qubits), inplace=True)
    
    # Measure
    qc.measure(range(n_qubits), range(n_qubits))
    
    # Run on simulator
    backend = Aer.get_backend("aer_simulator")
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    
    # Estimate density from measurement results
    # Count measurements that correspond to occupied indices
    occupied_count = 0
    for bitstring, count in counts.items():
        # Convert bitstring to index. Qiskit's get_counts returns strings
        # with qubit-0 as the rightmost char; our oracle used LSB-first
        # ordering, so reverse the string before converting to integer.
        idx = int(bitstring[::-1], 2)
        if idx < N and occupancy[idx] == 1:
            occupied_count += count
    
    estimated_density = occupied_count / shots
    
    return estimated_density, counts


def compute_classical_density(occupancy: List[int]) -> float:
    """
    Compute classical density as occupied_regions / total_regions.
    
    Args:
        occupancy: Binary list where 1 = occupied region.
    
    Returns:
        Density value between 0 and 1.
    """
    if not occupancy:
        return 0.0
    return sum(occupancy) / len(occupancy)


def print_occupancy_grid(occupancy: List[int], rows: int, cols: int) -> None:
    """
    Print the occupancy grid in a visual format.
    
    Args:
        occupancy: Binary occupancy list.
        rows: Number of rows.
        cols: Number of columns.
    """
    print("\nOccupancy Grid (■ = occupied, □ = empty):")
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            idx = r * cols + c
            row_str += "■ " if occupancy[idx] == 1 else "□ "
        print(f"  {row_str}")


def main():
    """
    Demo: Estimate traffic density from mock YOLO boxes using Grover's algorithm.
    """
    # Configuration
    rows = 8
    cols = 8
    image_w = 1920
    image_h = 1080
    n_iterations = 1  # Number of Grover iterations
    shots = 1024
    
    N = rows * cols  # Should be 64 (power of 2)
    n_qubits = int(math.log2(N))
    
    print("=" * 60)
    print("Grover-Based Traffic Density Estimation Demo")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Grid size: {rows} x {cols} = {N} regions")
    print(f"  Qubits: {n_qubits}")
    print(f"  Image size: {image_w} x {image_h}")
    print(f"  Grover iterations: {n_iterations}")
    print(f"  Measurement shots: {shots}")
    
    # Mock YOLO bounding boxes (x1, y1, x2, y2) for detected cars
    # These are scattered across the image to simulate traffic
    mock_boxes = [
        # Top-left area
        (100, 100, 300, 250),
        (350, 80, 500, 200),
        # Middle area
        (600, 400, 800, 550),
        (850, 350, 1000, 500),
        (1100, 450, 1300, 600),
        # Bottom area
        (200, 700, 400, 850),
        (500, 750, 700, 900),
        (1400, 800, 1600, 950),
        # Right side
        (1600, 200, 1800, 350),
        (1700, 500, 1900, 680),
    ]
    
    print(f"\nMock YOLO detections: {len(mock_boxes)} cars")
    for i, box in enumerate(mock_boxes):
        print(f"  Car {i+1}: {box}")
    
    # Convert boxes to occupancy grid
    occupancy = boxes_to_occupancy(mock_boxes, rows, cols, image_w, image_h)
    
    # Display the grid
    print_occupancy_grid(occupancy, rows, cols)
    
    # Classical density
    classical_density = compute_classical_density(occupancy)
    num_occupied = sum(occupancy)
    print(f"\nClassical Analysis:")
    print(f"  Occupied regions: {num_occupied} / {N}")
    print(f"  Classical density: {classical_density:.4f} ({classical_density*100:.1f}%)")
    
    # Quantum density estimation
    # Baseline uniform sampling (no Grover iterations) to estimate true density
    uniform_density, uniform_counts = estimate_density_quantum(
        occupancy, n_iterations=0, shots=shots
    )

    # Grover-amplified run
    amplified_prob, amplified_counts = estimate_density_quantum(
        occupancy, n_iterations=n_iterations, shots=shots
    )

    print(f"\nUniform sampling estimated density (n_iterations=0): {uniform_density:.4f} ({uniform_density*100:.1f}%)")
    print(f"Amplified occupied probability (Grover with {n_iterations} iteration(s)): {amplified_prob:.4f} ({amplified_prob*100:.1f}%)")
    
    # Show top measurement counts
    # Show top measurement counts from the amplified run (correct for endianness)
    sorted_counts = sorted(amplified_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 Measurement Results (amplified run, out of {len(amplified_counts)} unique outcomes):")
    for bitstring, count in sorted_counts[:10]:
        idx = int(bitstring[::-1], 2)
        r, c = index_to_rc(idx, cols)
        occ_status = "occupied" if occupancy[idx] == 1 else "empty"
        prob = count / shots
        print(f"  |{bitstring}⟩ (region {idx}, row {r}, col {c}): "
              f"{count}/{shots} = {prob:.3f} [{occ_status}]")
    
    # Comparison
    print(f"\n" + "=" * 60)
    print("Comparison Summary:")
    print("=" * 60)
    print(f"  Classical density:          {classical_density:.4f}")
    print(f"  Uniform sampling density:   {uniform_density:.4f}")
    print(f"  Amplified occupied prob.:   {amplified_prob:.4f}")
    print(f"  Difference (uniform - classical): {abs(uniform_density - classical_density):.4f}")
    
    # Note about Grover amplification
    print(f"\nNote: With M={num_occupied} marked states out of N={N},")
    optimal_iterations = int(round((math.pi / 4) * math.sqrt(N / max(num_occupied, 1))))
    print(f"      optimal Grover iterations ≈ {optimal_iterations}")
    print(f"      (Using {n_iterations} iteration(s) in this demo)")


if __name__ == "__main__":
    main()
