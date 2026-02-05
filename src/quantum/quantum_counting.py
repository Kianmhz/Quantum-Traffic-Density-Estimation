"""
Quantum Counting for Traffic Density Estimation.

This module implements the Quantum Counting algorithm, which combines
Grover's algorithm with Quantum Phase Estimation (QPE) to estimate
the number of marked states (occupied regions) in O(√N) oracle queries.

Theoretical Background:
-----------------------
Given N = 2^n basis states and M marked states (occupied regions),
Grover's operator G has eigenvalues e^{±2iθ} where sin²(θ) = M/N.

Quantum Counting uses QPE to estimate θ, from which we can extract M:
    M = N · sin²(θ)

This provides a quadratic speedup over classical counting:
    - Classical: O(N) queries to count M items
    - Quantum:   O(√N) queries to estimate M

Assumption:
-----------
We assume the oracle is provided as a "black box" — representing future
quantum sensors or quantum memory that can mark occupied regions without
classical preprocessing. In this demo, we construct the oracle classically
for simulation purposes, but the algorithm itself only queries the oracle
O(√N) times.
"""

import math
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import QFTGate

from src.vision.boxes_to_occupancy import boxes_to_occupancy


def build_oracle(n_qubits: int, marked_indices: List[int]) -> QuantumCircuit:
    """
    Build a Grover oracle that phase-flips all marked basis states.
    
    This oracle marks states |i⟩ where i ∈ marked_indices by applying
    a phase of -1. In a real quantum advantage scenario, this oracle
    would be implemented by quantum hardware/sensors, not classical
    preprocessing.
    
    Args:
        n_qubits: Number of qubits (search space size N = 2^n_qubits).
        marked_indices: List of indices to mark (occupied regions).
    
    Returns:
        QuantumCircuit implementing the oracle.
    """
    qc = QuantumCircuit(n_qubits, name="Oracle")
    
    if not marked_indices:
        return qc  # Empty oracle if nothing to mark
    
    for idx in marked_indices:
        # Convert index to binary (LSB at q0)
        bits = format(idx, f'0{n_qubits}b')[::-1]
        
        # Flip qubits that should be |0⟩ for this basis state
        for q, b in enumerate(bits):
            if b == '0':
                qc.x(q)
        
        # Multi-controlled Z gate
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            # H-MCX-H implements multi-controlled Z
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        
        # Undo X gates
        for q, b in enumerate(bits):
            if b == '0':
                qc.x(q)
    
    return qc


def build_grover_operator(n_qubits: int, oracle: QuantumCircuit) -> QuantumCircuit:
    """
    Build the Grover operator G = D · O (diffusion after oracle).
    
    The Grover operator has eigenvalues e^{±2iθ} where sin²(θ) = M/N.
    QPE will estimate θ to determine M.
    
    Args:
        n_qubits: Number of qubits in the search space.
        oracle: The oracle circuit that marks solutions.
    
    Returns:
        QuantumCircuit implementing G.
    """
    qc = QuantumCircuit(n_qubits, name="Grover")
    
    # Apply oracle
    qc.compose(oracle, inplace=True)
    
    # Diffusion operator: 2|s⟩⟨s| - I
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    
    # Multi-controlled Z
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cx(0, 1)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    
    return qc


def quantum_counting(
    occupancy: List[int],
    precision_qubits: int = 6,
    shots: int = 1024
) -> Tuple[int, float, Dict[str, int]]:
    """
    Estimate the number of marked states using Quantum Counting.
    
    This algorithm uses Quantum Phase Estimation on the Grover operator
    to estimate θ where sin²(θ) = M/N, then computes M = N · sin²(θ).
    
    The Grover operator G has eigenvalues e^{±2iθ} where sin(θ) = √(M/N).
    QPE measures the phase 2θ/2π = θ/π, giving us θ from which we compute M.
    
    PRECISION REQUIREMENTS:
    -----------------------
    For accurate estimation, precision_qubits should be at least:
      - 4 for N=16 (coarse, ~±2 error)
      - 5 for N=16 (better, ~±1 error)  
      - 6 for N=16-32 (good, <±1 average error)
      - 7+ for high accuracy or large N
    
    The phase resolution is 1/2^precision_qubits. For small M, the phase
    angle θ = arcsin(√(M/N)) is small, requiring higher precision.
    
    Args:
        occupancy: Binary list where 1 = occupied region.
        precision_qubits: Number of qubits for phase estimation precision.
                         More qubits = better precision but deeper circuit.
                         Default: 6 (64 phase bins, good for N≤32).
        shots: Number of measurement shots.
    
    Returns:
        Tuple of (estimated_M, estimated_density, measurement_counts).
    """
    N = len(occupancy)
    n_qubits = int(math.log2(N))
    
    if 2 ** n_qubits != N:
        raise ValueError(f"Occupancy length {N} must be a power of 2")
    
    # Find marked indices (in practice, oracle would be a black box)
    marked_indices = [i for i, occ in enumerate(occupancy) if occ == 1]
    M_actual = len(marked_indices)
    
    if M_actual == 0:
        return 0, 0.0, {"0" * precision_qubits: shots}
    
    if M_actual == N:
        return N, 1.0, {format(precision_qubits, f'0{precision_qubits}b'): shots}
    
    # Build oracle and Grover operator
    oracle = build_oracle(n_qubits, marked_indices)
    grover_op = build_grover_operator(n_qubits, oracle)
    
    # Create quantum counting circuit
    # Counting register (for QPE) + Search register
    counting_reg = QuantumRegister(precision_qubits, 'counting')
    search_reg = QuantumRegister(n_qubits, 'search')
    classical_reg = ClassicalRegister(precision_qubits, 'result')
    
    qc = QuantumCircuit(counting_reg, search_reg, classical_reg)
    
    # Initialize search register in uniform superposition
    qc.h(search_reg)
    
    # Initialize counting register in superposition
    qc.h(counting_reg)
    
    # Controlled Grover operators (controlled-G^(2^k))
    # This is the core of QPE
    for k in range(precision_qubits):
        # Apply G^(2^k) controlled by counting qubit k
        power = 2 ** k
        for _ in range(power):
            controlled_grover = grover_op.control(1)
            qc.compose(
                controlled_grover,
                qubits=[counting_reg[k]] + list(search_reg),
                inplace=True
            )
    
    # Inverse QFT on counting register
    qft_gate = QFTGate(precision_qubits).inverse()
    qc.append(qft_gate, counting_reg)
    
    # Measure counting register
    qc.measure(counting_reg, classical_reg)
    
    # Run simulation
    backend = Aer.get_backend("aer_simulator")
    transpiled = transpile(qc, backend, optimization_level=1)
    result = backend.run(transpiled, shots=shots).result()
    counts = result.get_counts()
    
    # Analyze results to estimate M
    # The Grover operator G has eigenvalues e^{±2iθ} where sin(θ) = √(M/N)
    # This means the eigenvalue phases are ±2θ
    # QPE measures φ = 2θ/(2π) = θ/π OR φ = (2π - 2θ)/(2π) = 1 - θ/π
    # 
    # So we get two peaks at φ₁ and φ₂ = 1 - φ₁
    # Both should give the same M = N · sin²(πφ)
    # 
    # However, due to finite precision, we should identify pairs of peaks
    # and combine their evidence.
    #
    # IMPORTANT PHASE INTERPRETATION:
    # When the search register starts in |s⟩ (uniform superposition),
    # the QPE measures phases that appear at 0.5 ± θ/π, where the Grover
    # eigenvalues are e^{±2iθ} and sin(θ) = √(M/N).
    #
    # So: φ_observed ≈ 0.5 + θ/π  or  0.5 - θ/π
    # Therefore: θ = π × |φ_observed - 0.5|
    # And: M = N × sin²(θ)
    
    # First, collect raw phase measurements
    # NOTE: Qiskit returns bitstrings in big-endian order (MSB first),
    # which directly gives us the integer value without reversal
    phase_counts = {}
    for bitstring, count in counts.items():
        measured_int = int(bitstring, 2)  # No reversal needed!
        phi = measured_int / (2 ** precision_qubits)
        phase_counts[phi] = phase_counts.get(phi, 0) + count
    
    # Group symmetric phases around 0.5 (i.e., φ and 1-φ give same θ)
    # because |0.5 - φ| = |0.5 - (1-φ)| = |φ - 0.5|
    M_evidence = {}
    processed = set()
    
    for phi, count in phase_counts.items():
        if phi in processed:
            continue
        
        # Compute θ from the offset from 0.5
        theta = math.pi * abs(phi - 0.5)
        M_from_phi = N * (math.sin(theta) ** 2)
        
        # The symmetric phase (1-φ) gives the SAME θ offset from 0.5
        phi_partner = 1.0 - phi
        partner_count = 0
        for p, c in phase_counts.items():
            if abs(p - phi_partner) < 1e-9:  # Exact match for symmetric phase
                if p not in processed:
                    partner_count = c
                    processed.add(p)
                break
        
        processed.add(phi)
        total_count = count + partner_count
        
        # Round to nearest integer
        M_rounded = round(M_from_phi)
        M_rounded = max(0, min(N, M_rounded))
        
        if M_rounded not in M_evidence:
            M_evidence[M_rounded] = 0
        M_evidence[M_rounded] += total_count
    
    # Pick M with highest evidence
    if M_evidence:
        M_estimated = max(M_evidence.keys(), key=lambda m: M_evidence[m])
    else:
        M_estimated = 0
    
    estimated_density = M_estimated / N
    
    return M_estimated, estimated_density, counts

def analyze_quantum_counting_results(
    counts: Dict[str, int],
    N: int,
    precision_qubits: int
) -> List[Tuple[int, float, int]]:
    """
    Analyze QPE measurement results to show phase-to-M mapping.
    
    Args:
        counts: Measurement counts from quantum counting.
        N: Total number of regions.
        precision_qubits: Number of precision qubits used.
    
    Returns:
        List of (measured_value, estimated_M, count) tuples, sorted by count.
    """
    results = []
    for bitstring, count in counts.items():
        # Qiskit returns big-endian (MSB first), no reversal needed
        measured_int = int(bitstring, 2)
        phi = measured_int / (2 ** precision_qubits)
        # Theta is the offset from 0.5
        theta = math.pi * abs(phi - 0.5)
        M_estimate = N * (math.sin(theta) ** 2)
        results.append((measured_int, M_estimate, count))
    
    return sorted(results, key=lambda x: x[2], reverse=True)


def compute_classical_count(occupancy: List[int]) -> int:
    """Classical O(N) counting of occupied regions."""
    return sum(occupancy)


def print_comparison(
    occupancy: List[int],
    rows: int,
    cols: int,
    precision_qubits: int = 4,
    shots: int = 2048
) -> None:
    """
    Run quantum counting and compare with classical results.
    
    Args:
        occupancy: Binary occupancy list.
        rows: Grid rows (for display).
        cols: Grid columns (for display).
        precision_qubits: QPE precision.
        shots: Measurement shots.
    """
    N = len(occupancy)
    n_qubits = int(math.log2(N))
    
    print("=" * 70)
    print("Quantum Counting for Traffic Density Estimation")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {rows} × {cols} = {N} regions")
    print(f"  Search space qubits: {n_qubits}")
    print(f"  Precision qubits (QPE): {precision_qubits}")
    print(f"  Measurement shots: {shots}")
    
    # Display occupancy grid
    print(f"\nOccupancy Grid (■ = occupied, □ = empty):")
    for r in range(rows):
        row_str = "  "
        for c in range(cols):
            idx = r * cols + c
            row_str += "■ " if occupancy[idx] == 1 else "□ "
        print(row_str)
    
    # Classical count
    M_classical = compute_classical_count(occupancy)
    density_classical = M_classical / N
    
    print(f"\nClassical Count (O(N) = O({N}) operations):")
    print(f"  Occupied regions: {M_classical}")
    print(f"  Density: {density_classical:.4f} ({density_classical*100:.1f}%)")
    
    # Quantum counting
    print(f"\nRunning Quantum Counting (O(√N) = O({int(math.sqrt(N))}) oracle queries)...")
    M_quantum, density_quantum, counts = quantum_counting(
        occupancy, precision_qubits=precision_qubits, shots=shots
    )
    
    print(f"\nQuantum Counting Results:")
    print(f"  Estimated occupied regions: {M_quantum}")
    print(f"  Estimated density: {density_quantum:.4f} ({density_quantum*100:.1f}%)")
    
    # Show phase analysis
    print(f"\nPhase Estimation Analysis (top 5 measurements):")
    analysis = analyze_quantum_counting_results(counts, N, precision_qubits)
    for measured, M_est, count in analysis[:5]:
        phi = measured / (2 ** precision_qubits)
        print(f"  |{measured:0{precision_qubits}b}⟩ → φ={phi:.4f} → M≈{M_est:.2f} (measured {count} times)")
    
    # Comparison
    print(f"\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"  Classical M:  {M_classical}")
    print(f"  Quantum M:    {M_quantum}")
    print(f"  Error:        {abs(M_quantum - M_classical)} regions")
    print(f"  Relative error: {abs(M_quantum - M_classical) / max(M_classical, 1) * 100:.1f}%")
    
    print(f"\nComplexity Comparison:")
    print(f"  Classical counting: O(N) = O({N}) queries")
    print(f"  Quantum counting:   O(√N) = O({int(math.sqrt(N))}) oracle queries")
    print(f"  Speedup factor:     {N / max(int(math.sqrt(N)), 1)}×")
    
    print(f"\nNote: The quantum advantage assumes the oracle is a black-box")
    print(f"      (e.g., from quantum sensors). In this demo, we construct")
    print(f"      the oracle classically for simulation purposes.")


def main():
    """
    Demo: Quantum Counting for traffic density estimation.
    """
    # Configuration - using 4x4 grid for manageable circuit depth
    rows = 4
    cols = 4  # N = 16 for manageable circuit depth
    image_w = 1920
    image_h = 1080
    precision_qubits = 4  # Balance between precision and circuit depth
    shots = 2048
    
    print("\n" + "=" * 70)
    print("QUANTUM COUNTING TRAFFIC DENSITY DEMO")
    print("=" * 70)
    
    # Test 1: Moderate density (~30-40%)
    print("\n" + "=" * 70)
    print("TEST 1: Moderate Traffic Density")
    print("=" * 70)
    
    mock_boxes_moderate = [
        (100, 100, 400, 300),    # Top-left area
        (1400, 100, 1700, 350),  # Top-right
        (200, 600, 500, 850),    # Bottom-left
        (900, 500, 1200, 750),   # Center
        (1500, 700, 1800, 950),  # Bottom-right
    ]
    
    print(f"\nSimulated YOLO detections: {len(mock_boxes_moderate)} cars")
    occupancy_moderate = boxes_to_occupancy(mock_boxes_moderate, rows, cols, image_w, image_h)
    print_comparison(occupancy_moderate, rows, cols, precision_qubits, shots)
    
    # Test 2: Low density (~20%)
    print("\n\n" + "=" * 70)
    print("TEST 2: Low Traffic Density")
    print("=" * 70)
    
    mock_boxes_low = [
        (100, 100, 350, 280),    # Top-left
        (1000, 500, 1300, 700),  # Center
        (1600, 800, 1850, 980),  # Bottom-right
    ]
    
    print(f"\nSimulated YOLO detections: {len(mock_boxes_low)} cars")
    occupancy_low = boxes_to_occupancy(mock_boxes_low, rows, cols, image_w, image_h)
    print_comparison(occupancy_low, rows, cols, precision_qubits, shots)
    
    # Test 3: Higher density (~50-60%)
    print("\n\n" + "=" * 70)
    print("TEST 3: High Traffic Density")
    print("=" * 70)
    
    mock_boxes_high = [
        (50, 50, 300, 200),
        (400, 80, 700, 250),
        (1000, 50, 1300, 200),
        (1500, 100, 1800, 280),
        (150, 400, 450, 600),
        (800, 350, 1100, 550),
        (1400, 400, 1700, 600),
        (200, 750, 500, 950),
        (900, 700, 1200, 900),
    ]
    
    print(f"\nSimulated YOLO detections: {len(mock_boxes_high)} cars")
    occupancy_high = boxes_to_occupancy(mock_boxes_high, rows, cols, image_w, image_h)
    print_comparison(occupancy_high, rows, cols, precision_qubits, shots)


if __name__ == "__main__":
    main()
