"""
Numerically Optimized Grassmannian Codebook via Alternating Projection
=======================================================================
Generates rank-1 Grassmannian codebooks for MISO beamforming using the
alternating projection method (Dhillon, Heath, Strohmer, Tropp, IEEE TSP 2008).

Simulates average beamforming gain for:
  - Numerically optimized Grassmannian codebook (this script)
  - EGT baseline (closed form)

Results are printed as a table so they can be overlaid on Figure 2 of the
TCOM paper (average gain vs CP code dimension k, Rayleigh MISO).

Usage:
    sage grassmannian_altproj.sage

Dependencies: numpy, scipy (available in SageMath's bundled Python)
"""

import numpy as np
from numpy.linalg import svd, norm
import itertools

# -----------------------------------------------------------------------
# Parameters — match your paper's Figure 2 setup
# -----------------------------------------------------------------------
NT_LIST     = [4, 6, 10]       # transmit antenna counts
N_SIM       = 1000             # Monte Carlo trials (increase for smoother curves)
SEED        = 42

# Codebook sizes to evaluate — chosen to match CP feedback bits B = k*log2(p)
# For NT=4,6,10 with p=5, k=2,3,4 gives |CP| = 25, 125, 625
# We match these sizes for the Grassmannian codebook
CODEBOOK_SIZES = [25, 125, 625]   # adjust to match your CP codebook sizes

# Alternating projection hyperparameters
AP_ITERS    = 200   # number of outer iterations
AP_RESTARTS = 5     # random restarts (take best result)

# -----------------------------------------------------------------------
# Alternating Projection Grassmannian Codebook
# -----------------------------------------------------------------------

def random_unit_codebook(n, size, rng):
    """Generate a random codebook of unit-norm complex vectors in C^n."""
    C = rng.standard_normal((n, size)) + 1j * rng.standard_normal((n, size))
    C = C / norm(C, axis=0, keepdims=True)
    return C

def grassmannian_coherence(C):
    """Compute max |<c_i, c_j>|^2 over all i != j (coherence / chordal packing metric)."""
    G = np.abs(C.conj().T @ C) ** 2
    np.fill_diagonal(G, 0.0)
    return G.max()

def alternating_projection(n, size, n_iters=200, rng=None):
    """
    Alternating projection to minimize coherence of a complex codebook.

    Dhillon, Heath, Strohmer, Tropp,
    'Constructing Packings in Grassmannian Manifolds via Alternating Projection'
    IEEE Trans. Information Theory, 2008.

    Each iteration:
      1. Form Gram matrix G = C^H C
      2. Project G onto the set of matrices with |G_ij| <= tau (off-diagonal)
         where tau is the Welch bound sqrt((size-n)/(n*(size-1)))
      3. Project back onto the set of Gram matrices of a unit-norm frame
         (via eigendecomposition, keep top-n eigenvalues)
      4. Extract new codebook columns from the factored Gram matrix
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Welch bound on coherence (theoretical lower bound)
    welch = np.sqrt((size - n) / (n * (size - 1)))

    C = random_unit_codebook(n, size, rng)
    best_C = C.copy()
    best_coh = grassmannian_coherence(C)

    for _ in range(n_iters):
        # Step 1: Gram matrix
        G = C.conj().T @ C   # (size x size)

        # Step 2: Shrink off-diagonal entries to magnitude <= welch
        G_proj = G.copy()
        diag_idx = np.arange(size)
        mask = np.ones((size, size), dtype=bool)
        mask[diag_idx, diag_idx] = False
        off_diag_abs = np.abs(G_proj[mask])
        # Shrink: keep phase, cap magnitude
        phases = np.angle(G_proj[mask])
        magnitudes = np.minimum(off_diag_abs, welch)
        G_proj[mask] = magnitudes * np.exp(1j * phases)
        # Ensure Hermitian
        G_proj = (G_proj + G_proj.conj().T) / 2

        # Step 3: Project onto PSD rank-n matrices with unit diagonal
        # Eigendecompose, keep top n eigenvalues
        eigvals, eigvecs = np.linalg.eigh(G_proj)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals_trunc = np.maximum(eigvals[:n], 0)
        G_rank_n = eigvecs[:, :n] @ np.diag(eigvals_trunc) @ eigvecs[:, :n].conj().T

        # Normalize diagonal to 1
        d = np.sqrt(np.diag(G_rank_n).real)
        d[d < 1e-12] = 1e-12
        G_rank_n = G_rank_n / np.outer(d, d)

        # Step 4: Extract codebook from Gram matrix
        # C = sqrt(Lambda) * V^H  (size x n) -> take transpose for (n x size)
        eigvals2, eigvecs2 = np.linalg.eigh(G_rank_n)
        idx2 = np.argsort(eigvals2)[::-1]
        eigvals2 = np.maximum(eigvals2[idx2], 0)[:n]
        eigvecs2 = eigvecs2[:, idx2][:, :n]
        C = (eigvecs2 * np.sqrt(eigvals2)).T   # (n x size)
        # Re-normalize columns
        col_norms = norm(C, axis=0, keepdims=True)
        col_norms[col_norms < 1e-12] = 1.0
        C = C / col_norms

        coh = grassmannian_coherence(C)
        if coh < best_coh:
            best_coh = coh
            best_C = C.copy()

    return best_C, best_coh, welch

def build_grassmannian_codebook(n, size, n_iters=200, n_restarts=5, seed=42):
    """Run alternating projection with multiple restarts, return best codebook."""
    rng = np.random.default_rng(seed)
    best_C, best_coh, welch = None, np.inf, None
    for r in range(n_restarts):
        C, coh, wb = alternating_projection(n, size, n_iters=n_iters, rng=rng)
        if coh < best_coh:
            best_coh = coh
            best_C = C.copy()
            welch = wb
        print(f"  Restart {r+1}/{n_restarts}: coherence = {coh:.6f}  (Welch bound = {wb:.6f})")
    return best_C, best_coh, welch

# -----------------------------------------------------------------------
# Beamforming Gain Simulation
# -----------------------------------------------------------------------

def egt_gain_miso(h):
    """
    EGT gain for MISO: (||h||_1)^2 / NT
    h: complex row vector (1 x NT)
    """
    return (np.sum(np.abs(h)))**2 / len(h)

def codebook_gain_miso(h, C):
    """
    Beamforming gain using codebook C for channel h.
    C: (NT x M) complex matrix, columns are unit-norm codewords
    h: (NT,) complex vector
    Returns: max over codewords of |h @ c|^2
    """
    gains = np.abs(h @ C) ** 2
    return gains.max()

def simulate_rayleigh_miso(NT, C, n_sim=1000, seed=42):
    """
    Average beamforming gain (dB) for i.i.d. Rayleigh MISO.
    Returns: avg_gain_codebook_dB, avg_gain_egt_dB
    """
    rng = np.random.default_rng(seed)
    gains_cb  = np.zeros(n_sim)
    gains_egt = np.zeros(n_sim)

    for i in range(n_sim):
        h = (rng.standard_normal(NT) + 1j * rng.standard_normal(NT)) / np.sqrt(2)
        gains_cb[i]  = codebook_gain_miso(h, C)
        gains_egt[i] = egt_gain_miso(h)

    avg_cb  = 10 * np.log10(np.mean(gains_cb))
    avg_egt = 10 * np.log10(np.mean(gains_egt))
    return avg_cb, avg_egt

# -----------------------------------------------------------------------
# Main: build codebooks and simulate for each (NT, codebook_size) pair
# -----------------------------------------------------------------------

print("=" * 65)
print("Grassmannian Codebook via Alternating Projection — MISO Rayleigh")
print("=" * 65)
print(f"Monte Carlo trials : {N_SIM}")
print(f"AP iterations      : {AP_ITERS}")
print(f"AP restarts        : {AP_RESTARTS}")
print()

results = []  # (NT, size, gain_cb_dB, gain_egt_dB, coherence, welch_bound)

for NT in NT_LIST:
    for size in CODEBOOK_SIZES:
        B_bits = int(np.ceil(np.log2(size)))
        print(f"NT={NT}, |C|={size} (B≈{B_bits} bits)")
        print(f"  Building Grassmannian codebook (n={NT}, size={size})...")
        C, coh, welch = build_grassmannian_codebook(
            NT, size,
            n_iters=AP_ITERS,
            n_restarts=AP_RESTARTS,
            seed=SEED
        )
        print(f"  Best coherence = {coh:.6f}  (Welch bound = {welch:.6f})")

        print(f"  Simulating beamforming gain ({N_SIM} trials)...")
        gain_cb, gain_egt = simulate_rayleigh_miso(NT, C, n_sim=N_SIM, seed=SEED)
        print(f"  Avg gain (Grassmannian) = {gain_cb:.4f} dB")
        print(f"  Avg gain (EGT)          = {gain_egt:.4f} dB")
        print()
        results.append((NT, size, B_bits, gain_cb, gain_egt, coh, welch))

# -----------------------------------------------------------------------
# Print summary table
# -----------------------------------------------------------------------
print("=" * 65)
print("SUMMARY TABLE")
print(f"{'NT':>4} {'|C|':>6} {'B(bits)':>8} {'Grass(dB)':>12} {'EGT(dB)':>10} {'Coherence':>12} {'Welch':>10}")
print("-" * 65)
for (NT, size, B, g_cb, g_egt, coh, wb) in results:
    print(f"{NT:>4} {size:>6} {B:>8} {g_cb:>12.4f} {g_egt:>10.4f} {coh:>12.6f} {wb:>10.6f}")
print("=" * 65)

print()
print("Copy the 'Grass(dB)' values into your Figure 2 plot as the")
print("'Grassmannian (Alt. Proj.)' reference curve.")
print()
print("Suggested figure caption addition:")
print("  'The numerically optimized Grassmannian codebook (alternating")
print("   projection [9]) serves as an upper bound for structured designs.'")