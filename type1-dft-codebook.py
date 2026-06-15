import numpy as np

def generate_type1_codebook_1d(NT, O1):
    """
    Generate Type I oversampled DFT codebook for a ULA.
    
    Parameters
    ----------
    NT : int
        Number of transmit antennas.
    O1 : int
        Oversampling factor.
    
    Returns
    -------
    codebook : np.ndarray, shape (NT, NT*O1)
        Each column is a unit-norm DFT beam vector.
    """
    num_beams = NT * O1
    n = np.arange(NT)
    l = np.arange(num_beams)
    codebook = np.exp(1j * 2 * np.pi * np.outer(n, l) / num_beams) / np.sqrt(NT)
    return codebook


def generate_type1_codebook_2d(N1, N2, O1, O2):
    """
    Generate Type I oversampled DFT codebook for a UPA via Kronecker product.
    v_{l,m} = a_l ⊗ u_m
    
    Parameters
    ----------
    N1, N2 : int
        Horizontal and vertical antenna counts.
    O1, O2 : int
        Horizontal and vertical oversampling factors.
    
    Returns
    -------
    codebook : np.ndarray, shape (N1*N2, N1*O1*N2*O2)
        Each column is a unit-norm 2D DFT beam vector.
    beam_indices : list of tuples (l, m)
    """
    NT = N1 * N2
    horiz = np.exp(1j * 2 * np.pi * np.outer(np.arange(N1), np.arange(N1 * O1)) / (N1 * O1))
    vert = np.exp(1j * 2 * np.pi * np.outer(np.arange(N2), np.arange(N2 * O2)) / (N2 * O2))
    
    num_beams = N1 * O1 * N2 * O2
    codebook = np.zeros((NT, num_beams), dtype=complex)
    beam_indices = []
    
    col = 0
    for l in range(N1 * O1):
        for m in range(N2 * O2):
            codebook[:, col] = np.kron(horiz[:, l], vert[:, m])
            beam_indices.append((l, m))
            col += 1
    
    codebook = codebook / np.sqrt(NT)
    return codebook, beam_indices


def beamforming_search(codebook, h):
    """
    Find the best codeword maximizing |h^H f|^2.
    
    Parameters
    ----------
    codebook : np.ndarray, shape (NT, num_codewords)
    h : np.ndarray, shape (NT,)
    
    Returns
    -------
    best_idx : int
    best_gain : float
    """
    h = h.flatten()
    gains = np.abs(h.conj() @ codebook) ** 2
    best_idx = np.argmax(gains)
    return best_idx, gains[best_idx]


def verify_orthogonality(NT, O1):
    """
    Verify orthogonal group structure of oversampled DFT codebook.
    """
    C = generate_type1_codebook_1d(NT, O1)
    G = C.conj().T @ C
    
    print(f"NT = {NT}, O1 = {O1}, total beams = {NT * O1}")
    print(f"Orthogonal groups: {O1}, beams per group: {NT}\n")
    
    for q in range(O1):
        group = [q + g * O1 for g in range(NT)]
        subG = G[np.ix_(group, group)]
        off_diag = np.abs(subG) - np.eye(NT)
        print(f"Group q1={q}: indices {group}, max off-diag = {np.max(np.abs(off_diag)):.2e}")
    
    print(f"\nCross-group |<a_0, a_1>| = {np.abs(G[0, 1]):.4f}")
    print(f"Within-group |<a_0, a_{O1}>| = {np.abs(G[0, O1]):.2e}")


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    
    # --- 1) Orthogonality check ---
    print("=" * 60)
    print("ORTHOGONALITY VERIFICATION")
    print("=" * 60)
    verify_orthogonality(NT=4, O1=4)
    
    # --- 2) 1D codebook ---
    print("\n" + "=" * 60)
    print("1D CODEBOOK")
    print("=" * 60)
    NT, O1 = 4, 4
    C = generate_type1_codebook_1d(NT, O1)
    print(f"Shape: {C.shape}  (NT={NT} x {NT*O1} beams)")
    print(f"Column norm: {np.linalg.norm(C[:, 0]):.4f}")
    print(f"Feedback bits: {int(np.ceil(np.log2(NT * O1)))}")
    
    # --- 3) 2D UPA codebook ---
    print("\n" + "=" * 60)
    print("2D UPA CODEBOOK")
    print("=" * 60)
    N1, N2, O1, O2 = 4, 2, 4, 4
    C_2d, idx = generate_type1_codebook_2d(N1, N2, O1, O2)
    NT = N1 * N2
    print(f"Config: N1={N1}, N2={N2}, NT={NT}")
    print(f"Shape: {C_2d.shape}  ({NT} x {N1*O1*N2*O2} beams)")
    print(f"Feedback bits: {int(np.ceil(np.log2(N1*O1*N2*O2)))}")
    
    # --- 4) Beamforming gain vs oversampling ---
    print("\n" + "=" * 60)
    print("BEAMFORMING GAIN: DFT vs MRT (Rayleigh fading)")
    print("=" * 60)
    NT = 4
    num_trials = 1000
    
    gains = {1: [], 2: [], 4: []}
    gains_mrt = []
    gains_egt = []
    
    for _ in range(num_trials):
        h = (np.random.randn(NT) + 1j * np.random.randn(NT)) / np.sqrt(2)
        
        # MRT gain: ||h||^2
        gains_mrt.append(np.linalg.norm(h) ** 2)
        
        # EGT gain: (sum |h_i|)^2 / NT
        gains_egt.append(np.sum(np.abs(h)) ** 2 / NT)
        
        # DFT codebook gain for various O1
        f_opt = h.conj() / np.linalg.norm(h)  # optimal beamforming vector
        for O1 in [1, 2, 4]:
            C = generate_type1_codebook_1d(NT, O1)
            _, g = beamforming_search(C, f_opt)
            gains[O1].append(g)
    
    print(f"NT = {NT}, {num_trials} Rayleigh trials\n")
    print(f"{'Method':<20} {'Avg Gain (dB)':>12}  {'Bits':>4}")
    print("-" * 40)
    print(f"{'MRT (perfect CSI)':<20} {10*np.log10(np.mean(gains_mrt)):>12.2f}  {'--':>4}")
    print(f"{'EGT (perfect CSI)':<20} {10*np.log10(np.mean(gains_egt)):>12.2f}  {'--':>4}")
    for O1 in [1, 2, 4]:
        avg = np.mean(gains[O1])
        bits = int(np.ceil(np.log2(NT * O1)))
        print(f"{'DFT O1=' + str(O1):<20} {10*np.log10(avg):>12.2f}  {bits:>4}")
