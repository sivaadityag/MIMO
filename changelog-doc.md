# Bug Fixes and Corrections to CP Code Implementation

## Summary

The original SageMath implementation of Character-Polynomial (CP) codes contained parameter convention mismatches with respect to the definitions in [Riasat–Mahdavifar, ISIT 2024]. These have been corrected as described below.

## Changes

### 1. Parameter Convention: `rs_dim` → `k`

- **Old:** The constructor parameter `rs_dim` was defined as $k + 1$, where $k$ is the degree of the CP message polynomial (Definition 7). This corresponded to the dimension of the larger Reed–Solomon code $\mathrm{RS}(\mathcal{F}(k, q))$ used in the embedding $\mathrm{GRS} \subseteq \mathrm{RS}(\mathcal{F}(k, q))$.
- **New:** The parameter is renamed to `k` and corresponds directly to the paper's $k$, i.e., the maximum degree of the CP message polynomial. The GRS code $\mathrm{GRS}(\mathcal{F}(k-1, q))$ has dimension $k$, which is passed directly to SageMath's `GeneralizedReedSolomonCode` constructor.
- **Impact:** This resolves the constraint violation where `rs_dim` $= k + 1 > n$ for $k = n = q - 1$, which previously caused a `ValueError` in SageMath.

### 2. Generator Matrix Zeroing Logic

- **Old:** The method `__grs_generator_matrix` zeroed out row $i$ whenever $i \bmod p = 0$. This enforced the zero-coefficient pattern of $\mathcal{F}_p(k, q)$, i.e., $f_{jp} = 0$ at indices $0, p, 2p, \ldots$
- **New:** Row $i$ is zeroed whenever $(i + 1) \bmod p = 0$. This correctly reflects the zero-coefficient pattern of $\mathcal{F}_p(k, q)' = \{f(X)/X : f \in \mathcal{F}_p(k, q)\}$. Since $g_i = f_{i+1}$, the forced zeros in $g$ occur at indices $i = p - 1,\; 2p - 1,\; 3p - 1, \ldots$
- **Impact:** For $k < p$, the original code incorrectly zeroed out the constant term of $g$ (row 0), reducing the code dimension by 1. The corrected version produces no spurious zero rows when $k < p$.

### 3. Dimension Formula

- **Old:** `dimension()` computed `rs_dim - rs_dim // p`, effectively evaluating $(k+1) - \lfloor (k+1)/p \rfloor$.
- **New:** `dimension()` computes `k - k // p`, matching Equation (3) of the paper: $\dim(\mathrm{CP}) = k - \lfloor k/p \rfloor$.

### 4. Minimum Distance Formula

- **Old:** `minimum_distance()` returned `field_size - rs_dim` $= q - (k+1) = n - k$.
- **New:** `minimum_distance()` returns `length - k + 1` $= n - k + 1$, consistent with the minimum distance $d = n - k + 1$ of the $[n, k, d]_q$ GRS code (Definition 9, and the discussion following it).

### 5. Degree Checks in `__contains__` and `convert_polynomial`

- **Old:** Checked `c.degree() < self.rs_dim`, i.e., $\deg(c) \leq k$.
- **New:** Checks `c.degree() < self.k + 1`, which is equivalent but uses the corrected parameter name.

## Verification

With $q = p = 5$, $n = 4$, $k = 4$:

| Quantity | Old (incorrect) | New (correct) |
|---|---|---|
| GRS dimension passed to SageMath | $k + 1 = 5 > n$ (error) | $k = 4 \leq n$ (valid) |
| Forced zeros in generator matrix | Row 0 ($g_0$) | None (since $p - 1 = 4 > k - 1 = 3$) |
| Code dimension | 3 | 4 |
| Number of codewords | $5^3 = 125$ | $5^4 = 625$ |
| Minimum distance | $n - k = 0$ | $n - k + 1 = 1$ |
