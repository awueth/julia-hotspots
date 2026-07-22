# Finite-dimensional pipeline

Computes the `d = 10^18` certificate values (the table at the end of
`writeup/certificate.typ`). Two phases: phase 2 builds the fit once, phase 3 is
the tuning loop. There is **no phase 1** — finite_dim reuses the shared
log_concave potential artifact
`checkpoints/log_concave_extension/high-resolution/lse_global_potential.chk` and
the log_concave results (infinite-dimensional eigenvalue bounds + potential
constants), which it scales to finite `d` via the comparison theorem
(`writeup/barrel.typ`).

> Rebuild the fit (phase 2) whenever the log_concave potential is rebuilt — a
> stale fit no longer satisfies the boundary condition.

## Phase 2 — eigenfunction (build once)

```sh
julia --project=. src/results/finite_dim/build_eigenfunction.jl
```

Fits the finite-`d` MPS candidate and serializes it to
`checkpoints/first_eigenfunction_finite_dim.chk`. Reads the `[fit]` table.

## Phase 3 — bounds (tuning loop)

```sh
# tuning: coarse grids + disk cache (fast iteration)
julia --project=. src/results/finite_dim/run.jl

# proof: no cache, everything recomputed fresh
julia --project=. src/results/finite_dim/run.jl --final
```

Writes `writeup/results/finite_dim/finite-dim/summary.toml` (the only file the writeup
reads). Read [`bounds.jl`](bounds.jl) top-to-bottom to trace every value.

The finite-dim specifics vs the log-concave pipeline:

- **Residual** `‖∂ₙφ*‖∞` — the finite-`d` unit normal is normalized by
  `sqrt(4d + ‖∇V‖²)` (applied in the solver's finite-`d` `boundary_residual`).
- **Eigenvalue bounds** — the log_concave `{λ₁,λ₂}` endpoints scaled by the
  comparison multipliers `(1±ε)²`, `β`.
- **L∞ norm** `‖φ*‖∞` — a raw floating-point grid sample (no normalization; the
  hot-spot criterion `H(φ*) > 2E(φ*)` is scale-invariant, see the remark after
  `writeup/pointwise.typ` @thm:pointwise-limit).
- **Ultracontractivity** — the log-space finite-`d` barrier / correction
  constants (the module's float versions overflow at `d = 1e18`). The wing mass
  beyond the configured cutoff `Lx + buffer_delta` is enclosed once by adaptive
  interval quadrature and reused throughout the time-split optimization.

### Caching (tuning only)

The expensive interval quadrature (`first_constant` across the time grid) is
isolated as an `*_impl` function in `bounds.jl` and wrapped by
[`caches.jl`](caches.jl) under `cache/finite_dim/`. The adaptive buffered-wing
quadrature is cheap enough to recompute once on every run. Because tuning
knobs like `decay_split` don't feed `first_constant`, editing them reuses every
quadrature. The `--final` run reads no cache. After editing a quadrature body or
rebuilding an artifact, wipe the cache first:

```julia
include("caches.jl"); reset_caches!()
```

## Files

| file | role |
|------|------|
| `build_eigenfunction.jl` | phase 2 |
| `bounds.jl`              | phase 3 math (trace quantities here) |
| `caches.jl`             | disk caches for the phase-3 quadratures |
| `run.jl`                | phase 3 entry point + `summary.toml` assembly |
| `finite-dim.toml`       | flat parameters |
