# Log-concave extension pipeline

Three steps. The first two build the potential and approximate eigenfunction once; the third is to compute the pointwise distance while allowing hyperparameter tuning. 

## Step 1 — potential (build once)

```sh
julia --project=. src/results/log_concave_extension/build_potential.jl
```

Builds the global log-sum-exp potential and its verified normalization / wing
mass, and writes two artifacts that both this pipeline and `finite_dim` consume:

- `checkpoints/log_concave_extension/high-resolution/lse_global_potential.chk`
- `results/log_concave_extension/high-resolution/0-make-potential.toml`
  (the nested `result.potential_constants` table is a contract with `finite_dim`)

## Step 2 — eigenfunction (build once)

```sh
julia --project=. src/results/log_concave_extension/build_eigenfunction.jl
```

Fits the MPS candidate and serializes it to
`checkpoints/log_concave_extension/high-resolution/first_eigenfunction.chk`.

Both build scripts default to `high-resolution.toml` (the shared, canonical
artifacts). Pass another config path as the first argument to override.

## Step 3 — bounds (tuning loop)

```sh
# tuning: coarse grids + disk cache (fast iteration)
julia --project=. src/results/log_concave_extension/run.jl

# proof: final resolution, no cache, everything recomputed fresh
julia --project=. src/results/log_concave_extension/run.jl \
  --config src/results/log_concave_extension/high-resolution.toml --final
```

`run.jl` loads the two artifacts and computes every certified quantity, writing
`results/log_concave_extension/<name>/summary.toml` (the only file the writeup
reads). Read [`bounds.jl`](bounds.jl) top-to-bottom to trace how each value is
produced — it is the whole computation, in order.

### Caching (tuning only)

The expensive interval quadratures (L² norm and Rayleigh quotient, ~20 min at
final resolution, plus the ultracontractivity constants) are isolated as the
`*_impl` functions in `bounds.jl` and wrapped by [`caches.jl`](caches.jl). A
tuning run reads/writes an on-disk cache under `cache/log_concave_extension/`, so
editing a *tuned hyperparameter* (in `[lower_bounds]` / `[pointwise]`) reuses the
unaffected quadratures instead of recomputing them. The `--final` run reads no
cache and is the authoritative, proof-bearing computation.

Caching keys on the quadratures' arguments only, not on source code. After
editing a quadrature body or rebuilding an artifact, wipe the cache first:

```julia
include("caches.jl"); reset_caches!()
```

## Files

| file | role |
|------|------|
| `build_potential.jl`     | step 1 |
| `build_eigenfunction.jl` | step 2 |
| `bounds.jl`              | step 3 math (trace quantities here) |
| `caches.jl`              | disk caches for the step-3 quadratures |
| `run.jl`                 | step 3 entry point + `summary.toml` assembly |
| `{low,high}-resolution.toml` | flat per-topic parameters |
