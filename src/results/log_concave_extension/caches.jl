# Disk caches for the expensive phase-3 quadratures (tuning only).
#
# Each cache wraps a pure `*_impl` from bounds.jl and rebinds the bare name that
# the orchestration calls. Keys are the function arguments (artifact paths +
# cheap scalars), hashed by Caching.jl.
#
# Toggle: a tuning run calls load_caches!() before and save_caches!() after, so
# unchanged quadratures are reused from disk. The --final run calls neither, so
# nothing is read from disk and every quantity is recomputed fresh. (In-memory
# memoization still applies within a run, so an identical (potential, fit, cells)
# quadrature — e.g. L² and Rayleigh at equal cells — runs at most once.)
#
# CAUTION: Caching.jl keys on arguments only, NOT on source. After editing a
# quadrature body or rebuilding an artifact, call reset_caches!() before tuning,
# or the stale entries will be served. The --final run is unaffected (reads no
# disk), so published numbers are always freshly computed.

using Caching: Cache, empty!
import Serialization: serialize, deserialize

const CACHE_DIR = joinpath(PROJECT_ROOT, "cache", "log_concave_extension")

fem_eigenvalues        = Cache(fem_eigenvalues_impl;        filename=joinpath(CACHE_DIR, "fem.bin"))
fourier_core_integrals = Cache(fourier_core_integrals_impl; filename=joinpath(CACHE_DIR, "fourier.bin"))
candidate_samples      = Cache(candidate_samples_impl;      filename=joinpath(CACHE_DIR, "candidate.bin"))
first_const            = Cache(first_const_impl;            filename=joinpath(CACHE_DIR, "first.bin"))
second_const           = Cache(second_const_impl;           filename=joinpath(CACHE_DIR, "second.bin"))
total_const            = Cache(total_const_impl;            filename=joinpath(CACHE_DIR, "total.bin"))

const CACHES = (
    fem_eigenvalues, fourier_core_integrals, candidate_samples,
    first_const, second_const, total_const,
)

# Persist the whole Cache (memory dict included) so a later process gets real
# hits. Caching's syncache!/persist! only round-trip within one session — they
# do not reload a persisted index into a fresh object — so we serialize the
# object directly and merge its memory dict back in on load.
load_caches!() = for c in CACHES
    isfile(c.filename) || continue
    merge!(c.cache, deserialize(c.filename, Cache; func=c.func).cache)
end
save_caches!() = for c in CACHES
    mkpath(dirname(c.filename))
    serialize(c.filename, c)
end
reset_caches!() = foreach(c -> empty!(c; empty_disk=true), CACHES)
