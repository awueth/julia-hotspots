# Disk caches for the expensive phase-3 quadratures (tuning only).
#
# Each cache wraps a pure `*_impl` from bounds.jl and rebinds the bare name that
# the orchestration calls. Keys are the function arguments (artifact paths + cheap
# scalars), hashed by Caching.jl. `first_const` uses the same signature as the
# log_concave cache, so tuning parameters that do not feed `first_constant` (e.g.
# decay_split) reuse every expensive quadrature. Adaptive buffered-mass
# quadrature is cheap and runs once before the optimization loop.
#
# Toggle: a tuning run calls load_caches!() before and save_caches!() after; the
# --final run calls neither, so nothing is read from disk and every quantity is
# recomputed fresh. (In-memory memoization still applies within a run.)
#
# CAUTION: Caching.jl keys on arguments only, NOT on source. After editing a
# quadrature body or rebuilding an artifact, call reset_caches!() before tuning.
# The --final run is unaffected (reads no disk).

using Caching: Cache, empty!
import Serialization: serialize, deserialize

const CACHE_DIR = joinpath(PROJECT_ROOT, "cache", "finite_dim")

candidate_samples = Cache(candidate_samples_impl; filename=joinpath(CACHE_DIR, "candidate.bin"))
first_const       = Cache(first_const_impl;       filename=joinpath(CACHE_DIR, "first.bin"))

const CACHES = (candidate_samples, first_const)

# Persist the whole Cache (memory dict included) so a later process gets real
# hits. Caching's syncache!/persist! only round-trip within one session, so we
# serialize the object directly and merge its memory dict back in on load.
load_caches!() = for c in CACHES
    isfile(c.filename) || continue
    merge!(c.cache, deserialize(c.filename, Cache; func=c.func).cache)
end
save_caches!() = for c in CACHES
    mkpath(dirname(c.filename))
    serialize(c.filename, c)
end
reset_caches!() = foreach(c -> empty!(c; empty_disk=true), CACHES)
