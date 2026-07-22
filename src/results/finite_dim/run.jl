# Phase 3 entry point: compute the finite-dimensional certificate values and
# write summary.toml.
#
#   julia --project=. src/results/finite_dim/run.jl [--config PATH] [--final]
#
# Defaults to finite-dim.toml with disk caching ON (fast tuning). Pass --final for
# the proof-bearing run: no cache is read, every quantity is recomputed fresh.
# Output goes to writeup/results/finite_dim/<name>/summary.toml where <name> is the
# config's `name` field.

include(joinpath(@__DIR__, "bounds.jl"))
include(joinpath(@__DIR__, "caches.jl"))

function parse_args(args)
    config_path = joinpath(@__DIR__, "finite-dim.toml")
    final = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--final"
            final = true
        elseif a == "--config"
            i += 1; config_path = args[i]
        elseif startswith(a, "--config=")
            config_path = split(a, "="; limit=2)[2]
        elseif a in ("-h", "--help")
            println("Usage: run.jl [--config PATH] [--final]")
            exit(0)
        else
            error("unknown argument '$a'; use --help")
        end
        i += 1
    end
    return (; config_path, final)
end

function main(args)
    opts = parse_args(args)
    config = TOML.parsefile(opts.config_path)

    opts.final || load_caches!()
    tables = compute_bounds(config)
    opts.final || save_caches!()

    summary = merge(tables, Dict(
        "step" => "summary",
        "inputs" => Dict(
            "config" => config["name"],
            "potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT),
            "fit_checkpoint" => relpath(FIT_CHK, PROJECT_ROOT),
            "log_concave_summary" => relpath(SUMMARY_TOML, PROJECT_ROOT),
            "log_concave_potential" => relpath(POTENTIAL_TOML, PROJECT_ROOT),
        ),
    ))

    out_dir = joinpath(PROJECT_ROOT, "writeup", "results", "finite_dim", config["name"])
    mkpath(out_dir)
    out = joinpath(out_dir, "summary.toml")
    open(io -> TOML.print(io, summary; sorted=true), out, "w")
    println("Summary saved to ", out)
    return out
end

main(ARGS)
