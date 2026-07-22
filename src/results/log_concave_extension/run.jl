# Phase 3 entry point: compute the certified bounds and write summary.toml.
#
#   julia --project=. src/results/log_concave_extension/run.jl [--config PATH] [--final]
#
# Defaults to low-resolution.toml with disk caching ON (fast tuning). Pass
# --final for the proof-bearing run: no cache is read, every quantity is
# recomputed fresh. Output goes to writeup/results/log_concave_extension/<name>/summary.toml
# where <name> is the config's `name` field. The potential artifacts are always
# the shared high-resolution ones.

include(joinpath(@__DIR__, "bounds.jl"))
include(joinpath(@__DIR__, "caches.jl"))

using TOML

const POTENTIAL_TOML = joinpath(
    PROJECT_ROOT, "writeup", "results", "log_concave_extension", "high-resolution",
    "0-make-potential.toml",
)

function parse_args(args)
    config_path = joinpath(@__DIR__, "low-resolution.toml")
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

    potential = TOML.parsefile(POTENTIAL_TOML)["result"]
    summary = merge(tables, Dict(
        "step" => "summary",
        "inputs" => Dict(
            "config" => config["name"],
            "potential_checkpoint" => relpath(POTENTIAL_CHK, PROJECT_ROOT),
            "fit_checkpoint" => relpath(FIT_CHK, PROJECT_ROOT),
        ),
        "potential_measure" => Dict(
            "unnormalized_core_mass" => potential["unnormalized_core_mass"],
            "unnormalized_wing_mass" => potential["unnormalized_wing_mass"],
            "normalization_constant" => potential["normalization_constant"],
            "relative_wing_mass" => potential["relative_wing_mass"],
        ),
    ))

    out_dir = joinpath(PROJECT_ROOT, "writeup", "results", "log_concave_extension", config["name"])
    mkpath(out_dir)
    out = joinpath(out_dir, "summary.toml")
    open(io -> TOML.print(io, summary; sorted=true), out, "w")
    println("Summary saved to ", out)
    return out
end

main(ARGS)
