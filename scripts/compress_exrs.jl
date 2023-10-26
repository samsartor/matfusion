import OpenEXR
import Printf
import Base.Filesystem
using Glob
using Statistics
using ProgressMeter

fmt_percent(x::String) = x
fmt_percent(x::R) where R<:Real = Printf.@sprintf("%0.2f%%", x * 100)
schemes = Dict([string(s) => s for s=instances(OpenEXR.Compression)])

scheme_name = ARGS[1]
iglob = ARGS[2]
odir = ARGS[3]
assert_lossless = get(ARGS, 4, nothing) == "--assert-lossless"

scheme = get(schemes, scheme_name, nothing)
if scheme === nothing
    println("Unknown compression scheme: $(scheme_name)")
    println("Supported schemes are:")
    for name=keys(schemes)
        println("\t- $(name)")
    end
    exit()
end

try
    Filesystem.mkdir(odir)
catch
end

paths = []
for ipath=glob(iglob)
    opath = Filesystem.joinpath(odir, Filesystem.basename(ipath))
    push!(paths, (ipath, opath))
end

function process_exr(ipath::String, opath::String)
    isize = filesize(ipath)
    input = OpenEXR.load_exr(ipath)[1]
    OpenEXR.save_exr(opath, input, scheme)
    if assert_lossless
        output = OpenEXR.load_exr(opath)[1]
        @assert input == output
        assertion = "✓"
    else
        assertion = "?"
    end
    osize = filesize(opath)
    fraction = osize / isize
    println("$(fmt_percent(fraction)) : $(ipath) -> $(opath) $(assertion)")
    return fraction
end

ipaths = glob(iglob)
fractions = zeros(length(ipaths))
@showprogress @Threads.threads for i = 1:length(ipaths)
    ipath = ipaths[i]
    opath = Filesystem.joinpath(odir, Filesystem.basename(ipath))
    fractions[i] = process_exr(ipath, opath)
end
println("AVERAGE : $(fmt_percent(mean(fractions)))")
