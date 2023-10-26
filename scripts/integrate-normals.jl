using FFTW
using Images
using ProgressMeter
import Base.Filesystem
using Glob
using LinearAlgebra
import Base.Iterators

normals2dels(n) = (
	-n[1,:,:] ./ n[3,:,:],
	 n[2,:,:] ./ n[3,:,:],
)

function normal_map(img::Array{RGB{N0f8}, 2}, strength::Float32; min_slope::Float32=1f0/15f0)::Array{Float32,  3}
	lerp(a, b, t) = a * (1-t) + b * t

	raw_normals = 2 .* channelview(img) .- 1
	normals = lerp.([0, 0, 1], raw_normals, strength)
	clamp!(view(normals, 3, :, :), min_slope, 1f0)
	normals = mapslices(normalize, normals, dims=1)
	normals
end

function poisson_fft(normals::Array{Float32, 3})::Array{Float32, 2}
	_,h,w = size(normals)
	p, q = normals2dels(normals)

	p̂ = fft(p)
	q̂ = fft(q)
	ẑ = similar(q̂)
	for u=1:h,v=1:w
		x = π*v/w
		y = π*u/h
		num = sin(2*x)*p̂[u,v]+sin(2*y)*q̂[u,v]
		den = im*max(eps(),4*(sin(x)^2 + sin(y)^2))
		ẑ[u,v] = num / den
	end
	real.(ifft(ẑ))
end

if abspath(PROGRAM_FILE) == @__FILE__
	local inputs
	inputs = []
	for a in ARGS
		if Filesystem.isdir(a)
			inputs = vcat(inputs, Filesystem.readdir(a, join=true))
		else
			push!(inputs, a)
		end
	end
	println("Integrating $(length(inputs)) normals.")
	p = Progress(length(inputs))
	@Threads.threads for input in inputs
		output = replace(input, ".png" => ".exr", "normals" => "height")
		if isfile(output)
			next!(p)
			println("$(output) exists")
			continue
		end
		normals = normal_map(Images.load(input), 1f0)
		z::Array{Float32, 2} = poisson_fft(normals)
	    scaled_z = colorview(Gray, 2 * (z .- minimum(z)) ./ maximum(size(z)))
		Images.save(output, colorview(Gray, scaled_z))
		next!(p)
	end
end