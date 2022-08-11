"
Postprocessing extraction of quadrupol amplitude A₂₀
Reference: https://arxiv.org/abs/1210.6984
"
module GW_extraction

using Glob
using HDF5
using JLD2
using BSplineKit
using FFTW
using Plots 
using NPZ

G = 6.67430e-8
c = 2.99792458e10
prefac = 32 * π^2 * G / (√(15) * c^4)

modelname = "../cmf2/output/z35_2d_cmf_symm"
#tstart = 0.77254
#t_pb = 0.43248043
t_ev  = 0.4  
i0    = 550 

function load_files(modelname,i0,iend)
    fnames = glob("$modelname.*")
    files=h5open.(fnames[i0:iend],"r")
    s=keys.(files)
    isempty(s) && print("Path \"$modelname\" incorrect") 
    return files, s
end


#permutedims(b,(3,1,2))

set_index(i)    = map(el->read(files[i][el]),s[i])
# in order to not load the file multiple times save it in variable:
# y_i = set_index(i) 

reshape_1d(q,i,yi) = reduce((x,y) -> cat(x,y,dims=2), map(el->yi[el][q],1:substr[i]))
reshape_2d(q,i,yi) = reduce((x,y) -> cat(x,y,dims=3), map(el->yi[el][q][:,:,1],1:substr[i]))
                         
tim(index,yi)       = map(el->yi[el]["time"],1:substr[index])
xzn(index,yi)       = reshape_1d("xzn",index,yi)
yzn(index,yi)       = reshape_1d("yzn",index,yi)
yzr(index,yi)       = reshape_1d("yzr",index,yi)
yzl(index,yi)       = reshape_1d("yzl",index,yi)
vex(index,yi)       = reshape_2d("vex",index,yi)
vey(index,yi)       = reshape_2d("vey",index,yi)
den(index,yi)       = reshape_2d("den",index,yi)
lpr(index,yi)       = reshape_2d("pre",index,yi)
ene(index,yi)       = reshape_2d("ene",index,yi)
gpo(index,yi)       = reshape_2d("gpo",index,yi)
Φ(index,yi)         = reshape_2d("phi",index,yi)

v(index,yi)      = .√(vex(index,yi).^2 .+ vey(index,yi).^2)
γ(index,yi)      = 1. ./ .√(1. .- (v(index,yi).^2) ./ c^2) 
enth(index,yi)   = 1. .+ G .* e_int(index,yi)./c^2 .+ lpr(index,yi) ./ (den(index,yi).*c^2) 
S_r(index,yi)    = den(index,yi) .* enth(index,yi) .* γ(index,yi).^2 .* vex(index,yi)
S_θ(index,yi)    = den(index,yi) .* enth(index,yi) .* γ(index,yi).^2 .* vey(index,yi)
delta_θ(index,yi) = abs.(cos.(yzl(index,yi)) - cos.(yzr(index,yi)))

function e_int(i,yi)
    e = ene(i,yi) .* (G / c^2)
    l = 1:substr[i]
    all(any.(x -> x <= 0., gpo(i,yi))) ? wl = ones(550,128,substr[i]) : wl = γ(i,yi) 
    rho = den(i,yi)
    pre = lpr(i,yi) 
    eps = e .+ c^2 .* (1.0 .- wl) ./ wl .+ pre .* (1.0 .- wl.^2) ./ (rho .* wl.^2)              
end

#yi = set_index(3)

function extract(i0,i_end)
    """
    i: start index of time window 
    """
    #ti = vcat(tim.(i0:i_end)...)
    nx = 550 
    ny = sum(substr) 
    res = Array{Float64}(undef,nx,ny)
    zeit = Array{Float64}(undef,ny)
    counter = 1 
    for el in 1:(i_end+1-i0)
        println(el)
        yi = set_index(el) 
        yi   = set_index(el) 
        tmp1 = cos.(yzn(el,yi))
        tmp2 = sin.(tmp1)
        r    = xzn(el,yi)
        dΩ   = 2π .* delta_θ(el,yi)
        Φ_l  = Φ(el,yi) 
        Sᵣ   = S_r(el,yi)
        Sθ   = S_θ(el,yi)
        for k in 1:substr[el]
            integrant = ((r[:,k].^3 .* Φ_l[:,:,k].^6) .* tmp2[:,k]').*
            ((Sᵣ[:,:,k] .* (3 .* (tmp1[:,k]'.^2) .- 1)  .+ (3 ./ r[:,k])) .* Sθ[:,:,k] .* tmp2[:,k]' .* tmp1[:,k]')
            res[:,counter] = integrant * dΩ[:,k] # no, there is no dot here 
            zeit[counter] = yi[k]["time"] 
            counter += 1 
        end 
    end 
    return prefac .* res,zeit
end 

# main 
#
# make sure here that files indizes match load_files call
#i0 = 500 
#iend = 510
#files,s = load_files(modelname,i0,iend)
#substr          = length.(s)
#integ, zeit = extract(i0,iend)
#jldsave("output/time.jld2"; zeit)
#jldsave("output/integral.jld2"; integ)
#close.(files)

#f = jldopen("output/time.jld2")
#d = jldopen("output/integral.jld2")
#xdata = f["zeit"]
#ydata = d["integ"]

# interpolate with BSline 4th order
#itp_arr = mapslices(r -> interpolate(xdata, r, BSplineOrder(4)),ydata,dims=2)
# take deriviative w.r. to time 
#df_arr =  map(r->diff(r,Derivative(1)),itp_arr)
# cut off big dt's and reform into (N_r, N_time)
#time_red = xdata[200:end]
#res = transpose(hcat(map(r -> r.(time_red), df_arr)...))

#jldsave("output/time_red.jld2";time_red)
#jldsave("output/derivative_red.jld2";res)

# or for numpy multi-D arrays using NPZ 
#npzwrite("output/numpy_res.npz",res)
g = jldopen("output/time_red.jld2")["time_red"]
h = jldopen("output/derivative_red.jld2")["res"] 
radius = npzread("output/radius_z35.npz")["arr_0"]
#plot(g,h[6,:],label = "test")
N = length(time)
fs = 1e-4
t0 = time[1]
tmax = t0 + (N-1) * fs
t = t0:Ts:max
# signal
signal = sin.(2π * 60 .* t) # sin (2π f t)

# Fourier Transform of it
F = fft(signal) |> fftshift
freqs = fftfreq(length(t), 1.0/Ts) |> fftshift

# plots
time_domain = plot(t, signal, title = "Signal")
freq_domain = plot(freqs, abs.(F), title = "Spectrum", xlim=(-1000, +1000))
plot(time_domain, freq_domain, layout = 2)
#close(g)
#close(h)

#close(f)
#close(d)

end 
