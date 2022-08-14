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
using LaTeXStrings
using Wavelets
using ContinuousWavelets

G = 6.67430e-8
c = 2.99792458e10
prefac = 32 * π^1.5 * G / (√(15) * c^4)

modelname = "../cmf2/output/z35_2d_cmf_symm"

function load_files(modelname,i0,iend)
    fnames = glob("$modelname.*")
    files=h5open.(fnames[i0:iend],"r")
    #isempty(s) && print("Path \"$modelname\" incorrect") 
    return files
end


# some useful helper functions 
set_index(i,files,s)    = map(el->read(files[i][el]),s[i])
reshape_1d(q,i,yi,substr) = reduce((x,y) -> cat(x,y,dims=2), map(el->yi[el][q],1:substr[i]))
reshape_2d(q,i,yi,substr) = reduce((x,y) -> cat(x,y,dims=3), map(el->yi[el][q][:,:,1],1:substr[i]))
tim(index,yi,substr)       = map(el->yi[el]["time"],1:substr[index])
xzn(index,yi,substr)       = reshape_1d("xzn",index,yi,substr)
xzr(index,yi,substr)       = reshape_1d("xzr",index,yi,substr)
xzl(index,yi,substr)       = reshape_1d("xzl",index,yi,substr)
yzn(index,yi,substr)       = reshape_1d("yzn",index,yi,substr)
yzr(index,yi,substr)       = reshape_1d("yzr",index,yi,substr)
yzl(index,yi,substr)       = reshape_1d("yzl",index,yi,substr)
vex(index,yi,substr)       = reshape_2d("vex",index,yi,substr)
vey(index,yi,substr)       = reshape_2d("vey",index,yi,substr)
den(index,yi,substr)       = reshape_2d("den",index,yi,substr)
lpr(index,yi,substr)       = reshape_2d("pre",index,yi,substr)
ene(index,yi,substr)       = reshape_2d("ene",index,yi,substr)
gpo(index,yi,substr)       = reshape_2d("gpo",index,yi,substr)
Φ(index,yi,substr)         = reshape_2d("phi",index,yi,substr)
v(index,yi,substr)      = .√(vex(index,yi,substr).^2 .+ vey(index,yi,substr).^2)
γ(index,yi,substr)      = 1. ./ .√(1. .- (v(index,yi,substr).^2) ./ c^2) 
enth(index,yi,substr)   = 1. .+ e_int(index,yi,substr)./c^2 .+ lpr(index,yi,substr) ./ (den(index,yi,substr).*c^2) 
S_r(index,yi,substr)    = den(index,yi,substr) .* enth(index,yi,substr) .* γ(index,yi,substr).^2 .* vex(index,yi,substr)
S_θ(index,yi,substr)    = den(index,yi,substr) .* enth(index,yi,substr) .* γ(index,yi,substr).^2 .* vey(index,yi,substr)
delta_θ(index,yi,substr) = abs.(cos.(yzl(index,yi,substr)) - cos.(yzr(index,yi,substr)))
dV(index,yi,substr)      = (1.0/3.0).*abs.(xzl(index,yi,substr).^3 .- xzr(index,yi,substr).^3)
dr(index,yi,substr)      = abs.(xzl(index,yi,substr) .- xzr(index,yi,substr))

function e_int(i,yi,substr)
    e = ene(i,yi,substr) .* (G / c^2)
    l = 1:substr[i]
    wl = γ(i,yi,substr)
    rho = den(i,yi,substr)
    pre = lpr(i,yi,substr) 
    eps = e .+ c^2 .* (1.0 .- wl) ./ wl .+ pre .* (1.0 .- wl.^2) ./ (rho .* wl.^2)              
end



function extract(i0,i_end,files)
    """
    i: start index of time window 
    """
    nx = 550 
    global s = keys.(files)
    global substr = length.(s)
    ny = sum(substr) 
    res = Array{Float64}(undef,nx,ny)
    zeit = Array{Float64}(undef,ny)
    check_A = Array{Float64}(undef,ny)
    integrant = Array{Float64}(undef,nx,128)
    counter = 1 
    tmp_r = zeros(nx)
    tmp2 = 0.0 
    for el in 1:(i_end+1-i0)
        println(el)
        yi   = set_index(el,files,s) 
        tmp1 = cos.(yzn(el,yi,substr))
        tmp3 = sin.(yzn(el,yi,substr))
        r    = xzn(el,yi,substr)
        dΩ   = delta_θ(el,yi,substr)
        Φ_l  = Φ(el,yi,substr) 
        Sᵣ   = S_r(el,yi,substr)
        Sθ   = S_θ(el,yi,substr)
        global dᵣ   = dr(el,yi,substr)
        for k in 1:substr[el]
            #integrant = ((r[:,k].^3 .* Φ_l[:,:,k].^8) .* tmp3[:,k]').*
            #(Sᵣ[:,:,k] .* (3 .* (tmp1[:,k]'.^2) .- 1)  .+ (3 ./ r[:,k]) .* Sθ[:,:,k] .* tmp3[:,k]' .* tmp1[:,k]')
            #res[:,counter] = integrant * dΩ[:,k] # no, there is no dot here 
            #check_A[counter] = res[:,counter]' * dᵣ[:,k]  
            zeit[counter] = yi[k]["time"] 
            for rr in 1:550
                for tt in 1:128
                    integrant[rr,tt] = ((r[rr,k]^3 * Φ_l[rr,tt,k]^8) * tmp3[tt,k])*
                    (Sᵣ[rr,tt,k] * (3 * (tmp1[tt,k]^2) - 1)  + (3 / r[rr,k]) * Sθ[rr,tt,k] * tmp3[tt,k] * tmp1[tt,k])
                    tmp_r[rr] += integrant[rr,tt] * dΩ[tt,k]
                end
                tmp2 += tmp_r[rr] * dᵣ[rr,k]
            end 
            res[:,counter] = tmp_r
            check_A[counter] = tmp2 
            tmp_r = 0 .* tmp_r
            tmp2 = 0.0
            counter += 1 
        end 
    end 
    close.(files)
    return prefac .* res, prefac .* check_A , zeit,dᵣ
end 


function run(i0,iend)
    files = load_files(modelname,i0,iend)
    integ, integ_r, zeit, dr = extract(i0,iend,files)
    jldsave("output/time.jld2"; zeit)
    jldsave("output/dr.jld2"; dr)
    jldsave("output/integral.jld2"; integ)
    jldsave("output/integral_r.jld2"; integ_r)
    close.(files)
    # interpolate with BSline 4th order
    itp_int = mapslices(r -> interpolate(zeit, r, BSplineOrder(2)),integ,dims=2)
    itp_int_r = interpolate(zeit, integ_r, BSplineOrder(2))
    # take deriviative w.r. to time 
    df_integ =  map(r->diff(r,Derivative(1)),itp_int)
    df_integ_r =  diff(itp_int_r,Derivative(1))
    # alternatively..
    # deriv = (hi[2:end] .- hi[1:end-1]) ./ (zeit[2:end] .- zeit[1:end-1])
    dt_integ = transpose(hcat(map(r -> r.(zeit), df_integ)...))
    dt_integ_r = df_integ_r.(zeit)
    jldsave("output/dt_integral.jld2";dt_integ)
    jldsave("output/dt_integral_r.jld2";dt_integ_r)
    # or for numpy multi-D arrays using NPZ 
    #npzwrite("output/numpy_res.npz",[dt_integ,zeit])
    return integ,integ_r,dt_integ,dt_integ_r
end 

function plot_radius_freq(tmin,tmax,rmin,rmax)
    zeit = jldopen("output/time.jld2")["zeit"]
    integ = jldopen("output/integral.jld2")["integ"]
    integ_r = jldopen("output/integral_r.jld2")["integ_r"]
    dr = jldopen("output/dr.jld2")["dr"]
    radius = jldopen("output/radius_z35.jld2")["radius"]
    dt_integ = jldopen("output/dt_integral.jld2")["dt_integ"] 
    dt_integ_r = jldopen("output/dt_integral_r.jld2")["dt_integ_r"] 
    N_time = length(zeit[tmin:tmax])
    ts = 1e4
    fs = 1/ts
    t0 = zeit[tmin]
    tend = t0 + (N_time-1) * 1/ts
    t = t0:fs:tend
    Δt = (tend -t0)*1000
    F(ir) = abs.(fft(dt_integ[ir,tmin:tmax]) |> fftshift)
    Fr    = fft(dt_integ_r) |> fftshift
    #freqs = fftfreq(length(zeit[tmin:tmax]), ts) |> fftshift
    freq = LinRange(0,ts/2,N_time)
    fi = convert(Int32,floor(length(freq)/2))
    s = hcat(F.(rmin:rmax)...) 
    s2 = hcat(F.(1:550)...)
    p1 = heatmap(radius[rmin:rmax] ./ 1e5,freq[1:fi],s[1:fi,:],title=
                 "t=$(floor(t0,sigdigits=2)) s ($(floor(Δt,sigdigits=
                 2)) ms)",xlabel="radius/km",ylabel="freq [Hz]")
    p2 = plot(zeit,dt_integ[rmin,:],label="r1= $(floor(radius[rmin]/1e5)) km",lw=0.4)
    p2 = plot!(zeit,dt_integ[rmax,:],label="r2= $(floor(radius[rmax]/1e5)) km",lw=0.4,
               ylabel="a20 (cm)",xlabel="time [s]",legend=:left)
    s1 = minimum([minimum(dt_integ[rmin,:]),minimum(dt_integ[rmax,:])])
    s2 = maximum([maximum(dt_integ[rmax,:]),maximum(dt_integ[rmin,:])])
    println(s1,"  ",s2)
    p2 = plot!([t0,tend], [s2,s2],fillrange=[s1,s1],fillalpha=0.2,c=1,label="")
    p2 = plot!([t0,tend], [s1,s1],fillrange=[s2,s2],fillalpha=0.2,c=1,label="")
    l=@layout [a{.3h};b{.7h}]
    display(plot(p2,p1,layout=l))
    close(jldopen("output/time.jld2"))
    close(jldopen("output/integral.jld2"))
    close(jldopen("output/integral_r.jld2"))
    close(jldopen("output/dr.jld2"))
    close(jldopen("output/dt_integral.jld2"))
    close(jldopen("output/dt_integral_r.jld2"))
    close(jldopen("output/radius_z35.jld2"))
    return 
end 

function plot_heatmap(ir)
    zeit       = jldopen("output/time.jld2")["zeit"]
    dt_integ   = jldopen("output/dt_integral.jld2")["dt_integ"] 
    dt_integ_r = jldopen("output/dt_integral_r.jld2")["dt_integ_r"] 
    radius     = jldopen("output/radius_z35.jld2")["radius"]
 
    # time band:
    t = zeit[400:end]
    fs =1/ diff(zeit)[400]
    f = dt_integ[ir,400:end]
    f2 = dt_integ_r[400:end]
    c = wavelet(Morlet(1.5π), averagingType=NoAve(), β=2)
    res = ContinuousWavelets.cwt(f, c)
    res2 = ContinuousWavelets.cwt(f2, c)
    freq = LinRange(0,fs/2,size(res[1,:])[1])
    
    rad = convert(Int32,floor(radius[ir]/1e5)) 
    p1 = plot(t,f,title="a20(t) (cm)",label="($rad km)")
    rad = convert(Int32,floor(radius[ir+4]/1e5)) 
    p1 = plot!(t,dt_integ[ir+4,400:end],label="($rad km)")
    rad = convert(Int32,floor(radius[ir+8]/1e5)) 
    p1 = plot!(t,dt_integ[ir+8,400:end],label="($rad km)",legend=:right)

    p2 = heatmap(t,freq,abs.(res)', xlabel= "time [s]",ylabel="frequency [Hz]",colorbar=false,ylims=(0,2000))

    p3 = plot(t,f2,legend=false,title=L"\int a20(t) (cm)")
    p4 = heatmap(t,freq,abs.(res2)', xlabel= "time [s]",ylabel="frequency [Hz]",colorbar=false,ylims=(0,2000))
    l=@layout [a{.3h};b{.7h}]
    display(plot(p1,p2,layout=l))

    close(jldopen("output/time.jld2"))
    close(jldopen("output/dr.jld2"))
    close(jldopen("output/dt_integral.jld2"))
    close(jldopen("output/dt_integral_r.jld2"))
    close(jldopen("output/radius_z35.jld2"))
end


end 
