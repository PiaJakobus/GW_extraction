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
using Smoothing 
using Measures
using ColorSchemes
using KissSmoothing
#using Gadfly
gr()

G = 6.67430e-8
c = 2.99792458e10
prefac = 32 * π^1.5 * G / (√(15) * c^4)

#model = "../cmf2/output/z85_2d_cmf_highreso"
model = "../cmf2/output/z35_2d_cmf_highreso"
#model = "../other_2d/z85_sfhx/output/z85_2d_sfhx"
#model = "../other_2d/z85_sfhx/output/z35_2d_sfhx"

modelname = "z35_cmf"
#modelname = "z35_sfhx"
tb = 0.34
#modelname = "z85_cmf"
#tb = 0.496 
#modelname = "z85_sfhx"
#tb = 0.426



function load_files(model)
    """
    Loads all hdf5 files  under /model 
    returns a list of hdf5 objects 
    """
    println("loading files...")
    fnames = glob("$model.*")
    println("loading files done.")
    return fnames
end


# some useful helper functions 
set_index(i,files,s)       = map(el -> read(files[el]),s)
reshape_1d(q,yi)           = hcat(map(el -> el[q],yi)...) 
reshape_2d(q,yi)           = reduce((x,y) -> cat(x,y,dims=3),map(el -> el[q][:,:,1], yi)) 

xzn(yi)       = reshape_1d("xzn",yi)
xzr(yi)       = reshape_1d("xzr",yi)
xzl(yi)       = reshape_1d("xzl",yi)
yzn(yi)       = reshape_1d("yzn",yi)
yzr(yi)       = reshape_1d("yzr",yi)
yzl(yi)       = reshape_1d("yzl",yi)
vex(yi)       = reshape_2d("vex",yi)
vey(yi)       = reshape_2d("vey",yi)
den(yi)       = reshape_2d("den",yi)
lpr(yi)       = reshape_2d("pre",yi)
ene(yi)       = reshape_2d("ene",yi)
gpo(yi)       = reshape_2d("gpo",yi)
Φ(yi)         = reshape_2d("phi",yi)
v(yi)         = .√(vex(yi).^2 .+ vey(yi).^2)
γ(yi)         = 1. ./ .√(1. .- (v(yi).^2) ./ c^2) 
enth(yi)      = 1. .+ e_int(yi)./c^2 .+ lpr(yi) ./ (den(yi).*c^2) 
S_r(yi)       = den(yi) .* enth(yi) .* γ(yi).^2 .* vex(yi)
S_θ(yi)       = den(yi) .* enth(yi) .* γ(yi).^2 .* vey(yi)
delta_θ(yi,)  = abs.(cos.(yzl(yi)) - cos.(yzr(yi)))
dV(yi)        = (1.0/3.0).*abs.(xzl(yi).^3 .- xzr(yi).^3)
dr(yi)        = abs.(xzl(yi) .- xzr(yi))

function e_int(yi)
    e = ene(yi) .* (G / c^2)
    wl = γ(yi)
    rho = den(yi)
    pre = lpr(yi) 
    eps = e .+ c^2 .* (1.0 .- wl) ./ wl .+ pre .* (1.0 .- wl.^2) ./ (rho .* wl.^2)              
end



function extract(i0,i_end,fnames)
    """
    i: start index of time window 
    Return: Gravitational wave quadrupole integrant 
    where A₂₀ = ∫a₂₀dθ (axisymmetry)(Mueller et al 2013)
    No integreation over radius to preserve radial dependence 
    (Using Rieman sums)
    """
    nx = 550 
    ny = 128 
    res = Vector{Float64}[]
    zeit = Float64[]
    check_A = Float64[]
    integrant = Array{Float64}(undef,nx,ny)
    for el in 1:(i_end+1-i0)
        println(el)
        files   = h5open(fnames[el],"r")
        s      = keys(files)
        substr = length(s)
        yi   = set_index(el,files,s) 
        tmp1 = cos.(yzn(yi))
        tmp3 = sin.(yzn(yi))
        global r    = xzn(yi)
        dΩ   = delta_θ(yi)
        Φ_l  = Φ(yi) 
        Sᵣ   = S_r(yi)
        Sθ   = S_θ(yi)
        dᵣ   = dr(yi)
        for k in 1:substr
            integrant = ((r[:,k].^3 .* Φ_l[:,:,k].^8) .* tmp3[:,k]').*
            (Sᵣ[:,:,k] .* (3 .* (tmp1[:,k]'.^2) .- 1)  .+ (3 ./ r[:,k]) .* Sθ[:,:,k] .* tmp3[:,k]' .* tmp1[:,k]')
            d_int = integrant * dΩ[:,k]
            push!(res,d_int)  
            push!(check_A, d_int' * dᵣ[:,k])  
            push!(zeit,yi[k]["time"]) 
        end
        close(files)
    end 
    res = hcat(res...)
    return prefac .* res, prefac .* check_A , zeit, r[:,1]
end 


function run(i0,iend)
    """
    Return: Integration by Riemann sum'ing ∫a₂₀dr 
    Interpolates radial quantities before integrating 
    """
    fnames = load_files(model)
    integ, integ_r, zeit, radius = extract(i0,iend,fnames[i0:iend])
    println("...Ectract  finished...")
    jldsave("output/time_$modelname.jld2"; zeit)
    jldsave("output/radius_$modelname.jld2"; radius)
    jldsave("output/integral_$modelname.jld2"; integ)
    jldsave("output/integral_r_$modelname.jld2"; integ_r)
    println("...Writing to output files finished...")
    # interpolate with BSline 4th order
    itp_int = mapslices(r -> spline(BSplineKit.interpolate(zeit, r, BSplineOrder(2))),integ,dims=2)
    itp_int_r = spline(BSplineKit.interpolate(zeit, integ_r, BSplineOrder(2)))
    # take deriviative w.r. to time 
    df_integ =  map(r->diff(r,Derivative(1)),itp_int)
    df_integ_r =  diff(itp_int_r,Derivative(1))
    # alternatively..
    # deriv = (hi[2:end] .- hi[1:end-1]) ./ (zeit[2:end] .- zeit[1:end-1])
    dt_integ = transpose(hcat(map(r -> r.(zeit), df_integ)...))
    dt_integ_r = df_integ_r.(zeit)
    jldsave("output/dt_integral_$modelname.jld2";dt_integ)
    jldsave("output/dt_integral_r_$modelname.jld2";dt_integ_r)
    # or for numpy multi-D arrays using NPZ 
    return integ,integ_r,dt_integ,dt_integ_r
end 

function plot_radius_freq(modelname,tmin,tmax,rmin,rmax;mirror=true,kiss=0.8,smooth=1,av=3,cmap=:RdGy_11,rev=true,cmax=17)
    f = jldopen("output/time_$modelname.jld2")
    zeit = f["zeit"]
    close(f) 
    f = jldopen("output/integral_$modelname.jld2")
    integ = f["integ"]
    close(f) 
    f = jldopen("output/integral_r_$modelname.jld2")
    integ_r = f["integ_r"]
    close(f) 
    f = jldopen("output/radius_$modelname.jld2")
    radius = f["radius"]
    close(f) 
    f = jldopen("output/dt_integral_$modelname.jld2") 
    dt_integ = f["dt_integ"]
    close(f) 
    f = jldopen("output/dt_integral_r_$modelname.jld2") 
    dt_integ_r = f["dt_integ_r"]
    close(f) 
    nx = length(dt_integ[:,1])
    tb = t_bounce(modelname) 
    #(!iszero(smooth)) && (dt_integ = hcat(map(it -> KissSmoothing.denoise(dt_integ[:,it])[1],1:length(zeit))...))
    (!iszero(smooth)) && (dt_integ = hcat(map(it -> Smoothing.binomial(dt_integ[:,it],smooth),1:length(zeit))...))
    aver = Int(floor(av/2))
    (!iszero(aver)) && (dt_integ = hcat([[sum(dt_integ[i-aver:i+aver,it])/av for i in (aver+1):nx-aver] for it in 1:length(zeit)]...))
    N_time = length(zeit[tmin:tmax])
    ts = 1e4
    fs = 1/ts
    t0 = zeit[tmin] 
    tend = t0 + (N_time-1) * 1/ts
    t = t0:fs:tend
    Δt = (tend -t0)*1000 
    tmp(ir) = mirror ? vcat(dt_integ[ir,tmin:tmax],reverse(dt_integ[ir,tmin:tmax])) : dt_integ[ir,tmin:tmax]
    freq = mirror ? [el/(2*(tend-t0)) for el in 1:(2*N_time)] : LinRange(0,ts,N_time)
    F(ir) = (!iszero(kiss)) ? KissSmoothing.denoise(abs.(fft(tmp(ir))),factor=kiss)[1] : abs.(fft(tmp(ir)))
    Fr    = fft(dt_integ_r) 
    fi = convert(Int32,floor(length(freq)/2))
    s = hcat(F.(rmin:rmax)...) 
    s2 = hcat(F.(1:length(dt_integ[:,1]))...)
    s1 = s .* freq /2
    p1 = heatmap(log10.(radius[rmin:rmax] ./ 1e5),freq[1:fi],(s1[1:fi,:]),
                 xlims = (log10(radius[rmin]/1e5),log10(radius[rmax]/1e5)),
                clim=(0,cmax),
                title="$(modelname[1:3]) $(modelname[5:end]): Δt=$(Int(floor(Δt,sigdigits= 2))) ms",
                           xlabel="Radius (km)",
                colorbar=:none,
                c=cgrad(cmap, rev=rev),
                xticks=([log10(2.5),log10(5),1,2],["2.5","5","10","100"]),
                ylabel="Frequency (Hz)")
    p1 = plot!(ylims=(0,1500))
    #p1= plot!([],[],color=:white,label=modelname,legend=:top)
    #p2 = annotate!([(90,1400, text("mytext", :red, :right))])
    
    npzwrite("output/r_$modelname.npy", log10.(radius[rmin:rmax] ./ 1e5))
    npzwrite("output/f_$modelname.npy", freq[1:fi])
    npzwrite("output/s_$modelname.npy", s1[1:fi,:])
    m1 = 22#Int32(floor((rmin + 0.8*rmax) /2))
    m2 = 30 
    m3 = 40
    mitte = m1
    p2 = plot(zeit[1:end-10] .- tb,1e5 .* dt_integ[m3,1:end-10],
              label="$(floor(radius[m3]/1e5)) km",
              lw=0.5,
              color=:black)
    p2 = plot!(zeit[1:end-10] .- tb,1e5 .* dt_integ[m2,1:end-10],
              label="$(floor(radius[m2]/1e5)) km",
              lw=0.4,
              color=:green)
    p2 = plot!(zeit[1:end-10] .- tb,1e5 .* dt_integ[m1,1:end-10],
              label="$(floor(radius[m1]/1e5)) km",
              lw=0.7,
              color=:orange,
              ylabel="\$\\tilde{A}_{20}\$ (10\$^5\$cm)",
              xlabel="t-t\$_\\mathrm{b}\$ [s]",
              legend=:outertopright)
 
    p2 = plot!(fg_legend = :false,bg_legend=nothing)

    s1 = 1e5*minimum([minimum(dt_integ[mitte,1:end-10]),minimum(dt_integ[mitte,1:end-10])])
    s2 = 1e5*maximum([maximum(dt_integ[mitte,1:end-10]),maximum(dt_integ[mitte,1:end-10])])
    plot!([t0,tend] .- tb, [s2,s2],fillrange=[s1,s1],fillalpha=0.07,c=:red3,label="")
    plot!([t0,tend] .- tb, [s1,s1],fillrange=[s2,s2],fillalpha=0.07,c=:red3,label="")
    l=@layout [a{.3h};b{.7h}]
    display(plot(p2,p1,layout=l,dpi=200))
    #savefig("plots/radius_freq_on_grey_$modelname.pdf")
    println("tmin=$(t0-tb)")
    println("tmax=$(tend-tb)")
    return
end 

function plot_heatmap(modelname,ir,smooth=1;cmap=:brg,cmax=5)
    f = jldopen("output/time_$modelname.jld2")
    zeit = f["zeit"]
    close(f) 
    f = jldopen("output/radius_$modelname.jld2")
    radius = f["radius"]
    close(f) 
    f = jldopen("output/dt_integral_$modelname.jld2") 
    dt_integ = f["dt_integ"]
    close(f) 
    f = jldopen("output/dt_integral_r_$modelname.jld2") 
    dt_integ_r = f["dt_integ_r"]
    close(f) 
 
    !iszero(smooth) && (dt_integ = hcat(map(it -> Smoothing.binomial(dt_integ[:,it],smooth),1:length(zeit))...))
    # time band:
    t = zeit
    fs =1/ diff(zeit)[1]
    f = dt_integ[ir,:]
    f2 = dt_integ_r
    c = wavelet(Morlet(4π), averagingType=NoAve(), β=0.6)
    res = ContinuousWavelets.cwt(f, c)
    res2 = ContinuousWavelets.cwt(f2, c)
    freq = LinRange(0,fs/2,size(res[1,:])[1])
    
    rad1 = convert(Int32,floor(radius[ir]/1e5)) 
    p1 = plot(t .- 0.34,1e5 .* f,title="a20(t) (cm)",label="r\$_0\$=$rad1 km")
    rad = convert(Int32,floor(radius[ir+4]/1e5)) 
    p1 = plot!(t .- 0.34,1e5 .* dt_integ[ir+4,:],label="$rad km")
    rad = convert(Int32,floor(radius[ir+8]/1e5)) 
    p1 = plot!(t .- 0.34,1e5 .* dt_integ[ir+8,:],
            ylabel = "\$10^5\$ \$\\cdot\$ a20 (cm)", xlabel="time [s]",label="$rad km",legend=:outertopright)
    p1 = plot!(fg_legend = :false,bg_legend=nothing)
    p2 = heatmap(t .- 0.34,freq,((abs.(res)).^0.1)', xlabel= "time (s)",ylabel="frequency [Hz]",c=cmap,colorbar=true,ylims=(0,1500),interpolation=true,clim=(0,cmax))

    p3 = plot(t,f2,legend=false,title=L"\int a20(t) (cm)")
    p4 = heatmap(t,freq,abs.(res2)', xlabel= "time [s]",ylabel="frequency [Hz]",interpolate=true,colorbar=false,ylims=(0,1000))
    l=@layout [a{.3h};b{.7h}]
    plot(p1,p2,layout=l)
    display(plot(p1,p2,layout=l))
end

function single_plot(modelname,tmin,tmax,rmin,rmax;mirror=true,smooth=1,av=3,cmap=:RdGy_11,rev=true)
    tb = t_bounce(modelname) 
    f1 = jldopen("output/time_$modelname.jld2")
    zeit = f1["zeit"]
    f2 = jldopen("output/integral_$modelname.jld2")
    integ = f2["integ"]
    f3 = jldopen("output/radius_$modelname.jld2")
    radius = f3["radius"]
    f4= jldopen("output/dt_integral_$modelname.jld2")
    dt_integ = f4["dt_integ"]
    nx = length(dt_integ[:,1])
    #(!iszero(smooth)) && (dt_integ = hcat(map(it -> KissSmoothing.denoise(dt_integ[:,it])[1],1:length(zeit))...))
    (!iszero(smooth)) && (dt_integ = hcat(map(it -> Smoothing.binomial(dt_integ[:,it],smooth),1:length(zeit))...))
    aver = Int(floor(av/2))
    (!iszero(aver)) && (dt_integ = hcat([[sum(dt_integ[i-aver:i+aver,it])/av 
                                          for i in (aver+1):nx-aver] for it in 1:length(zeit)]...))
    N_time = length(zeit[tmin:tmax])
    ts = 1e4
    fs = 1/ts
    t0 = zeit[tmin] 
    tend = t0 + (N_time-1) * 1/ts
    t = t0:fs:tend
    Δt = (tend -t0)*1000 
    tmp(ir) = mirror ? vcat(dt_integ[ir,tmin:tmax],reverse(dt_integ[ir,tmin:tmax])) : dt_integ[ir,tmin:tmax]
    freq = mirror ? [el/(2*(tend-t0)) for el in 1:(2*N_time)] : LinRange(0,ts,N_time)
    F(ir) = KissSmoothing.denoise(abs.(fft(tmp(ir))),factor=0.7)[1] 
    fi = convert(Int32,floor(length(freq)/2))
    s = hcat(F.(rmin:rmax)...) 
    s2 = hcat(F.(1:length(dt_integ[:,1]))...)
    s1 = s .* freq /2
    p = heatmap(log10.(radius[rmin:rmax] ./ 1e5),freq[1:fi],(s1[1:fi,:]),
                clim=(0,10),
                title="Δt=$(Int(floor(Δt,sigdigits= 1))) ms",
                xlabel="r (km)",
                colorbar=:none,
                c=cgrad(cmap, rev=rev),
                xticks=([log10(2.5),log10(5),1,2],["2.5","5","10","100"]),
                ylabel="freq [Hz]")
    p = plot!(ylims=(0,1500))
    close(f1)
    close(f2)
    close(f3)
    close(f4)
    return p 
end

function plot_timesteps()
    modelname1 = "z85_cmf"
    modelname2 = "z35_cmf"
    modelname3 = "z85_sfhx"
    f1 = jldopen("output/time_$modelname1.jld2")
    tmax1 = length(f1["zeit"]) - 10
    f2 = jldopen("output/time_$modelname2.jld2")
    tmax2 = length(f2["zeit"]) - 10
    f3 = jldopen("output/time_$modelname3.jld2")
    tmax3 = length(f3["zeit"]) - 10
    close(jldopen("output/time_$modelname1.jld2"))
    close(jldopen("output/time_$modelname2.jld2"))
    close(jldopen("output/time_$modelname3.jld2"))
    tmax1 = 400 
    p1 = single_plot(modelname1,1,tmax1,5,200,mirror=true,smooth=1,av=3,rev=true)
    p2 = single_plot(modelname2,1,tmax1,5,200,mirror=true,smooth=1,av=3,rev=true)
    p3 = single_plot(modelname3,1,tmax1,5,200,mirror=true,smooth=1,av=3,rev=true)
    l=@layout [grid(3,1)]
    p = plot(p1,p2,p3,
         layout = l,
         top_margin = 0.1mm,
         label=["a","b","c"])
    display(p)
    close(f1)
    close(f2)
    close(f3)
         
end 

 
function make_ani(Δw,modelname;cmap=:RdGy_11)
    tb = t_bounce(modelname) 
    f = jldopen("output/time_$modelname.jld2")
    zeit = f["zeit"]
    imax = length(zeit)-5 # 2861
    ΔN = 400
    tmp = [[i0,i0+ΔN] for i0 in range(1,imax-ΔN,step=Δw)]
    anim = @animate for i ∈ tmp
        plot_radius_freq(modelname,i[1],i[2],5,200;cmap=cmap,cmax=13,mirror=true,smooth=4,av=3,rev=false)
        end
    gif(anim,"plots/heatmap_$modelname.gif",fps=6)
    close(f)
end 


function t_bounce(modelname)
    tb = 0 
    if modelname == "z35_cmf"
        tb = 0.34
    elseif modelname == "z85_cmf"
        tb = 0.496 
    elseif modelname == "z85_sfhx"
        tb = 0.426
    end 
    return tb
end


end 
