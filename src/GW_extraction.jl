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
    dt_integ = transpose(hcat(map(r -> r.(zeit), df_integ)...))
    dt_integ_r = df_integ_r.(zeit)
    jldsave("output/dt_integral.jld2";dt_integ)
    jldsave("output/dt_integral_r.jld2";dt_integ_r)
    # or for numpy multi-D arrays using NPZ 
    #npzwrite("output/numpy_res.npz",[dt_integ,zeit])
    return integ,integ_r,dt_integ,dt_integ_r
end 

function fourrier(ir)
    zeit = jldopen("output/time.jld2")["zeit"]
    integ = jldopen("output/integral.jld2")["integ"]
    integ_r = jldopen("output/integral_r.jld2")["integ_r"]
    dr = jldopen("output/dr.jld2")["dr"]
    dt_integ = jldopen("output/dt_integral.jld2")["dt_integ"] 
    dt_integ_r = jldopen("output/dt_integral_r.jld2")["dt_integ_r"] 
    #radius = npzread("output/radius_z35.npz")["arr_0"]
    N_time = length(zeit)
    ts = 1e4
    fs = 1/ts
    t0 = zeit[1]
    tmax = t0 + (N_time-1) * 1/ts
    t = t0:fs:tmax

    F(ir) = fft(dt_integ[ir,:]) |> fftshift
    Fr    = fft(dt_integ_r) |> fftshift
    freqs = fftfreq(length(zeit), fs) |> fftshift
    #global F_r = plot(zeit,dt_integ_r,title= "Signal over r")
    #if plotting
    #    fi = F(6) 
    #    time_domain(ir) = plot(zeit, dt_integ[ir,:], title = "Signal")
    #    freq_domain(ir) = plot(freqs, abs.(F(ir)), title = "Spectrum", xlim=(-1000, +1000))
        #time_dr  = plot(zeit,integ_r,title="int over r")
    #    ir = 6  
    #    display(plot(time_domain(ir), freq_domain(ir),F_r ,layout = 3))
    #end
    close(jldopen("output/time.jld2"))
    close(jldopen("output/integral.jld2"))
    close(jldopen("output/integral_r.jld2"))
    close(jldopen("output/dr.jld2"))
    close(jldopen("output/dt_integral.jld2"))
    close(jldopen("output/dt_integral_r.jld2"))
    return freqs, F(ir),Fr
end 


i0 = 0
iend = 850

end 
