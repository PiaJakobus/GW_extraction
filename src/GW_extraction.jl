"
Postprocessing extraction of quadrupol amplitude A₂₀
Reference: https://arxiv.org/abs/1210.6984
"
module GW_extraction

using Glob
using HDF5
using JLD2

G = 6.67430e-8
c = 2.99792458e10

modelname = "../cmf2/output/z35_2d_cmf_symm"
#tstart = 0.77254
#t_pb = 0.43248043
t_ev  = 0.4  
i0    = 550 

function load_files(modelname)
    fnames = glob("$modelname.*")
    files=h5open.(fnames[50:500],"r")
    s=keys.(files)
    isempty(s) && print("Path \"$modelname\" incorrect") 
    return files, s
end

files,s = load_files(modelname)
substr          = length.(s)


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
γ(index,yi)      = 1. ./ .√(1. .- v(index,yi) ./ c^2) 
enth(index,yi)   = 1. .+ e_int(index,yi)./c^2 .+ lpr(index,yi) ./ (den(index,yi).*c^2) 
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

yi = set_index(3)

function extract(i0,i_end)
    """
    i: start index of time window 
    """
    #ti = vcat(tim.(i0:i_end)...)
    nx = 550 
    ny = sum(substr) 
    global res = Array{Float64}(undef,nx,ny)
    global zeit = Array{Float64}(undef,ny)
    global counter = 1 
    for el in 1:(i_end+1-i0)
        println(el)
        yi = set_index(el) 
        global yi   = set_index(el) 
        global tmp1 = cos.(yzn(el,yi))
        global tmp2 = sin.(tmp1)
        global r    = xzn(el,yi)
        global dΩ   = 2π .* delta_θ(el,yi)
        global Φ_l  = Φ(el,yi) 
        global Sᵣ   = S_r(el,yi)
        global Sθ   = S_θ(el,yi)
        for k in 1:substr[el]
            integrant = ((r[:,k].^3 .* Φ_l[:,:,k].^6) .* tmp2[:,k]').*
            ((Sᵣ[:,:,k] .* (3 .* (tmp1[:,k]'.^2) .- 1)  .+ (3 ./ r[:,k])) .* Sθ[:,:,k] .* tmp2[:,k]' .* tmp1[:,k]')
            global res[:,counter] = integrant * dΩ[:,k] # no, there is no dot here 
            global zeit[counter] = yi[k]["time"] 
            counter += 1 
        end 
    end 
    return res,zeit
end 

# make sure here that files indizes match load_files call
#integrant, zeit = extract(50,500)
#jldsave("output/time.jld2"; zeit)
#jldsave("output/integral.jld2"; integrant)

close.(files)

end 
