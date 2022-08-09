"
Postprocessing extraction of quadrupol amplitude A₂₀
Reference: https://arxiv.org/abs/1210.6984
"
module GW_extraction

using Glob
using HDF5

G = 6.67430e-8
c = 2.99792458e10

modelname = "../cmf2/output/z35_2d_cmf_symm"
#tstart = 0.77254
#t_pb = 0.43248043
t_ev  = 0.4  
i0    = 550 

function load_files(modelname)
    fnames = glob("$modelname.*")
    files=h5open.(fnames[1:50],"r")
    s=keys.(files)
    isempty(s) && print("Path \"$modelname\" incorrect") 
    return files, s
end

files,s = load_files(modelname)

sub_str(s)       = length.(s)
substruct = sub_str(s)
set_index(index) = map(el->read(files[index][el]),s[index])
tim(index)       = map(el->set_index(index)[el]["time"],1:sub_str(s)[index])
xzn(index)       = map(el->set_index(index)[el]["xzn"],1:sub_str(s)[index])
yzn(index)       = map(el->set_index(index)[el]["yzn"],1:sub_str(s)[index])
vex(index)       = map(el->set_index(index)[el]["vex"][:,:,1],1:sub_str(s)[index])
vey(index)       = map(el->set_index(index)[el]["vey"][:,:,1],1:sub_str(s)[index])
v(index)         = .√(vex(index).^2 + vey(index).^2)
den(index)       = map(el->set_index(index)[el]["den"][:,:,1],1:sub_str(s)[index])
lpr(index)       = map(el->set_index(index)[el]["pre"][:,:,1],1:sub_str(s)[index])
ene(index)       = map(el->set_index(index)[el]["ene"][:,:,1],1:sub_str(s)[index])
gpo(index)       = map(el->set_index(index)[el]["gpo"][:,:,1],1:sub_str(s)[index])
Φ(index)         = map(el->set_index(index)[el]["phi"][:,:,1],1:sub_str(s)[index])

h(index)         = 1. .+ eps./c^2 .+ lpr(index) ./ (rho(index).*c^2)
S(index)         = rho(index) .* enth(index) .* w(index)^2 * √(vex(index).^2 + vey(index).^2)

function e_int(i)
    e = ene(i) .* (G / c^2)
    l = 1:substruct[i]
    all(any.(x -> x <= 0., gpo(i))) ? wl = [ones(550,128) for el in 1:substruct[i]] : wl = 1. ./ .√(1. .- v ./ c^2)
    rho = den(i)
    pre = lpr(i) 
    eps = [(e[k] .+ c^2 .* (1.0 .- wl[k])) ./ wl[k] .+ pre[k] .* (1.0 .- wl[k].^2) ./ (rho[k] .* wl[k].^2) for k in l]             
end

Sᵣ  = 1
S_θ = 1
Φᵣ  = 1
Φ_θ = 1

function extract(i0,i_end)
    """
    i: start index of time window 
    """
    ti = vcat(tim.(i0:i_end)...)  
    for (i,el) in enumerate(i0:i_end)
        for k in 1:sub_str(el)
            tmp1 = cos.(transpose(yzn(el)[k]))
            tmp2 = sin.(tmp1)
            r = xzn(el)[k]
            global integrant = (Φ(el)[k].^6 .* r.^3 .* tmp2).*
              (((Sᵣ .* (3 .* (tmp1.^2) .- 1)  .+ (3 ./ r)) .* S_θ .* tmp2 .* tmp1) .- 
              ((Φᵣ .* (3 .* (tmp1.^2) .- 1)) .+ ((3 ./ r) .* Φ_θ .* tmp2 .* tmp1)))

        end 
    end 
    return integrant
end 


#close.(files)

end 
