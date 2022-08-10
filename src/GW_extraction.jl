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


#reduce((x,y) -> cat(x, y, dims=3), a)
#permutedims(b,(3,1,2))

substr          = length.(s)
set_index(i)    = map(el->read(files[i][el]),s[i])
# in order to not load the file multiple times save it in variable
# always define index y_i                = set_index(i) 

reshape_1d(q,i,yi) = reduce((x,y) -> cat(x,y,dims=2), map(el->yi[el][q],1:substr[i]))
reshape_2d(q,i,yi) = reduce((x,y) -> cat(x,y,dims=3), map(el->yi[el][q][:,:,1],1:substr[i]))
                         
#tim(index)       = map(el->set_index(index)[el]["time"],1:substr[index])
#xzn(index)       = map(el->set_index(index)[el]["xzn"],1:substr[index])
#yzn(index)       = map(el->set_index(index)[el]["yzn"],1:substr[index])
#vex(index)       = map(el->set_index(index)[el]["vex"][:,:,1],1:substr[index])
#vey(index)       = map(el->set_index(index)[el]["vey"][:,:,1],1:substr[index])
#den(index)       = map(el->set_index(index)[el]["den"][:,:,1],1:substr[index])
tim(index,yi)       = map(el->yi[el]["time"],1:substr[i])
xzn(index,yi)       = reshape_1d("xzn",index,yi)
yzn(index,yi)       = reshape_2d("yzn",index,yi)
vex(index,yi)       = reshape_2d("vex",index,yi)
vey(index,yi)       = reshape_2d("vey",index,yi)
den(index,yi)       = reshape_2d("den",index,yi)
lpr(index,yi)       = reshape_2d("pre",index,yi)
ene(index,yi)       = reshape_2d("ene",index,yi)
gpo(index,yi)       = reshape_2d("gpo",index,yi)
Φ(index,yi)         = reshape_2d("phi",index,yi)

v(index,yi)         = .√(vex(index,yi).^2 .+ vey(index,yi).^2)
γ(index,yi)         = 1. ./ .√(1. .- v(index,yi) ./ c^2) 
enth(index,yi)      = 1. .+ e_int(index,yi)./c^2 .+ lpr(index,yi) ./ (den(index,yi).*c^2) 
S(index,yi)         = den(index,yi) .* enth(index,yi) .* γ(index,yi).^2 .* v(index,yi)

function e_int(i,yi)
    #yi = set_index(i) 
    e = ene(i,yi) .* (G / c^2)
    l = 1:substr[i]
    all(any.(x -> x <= 0., gpo(i,yi))) ? wl = ones(550,128,substr[i]) : wl = γ(i,yi) 
    rho = den(i,yi)
    pre = lpr(i,yi) 
    eps = e .+ c^2 .* (1.0 .- wl) ./ wl .+ pre .* (1.0 .- wl.^2) ./ (rho .* wl.^2)              
end

Sᵣ  = 1
S_θ = 1
Φᵣ  = 1
Φ_θ = 1

function extract(i0,i_end)
    """
    i: start index of time window 
    """
    #ti = vcat(tim.(i0:i_end)...)  
    for (i,el) in enumerate(i0:i_end)
        global yi = set_index(el) 
        for k in 1:substr[el]
            tmp1 = cos.(transpose(yzn(el,yi)[k]))
            tmp2 = sin.(tmp1)
            r = xzn(el,yi)[k]
            global integrant = (Φ(el,yi)[k].^6 .* r.^3 .* tmp2).*
            ((Sᵣ .* (3 .* (tmp1.^2) .- 1)  .+ (3 ./ r)) .* S_θ .* tmp2 .* tmp1)
            global tmp = e_int(el,yi)[k] .* integrant 
        end 
    end 
    return tmp
end 


#close.(files)

end 
