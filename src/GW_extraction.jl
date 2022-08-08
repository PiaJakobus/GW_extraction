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

function extract(i,i_end)
    """
    i: start index of time window 
    """
    ti = vcat(tim.(i:i_end)...)  
    for el in ti
        a = 1 
    end 
    
end 



#close.(files)

end 
