import mdtraj as md
import enspara
from enspara import ra
import os 
import numpy as np
from numpy import where

folder = '/project/bowmanlab/ameller/fast-sims/ihm/FASTDistance_lad_Nterm_wt_5n69_min'
out_folder = 'fast_movie_LA_10.xtc'

# Setting up number of rounds, sims each round, file names, etc
dirs =  next(os.walk(folder))[1]
sim_dirs = [e for e in dirs if 'gen' in e]
total_rounds = len(sim_dirs)
print(sim_dirs)
gen0_path = os.path.join(folder,sim_dirs[0])
sims_each_round = len(next(os.walk(gen0_path))[1])
total_sims = total_rounds*sims_each_round
dist_file = os.path.join(folder, 'msm/rankings' , 'distance_per_state'+ str(total_rounds-1)+'.npy') #must be customized
start_state = np.flip(np.argsort(np.load(dist_file)))[0] #must be customzed
print(total_rounds,sims_each_round,start_state)



assigs = ra.load(os.path.join(folder, 'msm/data/assignments.h5'))
pdb = md.load(os.path.join(folder, 'msm/prot_masses.pdb'))

def find_new_start_state(prev_start_state, round_num):
    # define window in assignments file for each round to find where state occurs
    
    window_start = total_sims - sims_each_round * (round_num+1)
    window_end = total_sims - sims_each_round *(round_num)
    traj, idx = np.where(assigs[window_start : window_end] == prev_start_state)
    min_idx = np.argsort(idx)[0]   
    traj, idx = [traj[min_idx], idx[min_idx]]
    new_start_state = assigs[window_start+traj][0]
    print('traj: '+ str(window_start+traj) + ', idx: ' +str(idx) +', state: ' + str(new_start_state))
    return(traj, idx, new_start_state)

def slice_traj(start_state, r):
    # add to traj file
    
    traj, idx, start_state = find_new_start_state(start_state, r)
    xtc_file = os.path.join(folder, 
                             'msm/trajectories', 
                             'trj_gen'+str(total_rounds-r-1).zfill(3)+'_kid'+str(traj).zfill(3)+'.xtc')
    print(total_rounds-r-1)
    tmp_traj = md.load(xtc_file, top=pdb) 
    if idx!=0:
        tmp_traj_slce = tmp_traj[0:idx]
    else:
        tmp_traj_slce = tmp_traj[0]

    return(start_state, tmp_traj_slce)


#output_xtc
traj_list = {}
key_list = []
for r in range(total_rounds):
    #if r!= 4:
    start_state, tmp_traj_slce = slice_traj(start_state, r)
    traj_list[r] = tmp_traj_slce
    key_list.append(r)

traj_list_fin = []
[traj_list_fin.append(traj_list[e]) for e in np.flip(key_list)]
        
full_traj = md.join(traj_list_fin)

#only if want to alter what is aligned when superposed
#atoms=pdb.top.select('name CA and resi 20 to 700')
full_traj.superpose(pdb)#, atom_indices=atoms, ref_atom_indices=atoms)

#so output xtc isn't too large to load
full_traj_sliced = full_traj[0:len(full_traj):10]

full_traj_sliced.save_xtc(out_folder)
