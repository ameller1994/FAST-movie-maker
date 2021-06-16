import mdtraj as md
import enspara
from enspara import ra
import os
import numpy as np

def find_new_start_state(assigs, prev_start_state, round_num, sims_each_round):
    # define window in assignments file for each round to find where state occurs
    # print(total_sims, round_num, sims_each_round)
    window_end = sims_each_round * (round_num + 1)
    print(window_end)
    traj_indices, frame_indices = np.where(assigs[0: window_end] == prev_start_state)
    print(traj_indices, frame_indices)
    # Select the earliest instance of the previous starting state
    # This line sorts based on traj index and then frame index
    traj, idx = sorted(zip(traj_indices, frame_indices))[0]

    new_start_state = assigs[traj][0]
    print('traj: '+ str(traj) + ', idx: ' + str(idx) + ', state: ' + str(new_start_state))
    return traj, idx, new_start_state

def slice_traj(folder, assigs, pdb, start_state, round_num, sims_each_round):
    # add to traj file
    traj, idx, next_start_state = find_new_start_state(assigs, start_state, round_num, sims_each_round)
    traj_round = traj // sims_each_round
    traj_kid = traj % sims_each_round
    xtc_file = os.path.join(folder,
                            'msm/trajectories',
                            'trj_gen' + str(traj_round).zfill(3) + '_kid' + str(traj_kid).zfill(3)+'.xtc')

    tmp_traj = md.load(xtc_file, top=pdb)
    if idx != 0:
        tmp_traj_slce = tmp_traj[0:idx]
    else:
        tmp_traj_slce = tmp_traj[0]

    next_round = traj_round - 1

    return next_start_state, next_round, tmp_traj_slce


###################
##### INPUTS ######
###################

# input FAST trajectory
folder = '/project/bowmanlab/j.lotthammer/Simulations/myosin/cleft_opening/wt/fast/FAST_cleft_opening_7jhj'
# where to write output xtc
out_path = 'cleft_opening.xtc'
# boolean flag for whether or not objective is maximized
maximizing_objective = True
# objective name
objective_name = 'distance'

#######################
##### END INPUTS ######
#######################

# Setting up number of rounds, sims each round, file names, etc
dirs = next(os.walk(folder))[1]
sim_dirs = [e for e in dirs if 'gen' in e]
total_rounds = len(sim_dirs)
print(sim_dirs)
gen0_path = os.path.join(folder, sim_dirs[0])
sims_each_round = len(next(os.walk(gen0_path))[1])
total_sims = total_rounds * sims_each_round

# objective function values
objective_file = os.path.join(folder, 'msm/rankings', objective_name + '_per_state' + str(total_rounds-1)+'.npy')
if maximizing_objective:
    start_state = np.argsort(np.load(objective_file))[-1]
else:
    start_state = np.argsort(np.load(objective_file))[0]

print(total_rounds, sims_each_round, start_state)

assigs = ra.load(os.path.join(folder, 'msm/data/assignments.h5'))
pdb = md.load(os.path.join(folder, 'msm/prot_masses.pdb'))

traj_list = []
round_num = total_rounds - 1
while (round_num > 0):
    start_state, round_num, tmp_traj_slce = slice_traj(folder, assigs, pdb, start_state, round_num, sims_each_round)
    print(tmp_traj_slce)
    traj_list.append(tmp_traj_slce)

print(traj_list)
traj_list.reverse()
print(traj_list)
full_traj = md.join(traj_list)

# superpose since molecules can drift
full_traj.superpose(pdb)

#so output xtc isn't too large to load
full_traj_sliced = full_traj[0:len(full_traj):10]

full_traj_sliced.save_xtc(out_path)
