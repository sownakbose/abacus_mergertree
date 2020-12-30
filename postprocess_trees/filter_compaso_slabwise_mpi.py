#!/usr/bin/env python

from compaso_halo_catalog import CompaSOHaloCatalog
from scipy.spatial import cKDTree
from tqdm import *
from numba import njit
import astropy.table
from mpi4py import MPI
from astropy.table import Table
import numpy as np
import warnings
import asdf
import glob
import time
import os

warnings.filterwarnings("ignore")

factor    = 2.0 # tolerance on mass ratio
num_neigh = 50

sim       = "AbacusSummit_highbase_c000_ph100"
zout      = "z0.500"
cat_dir   = "/mnt/gosling2/bigsims/%s/"%(sim) + "halos/" + zout + "/halo_info/"
out_dir   = "/mnt/store1/sbose/scratch/data/compaSO_trees/%s/%s_HOD_halos/"%(sim, sim) + zout
#tree_dir  = "/global/cfs/cdirs/desi/cosmosim/Abacus/merger/%s/"%(sim)
mhist_dir = "/mnt/store1/sbose/scratch/data/%s/MergerHistory_Final_z0.500"%(sim)
ihist_dir = "/mnt/store1/sbose/scratch/data/%s/IndexHistory_Final_z0.500"%(sim)
#box       = 1000.0
#half_box  = 0.5*box

af   = asdf.open(mhist_dir+".000.asdf")
nout = af.tree["data"]["MassHistory"].shape[1]
af.close()

odir      = out_dir + "/kappa_%2.1f"%(factor)

if not os.path.exists(odir):
    os.makedirs(odir)

myrank    = MPI.COMM_WORLD.Get_rank()
i         = myrank
size      = MPI.COMM_WORLD.Get_size()

# Load mass history
#print("Loading halo mass history...")
#af  = asdf.open(mhist_dir)
#afi = asdf.open(ihist_dir)
#mhist   = af.tree["data"]["MassHistory"]
#indhist = af.tree["data"]["IndexHistory"]
#indhist = afi.tree["IndexHistory"]
#mpeak   = af.tree["data"]["Mpeak"]

# Compute mass ratio
#ratio   = mpeak / mhist[0]

# Define a function that retrieves file names on either side of slab of interest
def return_search_list(string_name, counter):
    search_list_next = []
    F = sorted(glob.glob(string_name))
    nfiles = len(F)
    if counter == (nfiles-1):
        search_list_next.append(F[counter:counter+1])
        search_list_next = list(np.array(search_list_next).flat)
        search_list_next.append(F[0])
    else:
        search_list_next.append(F[counter:counter+2])
        search_list_next = list(np.array(search_list_next).flat)
    search_list_next.insert(0, F[counter-1])
    return search_list_next

def return_associations_list(string_name, counter):
    search_list_next = []
    F = sorted(glob.glob(string_name), key = lambda x: int(x.split(".")[-2]))
    nfiles = len(F)
    if counter == (nfiles-1):
        search_list_next.append(F[counter:counter+1])
        search_list_next = list(np.array(search_list_next).flat)
        search_list_next.append(F[0])
    else:
        search_list_next.append(F[counter:counter+2])
        search_list_next = list(np.array(search_list_next).flat)
    search_list_next.insert(0, F[counter-1])
    return search_list_next


tree_dt = np.dtype([
("HaloMass", np.float32),
], align=True)

def read_multi_tree(alist):
    afs	  = [asdf.open(alistname, lazy_load=True, copy_arrays=True) for alistname in alist]
    N_halo_per_file = np.array([len(af["data"]["HaloMass"]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=tree_dt[col]) for col in tree_dt.names}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
        rawhalos = Table(data={field:af.tree["data"][field] for field in ["HaloMass"]}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()

    return halos


hist_dt = np.dtype([
("IndexHistory", np.int64, nout),
("MassHistory", np.float32, nout),
("Mpeak", np.float32),
], align=True)

def read_multi_history(alist, fields):
    afs	  = [asdf.open(alistname, lazy_load=True, copy_arrays=True) for alistname in alist]
    N_halo_per_file = np.array([len(af["data"][fields[0]]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=hist_dt[col]) for col in fields}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
        rawhalos = Table(data={field:af.tree["data"][field] for field in fields}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()

    return halos

@njit
def return_indexed(index, arr1, arr2):
    if index < arr1.shape[1]:
        return arr1[:, index]
    else:
        indexnow = index-arr1.shape[1]
        return arr2[:, indexnow]

@njit
def return_indexed_vec(index_array, arr1, arr2):
    out  = np.zeros(len(index_array))
    ind1 = np.where(index_array<arr1.shape[0])[0]
    out1 = arr1[index_array[ind1]]
    ind2 = np.where(index_array>=arr1.shape[0])[0]
    numnow = index_array[ind2] - arr1.shape[0]
    out2 = arr2[numnow]
    out[ind1] = out1; out[ind2] = out2
    return out

idx_delete = []; tot_mass=[]; x=[]; y=[]; z=[]; tot_vmax=[]

idx_start  = 0
idx_end    = 0

# Find out total number of superslabs per halo_info
all_files   = sorted(glob.glob(cat_dir + "/halo_info*.asdf"))
num_files   = len(all_files)
ifiles      = np.arange(num_files)
ifiles_todo = ifiles[::1]

t_start = time.time()
# Loop over individual superslabs
for ii in ifiles_todo:
#for ii in range(2):

    print("Doing file %d"%ii)

    # Load halo mass history
    mhist_files = return_associations_list(mhist_dir + ".*.asdf", ii)
    ihist_files = return_associations_list(ihist_dir + ".*.asdf", ii)
    mass_file   = read_multi_history(mhist_files, ["Mpeak", "MassHistory"])
    index_file  = read_multi_history(ihist_files, ["IndexHistory"])
    mhist_temp  = mass_file["MassHistory"] # Rows: haloes, Columns: redshift
    ihist_temp  = index_file["IndexHistory"]
    mpeak       = mass_file["Mpeak"]

    # Compute mass ratio
    ratio_temp  = mpeak / mhist_temp[:,0]

    # Load catalogue
    halo_info_list = return_search_list(cat_dir + "/halo_info_*.asdf", ii)
    print("Loading halo_info files ", halo_info_list)
    cat = CompaSOHaloCatalog(halo_info_list, load_subsamples=False, convert_units=True, fields = "corr")
    nhalos = cat.numhalos
    pos = cat.halos[:]["x_L2com"]
    if ii == 0:
        box = cat.header["BoxSize"]
        half_box = 0.5*box
    # Load associations file
    #associations_list = return_associations_list(tree_dir + "associations_z0.500.*.asdf", ii)
    #print("Loading associations files ", associations_list)
    #halos    = read_multi_tree(associations_list)
    #HaloMass = halos["HaloMass"]
    HaloMass = cat.halos["N"]
    HaloMass_temp = np.copy(HaloMass)
    merged_halo = np.zeros(len(HaloMass_temp), dtype="int")-1
    #break
    if ii > 0:
        merged_halo[:nhalos[0]] = merged_halo_store # We will need this later
    '''
    if ii == 0:
        idx_start  = -nhalos[0]
        idx_end    = np.sum(nhalos[1:])
        ratio_temp = np.append(ratio[idx_start:], ratio[:idx_end])
        #mhist_temp = np.hstack((mhist[:, idx_start:], mhist[:, :idx_end]))
        #ihist_temp = np.hstack((indhist[:, idx_start:], indhist[:, :idx_end]))
        mhist_temp1 = mhist[:, idx_start:]; mhist_temp2 = mhist[:, :idx_end]
        ihist_temp1 = indhist[:, idx_start:]; ihist_temp2 = indhist[:, :idx_end]
        print(idx_start, idx_end, len(HaloMass))
        # Update indices
        idx_start += nhalos[0]
    elif (ii > 0) and (ii < num_files-1):
        idx_end    = idx_start+len(HaloMass)
        ratio_temp = ratio[idx_start : idx_end]
        mhist_temp = mhist[:, idx_start : idx_end]
        ihist_temp = indhist[:, idx_start : idx_end]
        print(idx_start, idx_end, len(HaloMass))
        idx_start += nhalos[0]
    elif (ii == num_files-1):
        idx_end    = nhalos[2]
        ratio_temp = np.append(ratio[idx_start:], ratio[:idx_end])
        #mhist_temp = np.hstack((mhist[:, idx_start:], mhist[:, :idx_end]))
        #ihist_temp = np.hstack((indhist[:, idx_start:], indhist[:, :idx_end]))
        mhist_temp1 = mhist[:, idx_start:]; mhist_temp2 = mhist[:, :idx_end]
        ihist_temp1 = indhist[:, idx_start:]; ihist_temp2 = indhist[:, :idx_end]
    '''
    # Identify which haloes belong to current superslab
    mask_index = np.arange(nhalos[0], nhalos[0]+nhalos[1])
    ratio_temp = ratio_temp[mask_index]

    # Update halo masses from previous iteration
    if ii > 0:
        HaloMass_temp[:nhalos[0]+nhalos[1]] = HaloMass_store
    if ii == num_files-1:
        # Add the mass added to these haloes from the first iteration
        assert len(mask_index) == len(HaloMass_diff_final)
        HaloMass_temp[mask_index] += HaloMass_diff_final
        # Finally, we need to update the halo masses from superslab 0
        assert nhalos[2] == len(HaloMass_first)
        HaloMass_temp[nhalos[0]+nhalos[1]:] = HaloMass_first

    '''
    if idx_start == 0:
        idx_end    = nhalos[1]
        ratio_temp = ratio[idx_start : idx_end]
        mhist_temp = mhist[:, idx_start : idx_end]
        ihist_temp = indhist[:, idx_start : idx_end]
        # Update indices
        idx_start  = idx_end
    else:
        idx_end   += nhalos[1]
        ratio_temp = ratio[idx_start : idx_end]
        mhist_temp = mhist[:, idx_start : idx_end]
	ihist_temp = indhist[:, idx_start : idx_end]
        idx_start  = idx_end
    '''
    '''
    # If we are in the final superslab set, we need only process the first superslab of the triplet, as the other two will be done
    # E.g if there are 16 superslabs in total, by the time we get to file number 15, we need only process file 14 in the triple [14, 15, 0]
    # since [15, 0] will have already been done in the very first step
    if ii == num_files-1:
        ratio_temp = ratio_temp[:nhalos[0]]
        mhist_temp = mhist_temp[:,:nhalos[0]]
        ihist_temp = ihist_temp[:,:nhalos[0]]
    '''
    J          = np.where(ratio_temp >= factor)[0]
    # The *global* index of these objects is given by
    J          = mask_index[J]

    #if (ii == 0) or (ii == num_files-1):
    #    assert len(HaloMass) == mhist_temp1.shape[1] + mhist_temp2.shape[1]
    #else:
    assert len(HaloMass) == mhist_temp.shape[0]

    print("Building tree and finding nearest neighbours...")
    # Make a tree of halo positions
    tree     = cKDTree(pos+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)
    # Find neighbours
    neigh    = tree.query(pos[J]+half_box, k=num_neigh, n_jobs=-1)[1]
    '''
    if (ii == 0) or (ii == num_files-1):
        for jj in trange(len(J)):
            mhist_now = return_indexed(J[jj], mhist_temp1, mhist_temp2)
            # Find the epoch at which the maximum mass was attained
            arg_now   = np.argmax(mhist_now)
            # Find the halo(es) where this progenitor exists
            ihist_neigh = return_indexed_vec(neigh[jj], ihist_temp1[arg_now], ihist_temp2[arg_now]).astype(int)
            ihist_now   = return_indexed(J[jj], ihist_temp1, ihist_temp2)[arg_now]
            arg_min     = np.where(ihist_neigh == ihist_now)[0]
            arg_min   = neigh[jj][arg_min]
            # Find the most massive halo in this class
            mask2_arg = np.argmax(HaloMass_temp[arg_min])
            mask2     = arg_min[mask2_arg]
            # Can't have the halo find "itself"
            if not mask2 == J[jj]:
                HaloMass_temp[mask2] += HaloMass_temp[J[jj]]
                HaloMass_temp[J[jj]]  = 0.0
                merged_halo[J[jj]]    = return_indexed(mask2, ihist_temp1, ihist_temp2)[0]
    else:
    '''
    for jj in trange(len(J)):
        mhist_now = mhist_temp[J[jj],:]
        # Find the epoch at which the maximum mass was attained
        arg_now   = np.argmax(mhist_now)
        # Find the halo(es) where this progenitor exists
        arg_min   = np.where(ihist_temp[neigh[jj], arg_now] == ihist_temp[J[jj], arg_now])[0]
        arg_min   = neigh[jj][arg_min]
        # Find the most massive halo in this class
        mask2_arg = np.argmax(HaloMass_temp[arg_min])
        mask2     = arg_min[mask2_arg]
        # Can't have the halo find "itself"
        if not mask2 == J[jj]:
            HaloMass_temp[mask2] += HaloMass_temp[J[jj]]
            HaloMass_temp[J[jj]]  = 0
            merged_halo[J[jj]]    = ihist_temp[mask2,0]

    # Store the updated halo masses for the next iteration
    HaloMass_store = HaloMass_temp[nhalos[0]:]
    # Store list of merged haloes
    merged_halo_store = merged_halo[mask_index]

    if ii == 0:
        # We also want to keep track of how much mass we added to haloes in the superslab to the left of file 0, so that it can be added at the end
        HaloMass_diff_final = HaloMass_temp[:nhalos[0]] - HaloMass[:nhalos[0]]
    if ii == 1:
        # This is the last time file 0 will be updated. Store this for when the last superslab is being processed
        HaloMass_first = HaloMass_temp[:nhalos[0]]

    if ii > 0:
        # We can now create the "new" halo_info file
        data_tree = {"halos": {
            "N_total": HaloMass_temp[:nhalos[0]],
            "halo_global_index": ihist_temp[:nhalos[0],0],
            "halo_info_index": np.arange(0, nhalos[0]),
            "is_merged_to": merged_halo[:nhalos[0]]
            }
        }
        outfile = asdf.AsdfFile(data_tree)
        outfile.write_to(odir + "/cleaned_halo_info_%03d_tmp.asdf"%(ii-1))
        outfile.close()

    if (ii == num_files-1):
        # First, store the data for this file
        data_tree = {"halos": {
            "N_total": HaloMass_temp[mask_index],
            "halo_global_index": ihist_temp[mask_index,0],
            "halo_info_index": np.arange(0, nhalos[1]),
            "is_merged_to": merged_halo_store
            }
        }
        outfile = asdf.AsdfFile(data_tree)
        outfile.write_to(odir + "/cleaned_halo_info_%03d_tmp.asdf"%(ii))
        outfile.close()

        # We now also need to do the final mass updates for file 0
        ff = asdf.open(odir + "/cleaned_halo_info_000_tmp.asdf")
        halo_info_index = ff.tree["halos"]["halo_info_index"]
        is_merged_to = ff.tree["halos"]["is_merged_to"]
        assert len(halo_info_index) == nhalos[2]
        data_tree = {"halos": {
            "N_total": HaloMass_temp[nhalos[0]+nhalos[1]:],
            "halo_global_index": ihist_temp[nhalos[0]+nhalos[1]:, 0],
            "halo_info_index": halo_info_index,
            "is_merged_to": is_merged_to
            }
        }

        outfile = asdf.AsdfFile(data_tree)
        outfile.write_to(odir + "/cleaned_halo_info_000_tmp.asdf")
        outfile.close()

t_end = time.time()
print("Total runtime was: %4.2fs"%(t_end-t_start))

'''
#for ii in trange(len(J)):

    #idx_delete.append(J[ii])
#    mask2_arg = np.argmax( HaloMass[neigh[ii]] )
#    mask2     = neigh[ii][mask2_arg]
#    HaloMass_temp[mask2] += HaloMass[J[ii]]


print("Cleaning up...")
#IND_DEL  = np.concatenate(idx_delete)

# TEST
#IND_DEL = J





TOT_MASS = np.array(tot_mass)
x = np.array(x); y = np.array(y); z = np.array(z)

TOT_MASS = np.concatenate(TOT_MASS)
x = np.concatenate(x); y = np.concatenate(y); z = np.concatenate(z)

mask_clean  = np.where(TOT_MASS > 0.0)[0]
mask_del    = np.where(TOT_MASS == 0.0)[0]
mhalo_clean = TOT_MASS[mask_clean]
X = x[mask_clean]; Y = y[mask_clean]; Z = z[mask_clean]


#mhalo_clean = np.delete(HaloMass_temp, IND_DEL)
#X = np.delete(pos[:,0], IND_DEL); Y = np.delete(pos[:,1], IND_DEL); Z = np.delete(pos[:,2], IND_DEL)

assert len(X) == len(mhalo_clean)
assert len(Y) == len(mhalo_clean)
assert len(Z) == len(mhalo_clean)

# Add new entries
#mhalo_clean = np.append(mhalo_clean, TOT_MASS)
#X = np.append(X,x); Y = np.append(Y,y); Z = np.append(Z,z)

print("Sorting masses...")
sorter = np.argsort(mhalo_clean)[::-1]

M_sort = mhalo_clean[sorter]; X_sort = X[sorter]; Y_sort = Y[sorter]; Z_sort = Z[sorter]

X_sort += box; Y_sort += box; Z_sort += box

print("Saving data...")
np.save(out_dir + "z0.500_M_200c_sorted_agg_v1_%2.1f.npy"%factor, M_sort)
np.save(out_dir + "z0.500_X_sorted_agg_v1_%2.1f.npy"%factor, X_sort)
np.save(out_dir + "z0.500_Y_sorted_agg_v1_%2.1f.npy"%factor, Y_sort)
np.save(out_dir + "z0.500_Z_sorted_agg_v1_%2.1f.npy"%factor, Z_sort)
np.save(out_dir + "z0.500_IND_DELETED_agg_v1_%2.1f.npy"%factor, mask_del)
'''
