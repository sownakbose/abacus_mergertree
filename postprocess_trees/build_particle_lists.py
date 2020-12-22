#!/usr/bin/env python
#! Filename: build_particle_lists.py

from __future__ import division
from compaso_halo_catalog import CompaSOHaloCatalog
from astropy.table import Table
from mpi4py import MPI
from numba import njit
from tqdm import *
import match_searchsorted as ms
import numpy as np
import warnings
import asdf
import glob

warnings.filterwarnings("ignore")

sim       = "AbacusSummit_highbase_c000_ph100"
zout      = "z0.500"
cat_dir   = "/mnt/gosling2/bigsims/%s/"%(sim) + "halos/" + zout + "/halo_info/"
clean_dir = "/mnt/store1/sbose/scratch/data/compaSO_trees/%s/%s_HOD_halos/"%(sim, sim) + zout + "/kappa_2.0/"

size      = MPI.COMM_WORLD.Get_size()
myrank    = MPI.COMM_WORLD.Get_rank()

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

clean_dt = np.dtype([
    ("N_total", np.uint32),
    ("is_merged_to", np.int64),
    ("halo_global_index", np.uint64),
    ("halo_info_index", np.uint32),
    ], align=True)

part_dt  = np.dtype([
    ("rvint_A", np.int32),
    ("rvint_B", np.int32),
    ], align=True)

index_dt = np.dtype([
    ("N_total", np.uint32),
    ("N_merge", np.uint32),
    ("npstartA_merge", np.int64),
    ("npoutA_merge", np.uint32),
    ("npstartB_merge", np.int64),
    ("npoutB_merge", np.uint32),
    ], align=True)

def read_multi_cat(file_list):
    afs   = [asdf.open(file_listname, lazy_load=True, copy_arrays=True) for file_listname in file_list]
    N_halo_per_file = np.array([len(af["halos"]["halo_info_index"]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=clean_dt[col]) for col in clean_dt.names}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
          rawhalos = Table(data={field:af.tree["halos"][field] for field in clean_dt.names}, copy=False)
          halos[N_written:N_written+len(rawhalos)] = rawhalos
          N_written += len(rawhalos)
          af.close()
    return halos

def unpack_inds(halo_ids):
    id_factor   = 1000000000000 # 1e12
    slab_factor = 1000000000 # 1e9
    index       = (halo_ids%slab_factor).astype(int)
    slab_number = ((halo_ids%id_factor-index)//slab_factor).astype(int)
    return slab_number, index

nfiles_to_do = len(glob.glob(clean_dir + "hod_halo_info*.asdf"))

for i, ii in enumerate(range(nfiles_to_do)):
#for i, ii in enumerate(range(1, 2)):

    if i % size != myrank: continue
    print("Superslab number : %d (of %d) being done by processor %d"%(ii, nfiles_to_do, myrank))

    # Load cleaned catalogue files
    clean_list = return_search_list(clean_dir + "cleaned_halo_info*_v1.asdf", ii)
    clean_cat  = read_multi_cat(clean_list)
    merged_to  = clean_cat["is_merged_to"]
    N_total    = clean_cat["N_total"]
    global_ind = clean_cat["halo_global_index"]

    # Load halo info files
    info_list  = return_search_list(cat_dir + "halo_info*.asdf", ii)
    cat        = CompaSOHaloCatalog(info_list, load_subsamples="AB_halo_rvint", convert_units=True, fields="merger", unpack_bits=False)
    N          = cat.halos["N"]
    nstartA    = cat.halos["npstartA"]
    nstartB    = cat.halos["npstartB"]
    noutA      = cat.halos["npoutA"]
    noutB      = cat.halos["npoutB"]
    numhalos   = cat.numhalos # Objects in superslab of interest run from [numhalos[0]:numhalos[0]+numhalos[1]]
    rvint_sub  = cat.subsamples


    indices_that_merge   = np.where(merged_to != -1)[0]
    nhalos_this_slab     = numhalos[1]

    # In some rare instances, objects that are deleted have received merged particles. We need to reassign these particles.
    # First, find the "troublesome" objects that are marked with 0 particles but also receive merged particles
    mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
    hosts_to_reassign   = indices_that_merge[mask_to_reassign]
    while len(hosts_to_reassign) > 0:
        # Now, find the halos whose hosts are the "troublesome" objects
        matches = ms.match(merged_to, global_ind[hosts_to_reassign], arr2_sorted=False)
        halos_to_reassign   = np.where(matches != -1)[0]
        print("Found %d halos to reassign."%(len(halos_to_reassign)))

        # Reassign their hosts as the hosts of the "troublesome" objects themselves
        merged_to[halos_to_reassign] = merged_to[hosts_to_reassign][matches[halos_to_reassign]]

        mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
        hosts_to_reassign   = indices_that_merge[mask_to_reassign]

    # Check one more level
    #mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
    #hosts_to_reassign   = indices_that_merge[mask_to_reassign]
    #if not len(hosts_to_reassign) == 0:
    #    matches = ms.match(merged_to, global_ind[hosts_to_reassign], arr2_sorted=False)
    #    halos_to_reassign   = np.where(matches != -1)[0]
    #    merged_to[halos_to_reassign] = merged_to[hosts_to_reassign][matches[halos_to_reassign]]


    # Particle list will never be bigger than this
    tot_particles_to_add = int(np.sum(noutB[indices_that_merge]))

    # Create Table for particle list
    cols       = {col:np.empty((tot_particles_to_add, 3), dtype=part_dt[col]) for col in ["rvint_A", "rvint_B"]}
    particles  = Table(cols, copy=False)
    # Creat Table for indexing
    cols_index = {col:np.zeros(nhalos_this_slab, dtype=index_dt[col]) for col in index_dt.names}
    p_indexing = Table(cols_index, copy=False)

    # Make the index columns sufficiently different so that there are no confusions with indexing
    p_indexing["npstartA_merge"][:] = -999
    p_indexing["npstartB_merge"][:] = -999

    indices_this_slab   = np.arange(numhalos[0], numhalos[0]+numhalos[1])
    merged_to_this_slab = merged_to[indices_this_slab]
    deleted_halos = np.where(merged_to_this_slab != -1)[0]
    #p_indexing["is_deleted"][deleted_halos] = 1

    print("Done reading all data. Now sorting indices...")
    sort       = np.argsort(merged_to)
    merged_to_sorted = merged_to[sort]
    print("Finding unique indices...")
    unq_haloidx, array_idx, ncount = np.unique(merged_to_sorted, return_counts=True, return_index=True)

    # The first ones are all going to be minus -1
    assert unq_haloidx[0] == -1
    counter_A = 0; counter_B = 0

    for jj in trange(1, len(unq_haloidx)):

        # Get the slab number
        slabid, haloid = unpack_inds(unq_haloidx[jj])
        if slabid != ii: continue

        extra_halo              = array_idx[jj]
        extra_halo_global_index = sort[extra_halo:extra_halo+ncount[jj]]

        extra_npart             = np.sum(N[extra_halo_global_index])
        extra_subsampA          = np.sum(noutA[extra_halo_global_index])
        extra_subsampB          = np.sum(noutB[extra_halo_global_index])

        assert (N[indices_this_slab[haloid]]+extra_npart) == N_total[indices_this_slab[haloid]]

        # Update indices
        p_indexing["N_merge"][haloid]        = extra_npart
        p_indexing["npstartA_merge"][haloid] = counter_A
        p_indexing["npstartB_merge"][haloid] = counter_B
        p_indexing["npoutA_merge"][haloid]   = extra_subsampA
        p_indexing["npoutB_merge"][haloid]   = extra_subsampB

        # Write subsample A particles
        for nn in range(len(extra_halo_global_index)):
            halo_now = extra_halo_global_index[nn]
            particles["rvint_A"][counter_A:counter_A+noutA[halo_now]] = rvint_sub["rvint"][nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
            counter_A += noutA[halo_now]
            particles["rvint_B"][counter_B:counter_B+noutB[halo_now]] = rvint_sub["rvint"][nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
            counter_B += noutB[halo_now]

    #p_indexing["N_total"] = N[numhalos[0]:numhalos[0]+numhalos[1]] + p_indexing["N_merge"]
    #p_indexing["N_total"][deleted_halos] = 0
    #break
    print("Writing out data....")

    # Done looping over haloes in this file. Time to write out data
    data_tree = {"halos": {
        "halo_info_index": clean_cat["halo_info_index"][numhalos[0]:numhalos[0]+numhalos[1]],
        "is_merged_to": merged_to[numhalos[0]:numhalos[0]+numhalos[1]],
        "N_total": p_indexing["N_total"],
        "N_merge": p_indexing["N_merge"],
        "npstartA_merge": p_indexing["npstartA_merge"],
        "npoutA_merge": p_indexing["npoutA_merge"],
        "npstartB_merge": p_indexing["npstartB_merge"],
        "npoutB_merge": p_indexing["npoutB_merge"]
        }
    }

    outfile = asdf.AsdfFile(data_tree)
    outfile.write_to(clean_dir + "/cleaned_halo_info_%03d.asdf"%ii)
    outfile.close()

    # Write out new particles
    data_tree = {"particles":{
    "rvint_A": particles["rvint_A"][:counter_A], # Trim to last filled value
    "rvint_B": particles["rvint_B"][:counter_B]
        }
    }

    outfile = asdf.AsdfFile(data_tree)
    outfile.write_to(clean_dir + "/cleaned_pidrv_%03d.asdf"%ii)
    outfile.close()
