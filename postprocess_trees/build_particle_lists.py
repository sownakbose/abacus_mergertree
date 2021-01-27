#!/usr/bin/env python
#! Filename: build_particle_lists.py

from __future__ import division
from compaso_halo_catalog import CompaSOHaloCatalog
from Abacus.fast_cksum.cksum_io import CksumWriter
from astropy.table import Table
from mpi4py import MPI
from numba import njit
from tqdm import *
import match_searchsorted as ms
import numpy as np
import warnings
import asdf
import glob
import gc

warnings.filterwarnings("ignore")
asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)

sim       = "AbacusSummit_highbase_c000_ph100"
zout      = "z0.500"
cat_dir   = "/mnt/gosling2/bigsims/%s/"%(sim) + "halos/" + zout + "/halo_info/"
merge_dir = "/mnt/store1/sbose/scratch/data/%s/"%(sim)
clean_dir = "/mnt/store1/sbose/scratch/data/compaSO_trees/%s/%s_HOD_halos/"%(sim, sim) + zout + "/kappa_2.0/"

unq_prev_files = sorted(glob.glob(merge_dir + "temporary_mass_matches_z*.000.npy"))
Nsnapshot = len(unq_prev_files)
snapList  = sorted([float(sub.split('z')[-1][:-8]) for sub in unq_prev_files])

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

def float_trunc(a, zerobits):
    """Set the least significant <zerobits> bits to zero in a numpy float32 or float64 array.
    Do this in-place. Also return the updated array.
    Maximum values of 'nzero': 51 for float64; 22 for float32.
    """
    at = a.dtype
    assert at == np.float64 or at == np.float32 or at == np.complex128 or at == np.complex64
    if at == np.float64 or at == np.complex128:
        assert zerobits <= 51
        mask = 0xffffffffffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint64)
        bits &= mask
    elif at == np.float32 or at == np.complex64:
        assert zerobits <= 22
        mask = 0xffffffff - (1 << zerobits) + 1
        bits = a.view(np.uint32)
        bits &= mask

    return a

clean_dt = np.dtype([
    #("N_total", np.uint32),
    ("IsMergedTo", np.int64),
    ("HaloGlobalIndex", np.uint64),
    ], align=True)

part_dt  = np.dtype([
    ("rvint_A", np.int32),
    ("rvint_B", np.int32),
    ], align=True)

index_dt = np.dtype([
    ("N_total", np.uint32),
    ("N_merge", np.uint32),
    ("N_mainprog", np.uint32, Nsnapshot),
    ("vcirc_max_L2com_mainprog", np.float32, Nsnapshot),
    ("sigmav3d_L2com_mainprog", np.float32, Nsnapshot),
    ("npstartA_merge", np.int64),
    ("npoutA_merge", np.uint32),
    ("npoutA_L0L1_merge", np.uint32),
    ("npstartB_merge", np.int64),
    ("npoutB_merge", np.uint32),
    ("npoutB_L0L1_merge", np.uint32),
    ], align=True)

def read_multi_cat(file_list):
    afs   = [asdf.open(file_listname, lazy_load=True, copy_arrays=True) for file_listname in file_list]
    N_halo_per_file = np.array([len(af["data"]["HaloGlobalIndex"]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=clean_dt[col]) for col in clean_dt.names}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
          rawhalos = Table(data={field:af.tree["data"][field] for field in clean_dt.names}, copy=False)
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

nfiles_to_do = len(glob.glob(merge_dir + "MergerHistory_Final*.asdf_test"))

for i, ii in enumerate(range(nfiles_to_do)):
#for i, ii in enumerate(range(0, 1)):

    if i % size != myrank: continue
    print("Superslab number : %d (of %d) being done by processor %d"%(ii, nfiles_to_do, myrank))

    # Load cleaned catalogue files
    clean_list = return_search_list(merge_dir + "MergerHistory_Final*.asdf_test", ii)
    clean_cat  = read_multi_cat(clean_list)
    merged_to  = clean_cat["IsMergedTo"].data
    #N_total    = clean_cat["N_total"]
    global_ind = clean_cat["HaloGlobalIndex"]

    # Load halo info files
    info_list  = return_search_list(cat_dir + "halo_info*.asdf", ii)
    cat        = CompaSOHaloCatalog(info_list, clean_path=None, cleaned_halos=False, load_subsamples="AB_halo_pidrvint", convert_units=True, fields=["N", "npstartA", "npstartB", "npoutA", "npoutB"], unpack_bits=False)
    #cat        = CompaSOHaloCatalog(info_list, clean_path=None, cleaned_halos=False, load_subsamples=False, convert_units=True, fields=["N", "npstartA", "npstartB", "npoutA", "npoutB"], unpack_bits=False)
    N          = cat.halos["N"].data
    nstartA    = cat.halos["npstartA"].data
    nstartB    = cat.halos["npstartB"].data
    noutA      = cat.halos["npoutA"].data
    noutB      = cat.halos["npoutB"].data
    header     = cat.header
    numhalos   = cat.numhalos # Objects in superslab of interest run from [numhalos[0]:numhalos[0]+numhalos[1]]
    pidrvint_sub  = cat.subsamples
    num_subsamples = np.array([len(pidrvint_sub)], dtype=np.uint64)

    # Let's redefine noutA, noutB so that we also include L0 particles
    A_start_indices = np.append(nstartA, nstartB[0])
    B_start_indices = np.append(nstartB, num_subsamples)
    noutA_L0L1 = np.diff(A_start_indices).astype(np.uint32)
    noutB_L0L1 = np.diff(B_start_indices).astype(np.uint32)

    indices_that_merge   = np.where(merged_to != -1)[0]
    nhalos_this_slab     = numhalos[1]

    header["NumTimeSliceRedshiftsPrev"] = Nsnapshot
    header["TimeSliceRedshiftsPrev"]    = np.array(snapList)

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
    particles.add_column(np.empty(tot_particles_to_add, dtype=np.uint64), name="pid_A", copy=False)
    particles.add_column(np.empty(tot_particles_to_add, dtype=np.uint64), name="pid_B", copy=False)

    # Creat Table for indexing
    cols_index = {col:np.zeros(nhalos_this_slab, dtype=index_dt[col]) for col in index_dt.names}
    p_indexing = Table(cols_index, copy=False)

    # For N_mainprog, we need a Table whose size = numhalos.sum()
    cols_index = {col:np.zeros(numhalos.sum(), dtype=index_dt[col]) for col in ["N_mainprog"]}
    mass_indexing = Table(cols_index, copy=False)

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

    # Finally, we want to fill out the progenitor information
    for mm in trange(Nsnapshot):
        #mass_mainprog = np.load(merge_dir + "temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
        vmax_mainprog = np.load(merge_dir + "temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
        vrms_mainprog = np.load(merge_dir + "temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
        #mass_mainprog[deleted_halos] = 0
        vmax_mainprog[deleted_halos] = 0.0
        vrms_mainprog[deleted_halos] = 0.0
        #data_tree["data"]["N_mainprog_z%4.3f"%(snapList[mm])] = mass_mainprog
        #data_tree["data"]["vcirc_max_L2com_mainprog_z%4.3f"%(snapList[mm])] = vmax_mainprog
        #data_tree["data"]["sigmav3d_L2com_mainprog_z%4.3f"%(snapList[mm])] = vrms_mainprog
        #p_indexing["N_mainprog"][:, mm] = mass_mainprog
        vmax_mainprog = float_trunc(vmax_mainprog, 12)
        vrms_mainprog = float_trunc(vrms_mainprog, 12)
        p_indexing["vcirc_max_L2com_mainprog"][:, mm] = vmax_mainprog
        p_indexing["sigmav3d_L2com_mainprog"][:, mm] = vrms_mainprog


    # We have to fill mass_indexing from three superslabs, and for Nsnapshot
    # First, grab the superslabs of interest
    slabs = [sub.split('.')[-2] for sub in clean_list]
    for nn in trange(3):
        if nn == 0:
            start = 0
            end   = numhalos[0]
        elif nn == 1:
            start = numhalos[0]
            end   = numhalos[0]+numhalos[1]
        else:
            start = numhalos[0]+numhalos[1]
            end   = None
        for mm in range(Nsnapshot):
            mass_mainprog = np.load(merge_dir + "temporary_mass_matches_z%4.3f.%s.npy"%(snapList[mm], slabs[nn]))
            mass_indexing["N_mainprog"][start:end, mm] = mass_mainprog

    N_mainprog = mass_indexing["N_mainprog"].data

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
        extra_subsampA_L0L1     = np.sum(noutA_L0L1[extra_halo_global_index])
        extra_subsampB_L0L1     = np.sum(noutB_L0L1[extra_halo_global_index])
        extra_subsampA          = np.sum(noutA[extra_halo_global_index])
        extra_subsampB          = np.sum(noutB[extra_halo_global_index])

        N_mainprog_thishalo_initial = np.copy(N_mainprog[indices_this_slab[haloid]])
        # Correct main progenitor info
        for extra in extra_halo_global_index:

            N_mainprog_extra = N_mainprog[extra]
            N_difference     = N_mainprog_thishalo_initial - N_mainprog_extra
            mask             = N_difference != 0
            N_mainprog[indices_this_slab[haloid]][mask] += N_mainprog_extra[mask]
            N_mainprog[extra] = 0

        #assert (N[indices_this_slab[haloid]]+extra_npart) == N_total[indices_this_slab[haloid]]

        # Update indices
        p_indexing["N_merge"][haloid]        = extra_npart
        p_indexing["npstartA_merge"][haloid] = counter_A
        p_indexing["npstartB_merge"][haloid] = counter_B
        p_indexing["npoutA_merge"][haloid]   = extra_subsampA
        p_indexing["npoutB_merge"][haloid]   = extra_subsampB
        p_indexing["npoutA_L0L1_merge"][haloid]   = extra_subsampA_L0L1
        p_indexing["npoutB_L0L1_merge"][haloid]   = extra_subsampB_L0L1

        # Write subsample A particles
        for nn in range(len(extra_halo_global_index)):
            halo_now = extra_halo_global_index[nn]
            # Add subsample A rvints
            particles["rvint_A"][counter_A:counter_A+noutA_L0L1[halo_now]] = pidrvint_sub["rvint"][nstartA[halo_now]:nstartA[halo_now]+noutA_L0L1[halo_now]]
            # Add subsample A pids
            particles["pid_A"][counter_A:counter_A+noutA_L0L1[halo_now]] = pidrvint_sub["pid"][nstartA[halo_now]:nstartA[halo_now]+noutA_L0L1[halo_now]]
            counter_A += noutA_L0L1[halo_now]
            # Add subsample B rvints
            particles["rvint_B"][counter_B:counter_B+noutB_L0L1[halo_now]] = pidrvint_sub["rvint"][nstartB[halo_now]:nstartB[halo_now]+noutB_L0L1[halo_now]]
            # Add subsample B pids
            particles["pid_B"][counter_B:counter_B+noutB_L0L1[halo_now]] = pidrvint_sub["pid"][nstartB[halo_now]:nstartB[halo_now]+noutB_L0L1[halo_now]]
            counter_B += noutB_L0L1[halo_now]

    p_indexing["N_total"] = N[numhalos[0]:numhalos[0]+numhalos[1]] + p_indexing["N_merge"]
    p_indexing["N_total"][deleted_halos] = 0
    #break

    # Done looping over haloes in this file. Time to write out data
    data_tree = {"data":{
        "is_merged_to": merged_to[numhalos[0]:numhalos[0]+numhalos[1]],
        "N_total": p_indexing["N_total"].data,
        "N_merge": p_indexing["N_merge"].data,
        "N_mainprog": N_mainprog[indices_this_slab],
        "vcirc_max_L2com_mainprog": p_indexing["vcirc_max_L2com_mainprog"].data,
        "sigmav3d_L2com_mainprog": p_indexing["sigmav3d_L2com_mainprog"].data,
        "npoutA_L0L1": noutA_L0L1[numhalos[0]:numhalos[0]+numhalos[1]],
        "npoutB_L0L1": noutB_L0L1[numhalos[0]:numhalos[0]+numhalos[1]],
        "npstartA_merge": p_indexing["npstartA_merge"].data,
        "npoutA_merge": p_indexing["npoutA_merge"].data,
        "npoutA_L0L1_merge": p_indexing["npoutA_L0L1_merge"].data,
        "npstartB_merge": p_indexing["npstartB_merge"].data,
        "npoutB_merge": p_indexing["npoutB_merge"].data},
        "npoutB_L0L1_merge": p_indexing["npoutB_L0L1_merge"].data,
        "header": header
    }

    print("Writing out data....")

    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(clean_dir + "/cleaned_halo_info_%03d.asdf"%ii)
    #outfile.close()
    asdf_fn = clean_dir + "/cleaned_halo_info_%03d.asdf"%ii
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")

    # Write out new particles
    data_tree = {"data":{
    "packedpid_A": particles["pid_A"][:counter_A].data,
    "packedpid_B": particles["pid_B"][:counter_B].data,
    "rvint_A": particles["rvint_A"][:counter_A].data, # Trim to last filled value
    "rvint_B": particles["rvint_B"][:counter_B].data},
    "header": header
    }

    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(clean_dir + "/cleaned_rvpid_%03d.asdf"%ii)
    #outfile.close()
    asdf_fn = clean_dir + "/cleaned_rvpid_%03d.asdf"%ii
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")

    gc.collect()
