#!/usr/bin/env python
#! Filename: extract_mergerhistory_slabwise_slim_mpi.py

from __future__ import division
from scipy.stats import binned_statistic
from Abacus.fast_cksum.cksum_io import CksumWriter
from compaso_halo_catalog import CompaSOHaloCatalog
import match_searchsorted as ms
from scipy.spatial import cKDTree
#from mpi4py import MPI
from tqdm import *
import numpy as np
import h5py as h5
import warnings
import time
import asdf
import glob
import sys
import os
import gc

import astropy.table
from astropy.table import Table

from asdf import AsdfFile, Stream

asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)

warnings.filterwarnings("ignore")

if len(sys.argv) < 5:
    sys.exit("python extract_mergerhistory_slabwise_slim_mpi.py base sim snapshot outdir")

basedir  = sys.argv[1]
sim      = sys.argv[2]
snapin   = float(sys.argv[3])
outdir   = sys.argv[4]
snapshot = "z%4.3f"%(snapin)

#myrank   = MPI.COMM_WORLD.Get_rank()
#i        = myrank
#size     = MPI.COMM_WORLD.Get_size()

num_neigh = 50
factor    = 2.0

#basedir  = "/home/sbose/analysis/data/%s"%(sim)
#basedir  = "/mnt/store/sbose/%s"%(sim)

#basedir += "/%s"%(sim)

#base     = basedir + "/associations_StepNr_%d"%(snapin)
#base    += ".%d.asdf"
#print(basedir)
nfiles   = len(glob.glob( basedir + "/merger/%s/"%(sim) +"associations_z0.500.*.asdf" ))
unique_files = glob.glob( basedir + "/merger/%s/"%(sim) + "associations_z*.0.asdf" )

# Since some halo_info output times != association output times
halo_unique_files = glob.glob( basedir + sim + "/halos/z*" )

print( basedir + "/merger/%s/"%(sim) + "associations_z*.0.asdf" )
sys.stdout.flush()

# Get box size
af       = asdf.open(unique_files[0])
box      = af["header"]["BoxSize"]
mpart    = af["header"]["ParticleMassHMsun"]
half_box = 0.5*box

print("Simulation: ", sim)
print("Snapshot: ", snapshot)
sys.stdout.flush()

# Get the relevant list of snapshot numbers

snapList  = sorted([float(sub.split('z')[-1][:-7]) for sub in unique_files])
snapList  = np.array(snapList)

halo_snapList = sorted([float(sub.split('z')[-1]) for sub in halo_unique_files])
halo_snapList = np.array(halo_snapList)

argSnap   = np.argmin(abs(snapList - snapin))
snapList  = snapList[argSnap:]
halo_snapList = halo_snapList[argSnap:]
halo_snapList = halo_snapList[:len(snapList)]

Nsnapshot = len(snapList)

# Bins

Min       = -4.
Max       = 0.02
Nbins     = 24
bins      = np.logspace(Min, Max, Nbins)

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

tree_dt = np.dtype([
("HaloIndex", np.int64),
("HaloMass", np.float32),
("HaloVmax", np.float32),
("Position", np.float32, 3),
("MainProgenitor", np.int64),
("NumProgenitors", np.int8),
], align=True)

tree_dt_slim = np.dtype([
("HaloIndex", np.int64),
("HaloMass", np.float32),
("HaloVmax", np.float32),
("MainProgenitor", np.int64),
], align=True)

def read_multi_tree(alist):
    afs	  = [asdf.open(alistname, lazy_load=True, copy_arrays=True) for alistname in alist]
    N_halo_per_file = np.array([len(af["data"]["HaloMass"]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=tree_dt[col]) for col in tree_dt.names}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
        rawhalos = Table(data={field:af["data"][field] for field in tree_dt.names}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()

    return halos, N_halo_per_file

def read_multi_progenitors(alist):
    afs   = [asdf.open(alistname, lazy_load=True, copy_arrays=True) for alistname in alist]
    N_progs_per_file = np.array([af["data"]["NumProgenitors"].sum() for af in afs])
    N_progs = N_progs_per_file.sum()
    cols  = {col:np.empty(N_progs, dtype=np.int64) for col in ["Progenitors"]}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
        rawhalos = Table(data={field:af["data"][field] for field in ["Progenitors"]}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()

    return halos, N_progs_per_file

def read_multi_tree_slim(alist):
    afs   = [asdf.open(alistname, lazy_load=True, copy_arrays=True) for alistname in alist]
    N_halo_per_file = np.array([len(af["data"]["HaloMass"]) for af in afs])
    N_halos = N_halo_per_file.sum()
    cols  = {col:np.empty(N_halos, dtype=tree_dt_slim[col]) for col in tree_dt_slim.names}
    halos = Table(cols, copy=False)
    N_written = 0
    for af in afs:
        rawhalos = Table(data={field:af["data"][field] for field in tree_dt_slim.names}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()
        
    return halos, N_halo_per_file

odir = outdir + "/%s/"%sim + snapshot + "/"

if not os.path.exists(odir):
	os.makedirs(odir, exist_ok=True)

tstart=time.time()
io_time=0.0
info_read_time=0.0
sort_time=0.0
match_time=0.0
tree_time=0.0
temp_save_time=0.0
write_time=0.0
n_fixed=0

#for ii in range(nfiles):
for i, ii in enumerate(range(nfiles)):
    #if i%size != myrank: continue

    #print("Superslab number: %d (of %d) being done by processor %d"%(ii, nfiles, myrank))
    print("Superslab number: %d"%(ii))
    sys.stdout.flush()
    '''
    print("Loading associations for snapshot %d of %d"%(1,Nsnapshot))
    sys.stdout.flush()
    HaloIndex, HaloMass, HaloVmax, MainProg, isSplit, MainProgPrec, NumProg, Progenitors = joinFile( basedir + \
        "/associations_z%4.3f."%(snapList[0]) + "%d.asdf", 0, nfiles)

    ncent     = len(HaloMass)
    print("Found %d haloes of interest!"%(ncent))
    sys.stdout.flush()
    '''
    file_list_now  = return_associations_list(basedir + "/merger/%s/"%(sim) + "associations_z%4.3f."%(snapList[0]) + "*.asdf", ii)
    #cat_list_now   = return_search_list(basedir + "/halos/z%4.3f/halo_info/"%(snapList[0]) + "halo_info*.asdf", ii)
    t0 = time.time()
    halos, numhalos = read_multi_tree(file_list_now)#asdf.open(basedir + "/associations_z%4.3f."%(snapList[0]) + "%d.asdf"%ii)
    t1 = time.time()
    io_time += (t1-t0)
    #cat             = CompaSOHaloCatalog(cat_list_now, None, cleaned_halos=False, load_subsamples=False, convert_units=True, unpack_bits=False, fields=["sigmav3d_L2com"], cleaned_fields="all", verbose=False)
    HaloIndex       = halos["HaloIndex"]
    HaloMass        = halos["HaloMass"]
    #HaloVmax        = halos["HaloVmax"]
    #HaloVrms        = cat.halos["sigmav3d_L2com"]
    #assert len(HaloVdisp) == len(HaloMass)
    HaloMass        = (HaloMass.data / mpart).astype(int)
    MainProg        = halos["MainProgenitor"]
    Position        = halos["Position"]
    NumProgenitors  = halos["NumProgenitors"]

    # Compute progenitor offset list here
    cindex          = np.append(0, NumProgenitors)
    offset_array    = np.cumsum(cindex)

    progsTable, numprogs = read_multi_progenitors(file_list_now)
    Progenitors     = progsTable["Progenitors"].data

    args_now        = np.arange(numhalos[0], numhalos[0]+numhalos[1])
    mmax            = np.copy(HaloMass[args_now])
    indmax          = np.copy(HaloIndex[args_now])
    HaloIndex_Start = np.copy(HaloIndex)
    HaloMass_Start  = np.copy(HaloMass)
    MainProg_Start  = np.copy(MainProg)
    #triplet_mergeto = np.zeros(len(args_now), dtype="int") - 1
    is_merged_to    = np.zeros(len(args_now), dtype=np.int64) - 1

    print("Building tree...")
    sys.stdout.flush()
    tree           = cKDTree(Position+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)

    # Being loop over output times
    #ff = AsdfFile()
    #ff_index = AsdfFile()
    #ff.tree["HaloMass"]    = HaloMass
    #ff.tree["MassHistory"] = Stream([len(HaloMass)], np.float32)
    #ff_index.tree["IndexHistory"] = Stream([len(HaloMass)], np.int64)
    #ff.tree["IndexHistory"] = Stream([len(HaloMass)], np.int64)

    #with open(odir + "MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii), "wb") as fd, open(odir + "IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii), "wb") as fdi:
        #ff.write_to(fd)
        #ff_index.write_to(fdi)

    for jj in range(1, Nsnapshot):

        # Load next snapshot
        print("Loading associations for snapshot %d of %d"%(jj+1,Nsnapshot))
        sys.stdout.flush()
        file_list_next = return_associations_list(basedir + "/merger/%s/"%(sim) + "associations_z%4.3f."%(snapList[jj]) + "*.asdf", ii)
        cat_list_next  = return_search_list(basedir + sim + "/halos/z%4.3f/halo_info/"%(halo_snapList[jj]) + "halo_info*.asdf", ii)
        t0 = time.time()
        halos_next, numhalos_next = read_multi_tree_slim(file_list_next)
        t00 = time.time()
        if jj == 1:
            cat_next       = CompaSOHaloCatalog(cat_list_next, None, cleaned_halos=False, load_subsamples=False, convert_units=True, unpack_bits=False, fields=["sigmav3d_L2com", "v_L2com"], cleaned_fields="all", verbose=False)
        else:
            cat_next       = CompaSOHaloCatalog(cat_list_next, None, cleaned_halos=False, load_subsamples=False, convert_units=True, unpack_bits=False, fields=["sigmav3d_L2com"], cleaned_fields="all", verbose=False)
        t1 = time.time()
        io_time += (t1-t0)
        info_read_time += (t1-t00)
        HaloIndexNext  = halos_next["HaloIndex"]
        HaloMassNext   = halos_next["HaloMass"]
        HaloVmaxNext   = halos_next["HaloVmax"]
        HaloVrmsNext   = cat_next.halos["sigmav3d_L2com"]
        if jj == 1:
            HaloVelNext = cat_next.halos["v_L2com"]
        assert len(HaloVrmsNext) == len(HaloMassNext)
        HaloMassNext   = (HaloMassNext.data / mpart).astype(int)
        MainProgNext   = halos_next["MainProgenitor"]

        #print(NumProg.max(), NumProgNext.max())
        #print("Sorting halo indices...")
        #sys.stdout.flush()
        t0 = time.time()
        # Sort halo indices
        sort = np.argsort(HaloIndexNext)
        HaloIndexNextSorted = HaloIndexNext[sort]
        t1 = time.time()
        sort_time += (t1-t0)
        #print("Sorting took %4.2fs."%(t1-t0))
        #sys.stdout.flush()

        if ii == 0:
            t0 = time.time()
            match_index = ms.match(MainProg, HaloIndexNextSorted, arr2_sorted = True)
            t1 = time.time()
            #print("Matching took %4.2fs."%(t1-t0))
            match_time += (t1-t0)
            #sys.stdout.flush()
            # Find the entries where no main progenitor found
            mask = np.where(match_index == -1)[0]
            # Get the matched indices in terms of the unsorted array
            match_index = sort[match_index]
            mbp_mass = HaloMassNext[match_index]
            mbp_idx  = HaloIndexNext[match_index]
            mbp_vmax = HaloVmaxNext[match_index]
            mbp_vrms = HaloVrmsNext[match_index]
            if jj == 1:
                mbp_vel = HaloVelNext[match_index]
            #sys.exit()
            MainProg = MainProgNext[match_index]

            mbp_mass[mask] = 0
            mbp_idx[mask]  = 0
            mbp_vmax[mask] = 0.0
            mbp_vrms[mask] = 0.0
            if jj == 1:
                mbp_vel[mask, :] = 0.0
            MainProg[mask] = 0
            mask_central   = mask[(mask>=args_now[0])&(mask<=args_now[-1])]
            mbp_idx[mask_central]  = -999
            MainProg[mask_central] = -999

        else:
            print("Reading pre-saved match information...")
            mbp_idx   = np.zeros(numhalos.sum(), dtype=np.int64)
            mbp_mass  = np.zeros(numhalos.sum(), dtype=np.uint32)
            mbp_vmax  = np.zeros(numhalos.sum(), dtype=np.float32)
            mbp_vrms  = np.zeros(numhalos.sum(), dtype=np.float32)
            if jj == 1:
                mbp_vel = np.zeros((numhalos.sum(), 3), dtype=np.float32)
            if jj > 1:
                MainProg  = np.zeros(numhalos.sum(), dtype=np.int64)
            prev_slab = np.load(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            prev_mass = np.load(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            prev_vmax = np.load(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            prev_vrms = np.load(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            if jj == 1:
                prev_vel = np.load(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            if jj > 1:
                prev_prog = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], ii-1))
            assert len(prev_slab) == numhalos[0]
            mbp_idx[:numhalos[0]]  = prev_slab
            mbp_mass[:numhalos[0]] = prev_mass
            mbp_vmax[:numhalos[0]] = prev_vmax
            mbp_vrms[:numhalos[0]] = prev_vrms
            if jj == 1:
                mbp_vel[:numhalos[0],:] = prev_vel
            if jj > 1:
                MainProg[:numhalos[0]] = prev_prog
            #os.remove(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            #os.remove(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            #if jj > 1:
            #    os.remove(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], ii-1))
            mid_slab  = np.load(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            mid_mass  = np.load(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            mid_vmax  = np.load(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            mid_vrms  = np.load(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            if jj == 1:
                mid_vel = np.load(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            if jj > 1:
                mid_prog  = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], ii))
            assert len(mid_slab) == numhalos[1]
            mbp_idx[numhalos[0]:numhalos[0]+numhalos[1]]  = mid_slab
            mbp_mass[numhalos[0]:numhalos[0]+numhalos[1]] = mid_mass
            mbp_vmax[numhalos[0]:numhalos[0]+numhalos[1]] = mid_vmax
            mbp_vrms[numhalos[0]:numhalos[0]+numhalos[1]] = mid_vrms
            if jj == 1:
                mbp_vel[numhalos[0]:numhalos[0]+numhalos[1],:] = mid_vel
            if jj > 1:
                MainProg[numhalos[0]:numhalos[0]+numhalos[1]] = mid_prog
            #os.remove(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            #os.remove(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            #if jj > 1:
            #    os.remove(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], ii))
            if jj > 1:
                if ii < nfiles-1:
                    end_prog  = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], ii+1))
                else:
                    end_prog  = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj-1], 0))
                assert len(end_prog) == numhalos[2]
                MainProg[numhalos[0]+numhalos[1]:] = end_prog
            inds_to_match = np.where(mbp_idx == 0)[0]
            t0 = time.time()
            match_index = ms.match(MainProg[inds_to_match], HaloIndexNextSorted, arr2_sorted = True)
            t1 = time.time()
            match_time += (t1-t0)
            #print("Matching took %4.2fs."%(t1-t0))
            #sys.stdout.flush()
            mask = np.where(match_index == -1)[0]
            # Get the matched indices in terms of the unsorted array
            match_index = sort[match_index]
            mbp_mass[inds_to_match] = HaloMassNext[match_index]
            mbp_idx[inds_to_match]  = HaloIndexNext[match_index]
            mbp_vmax[inds_to_match] = HaloVmaxNext[match_index]
            mbp_vrms[inds_to_match] = HaloVrmsNext[match_index]
            if jj == 1:
                mbp_vel[inds_to_match,:] = HaloVelNext[match_index]
            # Update MainProg to MainProgNext
            #if jj > 1:
            prev_prog_next = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], ii-1))
            MainProg[:numhalos[0]]  = prev_prog_next
            mid_prog_next  = np.load(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], ii))
            MainProg[numhalos[0]:numhalos[0]+numhalos[1]] = mid_prog_next
            MainProg[inds_to_match] = MainProgNext[match_index]

            mbp_mass[inds_to_match[mask]] = 0
            mbp_idx[inds_to_match[mask]]  = 0
            mbp_vmax[inds_to_match[mask]] = 0.0
            mbp_vrms[inds_to_match[mask]] = 0.0
            if jj == 1:
                mbp_vel[inds_to_match[mask], :] = 0.0
            MainProg[inds_to_match[mask]] = 0
            #mask_central   = mask[(mask>=args_now[0])&(mask<=args_now[-1])]
            mask_central = np.where((inds_to_match[mask]>=args_now[0])&(inds_to_match[mask]<=args_now[-1]))[0]
            mbp_idx[inds_to_match[mask_central]]  = -999
            MainProg[inds_to_match[mask_central]] = -999
        #print(MainProg)

        mass_diff           = mbp_mass[args_now] - mmax
        mask_update         = np.where(mass_diff > 0)[0]
        mmax[mask_update]   = mbp_mass[args_now][mask_update]
        indmax[mask_update] = mbp_idx[args_now][mask_update]
        pos_now             = Position[args_now]

        mass_ratio          = mmax.astype(float) / HaloMass_Start[args_now]
        halos_to_fix        = np.where((mass_ratio >= factor)&(mass_diff>0.0))[0]

        if len(halos_to_fix) > 0:

            print("Finding neighbours for %d haloes that need fixing."%(len(halos_to_fix)))
            sys.stdout.flush()

            # Find neighbours
            t0 = time.time()
            neigh = tree.query(pos_now[halos_to_fix]+half_box, k=num_neigh, n_jobs=-1)[1]
            #neigh = tree.query_ball_point(pos_now[halos_to_fix]+half_box, r=3.0)
            t1 = time.time()
            tree_time += (t1-t0)
            # Loop over dodgy ones
            for nn in range(len(halos_to_fix)):
                halo_now   = halos_to_fix[nn]
                indmax_now = indmax[halo_now]
                arg_min    = np.where(mbp_idx[neigh[nn]] == indmax_now)[0]
                arg_min    = neigh[nn][arg_min]
                # Find the most massive halo
                mask2_arg  = np.argmax(HaloMass_Start[arg_min])
                mask2      = arg_min[mask2_arg]

                # Update the mbp_mass
                #mbp_mass[mask2] += mbp_mass[args_now[halo_now]]

                if not mask2 == args_now[halo_now]:
                    #triplet_mergeto[halo_now] = mask2
                    is_merged_to[halo_now]    = HaloIndex_Start[mask2]
                else:
                    for kk in range(len(neigh[nn])):
                        offset = offset_array[neigh[nn][kk]]
                        end = offset + NumProgenitors[neigh[nn][kk]]
                        progs_now = Progenitors[offset:end]
                        if (MainProg_Start[args_now[halo_now]] in progs_now) & (neigh[nn][kk] != args_now[halo_now]):
                            is_merged_to[halo_now] = HaloIndex_Start[neigh[nn][kk]]
                            n_fixed += 1
                            break

        t0 = time.time()
        # We can create some temporary files so that objects that are matched can be recalled later
        np.save(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), mbp_idx.data[numhalos[0]:numhalos[0]+numhalos[1]])
        np.save(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), mbp_mass.data[numhalos[0]:numhalos[0]+numhalos[1]])
        np.save(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), mbp_vmax.data[numhalos[0]:numhalos[0]+numhalos[1]])
        np.save(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), mbp_vrms.data[numhalos[0]:numhalos[0]+numhalos[1]])
        np.save(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), MainProg.data[numhalos[0]:numhalos[0]+numhalos[1]])
        if jj == 1:
            np.save(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], ii), mbp_vel.data[numhalos[0]:numhalos[0]+numhalos[1]])
        if ii == 0:
            file_prev = nfiles-1
            np.save(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), mbp_idx.data[:numhalos[0]])
            np.save(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), mbp_mass.data[:numhalos[0]])
            np.save(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), mbp_vmax.data[:numhalos[0]])
            np.save(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), mbp_vrms.data[:numhalos[0]])
            np.save(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), MainProg.data[:numhalos[0]])
            if jj == 1:
                np.save(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], file_prev), mbp_vel.data[:numhalos[0]])
            file_next = ii+1
            np.save(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_idx.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_mass.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vmax.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vrms.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), MainProg.data[numhalos[0]+numhalos[1]:])
            if jj == 1:
                np.save(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vel.data[numhalos[0]+numhalos[1]:])
        elif (ii > 0) and (ii < nfiles-1):
            file_next = ii+1
            np.save(odir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_idx.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_mass.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vmax.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vrms.data[numhalos[0]+numhalos[1]:])
            np.save(odir + "/temporary_prog_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), MainProg.data[numhalos[0]+numhalos[1]:])
            if jj == 1:
                np.save(odir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[jj], file_next), mbp_vel.data[numhalos[0]+numhalos[1]:])
        t1 = time.time()
        temp_save_time += (t1-t0)
        '''
        if jj == 1:
            fd.write(np.array(HaloMass, np.float32).tostring())
            fd.write(np.array(mbp_mass, np.float32).tostring())
            fdi.write(np.array(HaloIndex, np.int64).tostring())
            fdi.write(np.array(mbp_idx, np.int64).tostring())
            #MBP = np.row_stack((HaloIndex, mbp_idx))
        else:
            fd.write(np.array(mbp_mass, np.float32).tostring())
            fdi.write(np.array(mbp_idx, np.int64).tostring())
            #MBP = np.row_stack((MBP, mbp_idx))
        '''
        #if jj == Nsnapshot-1:
        #    ff.tree["IndexHistory"] = MBP

        #TODO: implement MainProgPrecNext check.

        # IN DEVELOPMENT
        # Match the remaining progenitors
        '''
        match_progs = ms.match(Progenitors, HaloIndexNextSorted, arr2_sorted = True)
        match_progs = sort[match_progs]
        prog_masses = HaloMassNext[match_progs]

        for nn in trange(mbp_mass.shape[1]):
            if mbp_mass[jj, nn] == 0.:
                continue
            prog_now_start  = np.sum(NumProg[:nn])
            prog_now_end    = prog_now_start + NumProg[nn]
            progs_this_halo = prog_masses[prog_now_start:prog_now_end]
            ratio           = progs_this_halo / mbp_mass[jj, nn]
            N, d1, d2       = binned_statistic(ratio, ratio, "count", bins = bins)
        '''
        # TODO: HAVE TO ADD THESE DATA TO THE DATA TREE. IN ASDF STREAM MODE?

        #HaloIndex = mbp_idx[jj]
        #MainProg       = MainProgNext[match_index]
        #MainProg[mask] = -999
        gc.collect()


    #MainProgPrec = MainProgPrecNext
    #Progenitors = ProgenitorsNext

    print("Cleanups...")
    sys.stdout.flush()
    asdf_fn = odir + "MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii)
    #ff = asdf.open(asdf_fn)
    #MassHistory = ff.tree["MassHistory"]

    data_tree = {"data": {
    "HaloGlobalIndex": np.array(HaloIndex_Start[args_now]),
    "HaloIndexPeak": np.array(indmax),
    #"IndexHistory": MBP,
        "HaloNpartPeak": np.array(mmax),
        #"TripletMergeTo": np.array(triplet_mergeto),
        "IsMergedTo": np.array(is_merged_to)}}


    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(odir + "/MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii))
    #outfile.close()
    t0 = time.time()
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")
    t1 = time.time()
    write_time += (t1-t0)
    '''
    asdf_fn = odir + "IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii)
    ff = asdf.open(asdf_fn)
    IndexHistory = ff.tree["IndexHistory"]
    data_tree = {"data": {
    "IndexHistory": np.array(IndexHistory.T)
    }}

    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(odir + "/IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii))
    #outfile.close()
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")
    '''
#ff.tree["IndexHistory"] = MBP

tend=time.time()
print("Processing complete in %4.2fs!"%(tend-tstart))
print("Total IO time: %4.2fs"%io_time)
print("Of which halo_info read time: %4.2fs"%info_read_time)
print("Total tree query time: %4.2fs"%tree_time)
print("Total sort time: %4.2fs"%sort_time)
print("Total match time: %4.fs"%match_time)
print("Total temporary save time: %4.2fs"%temp_save_time)
print("Total write time: %4.2fs"%write_time)
print("Total number of contact haloes fixed: %d"%n_fixed)
sys.stdout.flush()
