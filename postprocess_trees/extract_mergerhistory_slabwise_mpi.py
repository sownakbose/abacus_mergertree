#!/usr/bin/env python
#! Filename: extract_mergerhistory.py

from __future__ import division
from scipy.stats import binned_statistic
from Abacus.fast_cksum.cksum_io import CksumWriter
import match_searchsorted as ms
from mpi4py import MPI
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

warnings.filterwarnings("ignore")

if len(sys.argv) < 5:
    sys.exit("python extract_mergerhistory.py base sim snapshot outdir")

basedir  = sys.argv[1]
sim      = sys.argv[2]
snapin   = float(sys.argv[3])
outdir   = sys.argv[4]

myrank   = MPI.COMM_WORLD.Get_rank()
i        = myrank
size     = MPI.COMM_WORLD.Get_size()

#basedir  = "/home/sbose/analysis/data/%s"%(sim)
#basedir  = "/mnt/store/sbose/%s"%(sim)

basedir += "/%s"%(sim)

#base     = basedir + "/associations_StepNr_%d"%(snapin)
#base    += ".%d.asdf"
nfiles   = len(glob.glob( basedir + "/associations_z0.100.*.asdf" ))
unique_files = glob.glob( basedir + "/associations_z*.0.asdf" )

print("Simulation: ", sim)
sys.stdout.flush()

# Get the relevant list of snapshot numbers

snapList  = sorted([float(sub.split('z')[-1][:-7]) for sub in unique_files])
snapList  = np.array(snapList)

argSnap   = np.argmin(abs(snapList - snapin))
snapList  = snapList[argSnap:]

Nsnapshot = len(snapList)

# TEST
#Nsnapshot = 3

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

tree_dt = np.dtype([
("HaloIndex", np.int64),
("HaloMass", np.float32),
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
        rawhalos = Table(data={field:af.tree["data"][field] for field in ["HaloIndex", "HaloMass", "MainProgenitor"]}, copy=False)
        halos[N_written:N_written+len(rawhalos)] = rawhalos
        N_written += len(rawhalos)
        af.close()

    return halos

odir = outdir + "/%s/"%sim

if not os.path.exists(odir):
	os.makedirs(odir, exist_ok=True)

tstart=time.time()
#for ii in range(nfiles):
for i, ii in enumerate(range(nfiles)):

    if i%size != myrank: continue

    print("Superslab number: %d (of %d) being done by processor %d"%(ii, nfiles, myrank))
    '''
    print("Loading associations for snapshot %d of %d"%(1,Nsnapshot))
    sys.stdout.flush()
    HaloIndex, HaloMass, HaloVmax, MainProg, isSplit, MainProgPrec, NumProg, Progenitors = joinFile( basedir + \
        "/associations_z%4.3f."%(snapList[0]) + "%d.asdf", 0, nfiles)

    ncent     = len(HaloMass)
    print("Found %d haloes of interest!"%(ncent))
    sys.stdout.flush()
    '''
    halos      = asdf.open(basedir + "/associations_z%4.3f."%(snapList[0]) + "%d.asdf"%ii)
    HaloIndex  = halos["data"]["HaloIndex"]
    HaloMass   = halos["data"]["HaloMass"]
    MainProg   = halos["data"]["MainProgenitor"]
    mmax       = np.copy(HaloMass)

    # Being loop over output times
    ff = AsdfFile()
    ff_index = AsdfFile()
    #ff.tree["HaloMass"]    = HaloMass
    ff.tree["MassHistory"] = Stream([len(HaloMass)], np.float32)
    ff_index.tree["IndexHistory"] = Stream([len(HaloMass)], np.int64)
    #ff.tree["IndexHistory"] = Stream([len(HaloMass)], np.int64)

    with open(odir + "MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii), "wb") as fd, open(odir + "IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii), "wb") as fdi:
        ff.write_to(fd)
        ff_index.write_to(fdi)

        for jj in trange(1, Nsnapshot):
        #for jj in range(1, 3):

            # Load next snapshot
            #print("Loading associations for snapshot %d of %d"%(jj+1,Nsnapshot))
            #sys.stdout.flush()
            file_list_next = return_associations_list(basedir + "/associations_z%4.3f."%(snapList[jj]) + "*.asdf", ii)
            halos_next     = read_multi_tree(file_list_next)
            HaloIndexNext  = halos_next["HaloIndex"]
            HaloMassNext   = halos_next["HaloMass"]
            MainProgNext   = halos_next["MainProgenitor"]
            #print(NumProg.max(), NumProgNext.max())

            #print("Sorting halo indices...")
            #sys.stdout.flush()
            t0 = time.time()
            # Sort halo indices
            sort = np.argsort(HaloIndexNext)
            HaloIndexNextSorted = HaloIndexNext[sort]
            t1 = time.time()
            #print("Sorting took %4.2fs."%(t1-t0))
            #sys.stdout.flush()

            match_index = ms.match(MainProg, HaloIndexNextSorted, arr2_sorted = True)
            #print("Done matching haloes.")
            #sys.stdout.flush()

            # Find the entries where no main progenitor found
            mask = np.where(match_index == -1)[0]

            # Get the matched indices in terms of the unsorted array
            match_index = sort[match_index]

            #mbp_mass[jj, :] = HaloMassNext[match_index]
            #mbp_vmax[jj, :] = HaloVmaxNext[match_index]
            #mbp_idx[jj, :]  = HaloIndexNext[match_index]

            #mbp_mass[jj, mask] = 0.
            #mbp_vmax[jj, mask] = 0.
            #mbp_idx[jj, mask]  = -999

            mbp_mass = HaloMassNext[match_index]
            mbp_mass[mask] = 0.
            mbp_idx  = HaloIndexNext[match_index]
            mbp_idx[mask]  = -999

            mass_diff         = mbp_mass - mmax
            mask_update       = np.where(mass_diff > 0)[0]
            mmax[mask_update] = mbp_mass[mask_update]

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
            MainProg       = MainProgNext[match_index]
            MainProg[mask] = -999
            gc.collect()

    #MainProgPrec = MainProgPrecNext
    #Progenitors = ProgenitorsNext

    print("Cleanups...")
    sys.stdout.flush()
    asdf_fn = odir + "MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii)
    ff = asdf.open(asdf_fn)
    MassHistory = ff.tree["MassHistory"]
    data_tree = {"data": {
    "MassHistory": MassHistory.T,
    #"IndexHistory": MBP,
        "Mpeak": mmax}}

    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(odir + "/MergerHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii))
    #outfile.close()
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")

    asdf_fn = odir + "IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii)
    ff = asdf.open(asdf_fn)
    IndexHistory = ff.tree["IndexHistory"]
    data_tree = {"data": {
    "IndexHistory": IndexHistory.T
    }}

    #outfile = asdf.AsdfFile(data_tree)
    #outfile.write_to(odir + "/IndexHistory_Final_z%4.3f.%03d.asdf"%(snapin,ii))
    #outfile.close()
    with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
        af.write_to(fp, all_array_compression="blsc")

#ff.tree["IndexHistory"] = MBP

tend=time.time()
print("Processing complete in %4.2fs!"%(tend-tstart))
sys.stdout.flush()
