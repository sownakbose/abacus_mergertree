#!/usr/bin/env python
#! Filename: build_particle_lists.py

from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "6"

from compaso_halo_catalog import CompaSOHaloCatalog
from Abacus.fast_cksum.cksum_io import CksumWriter
from multithreaded_reader import multithreaded_read
from astropy.table import Table
#from mpi4py import MPI
from numba import njit
from tqdm import *
import match_searchsorted as ms
import numpy as np
import subprocess
import warnings
import asdf
import glob
import time
import sys
import gc

warnings.filterwarnings("ignore")
asdf.compression.set_decompression_options(nthreads=4)
asdf.compression.set_compression_options(typesize="auto", shuffle="shuffle", asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4)

#import tracemalloc
#tracemalloc.start()

if len(sys.argv) < 3:
	sys.exit("Usage: python build_particle_lists.py sim snapin")

sim       = sys.argv[1]
snapin    = float(sys.argv[2])
zout      = "z%4.3f"%(snapin)

disk_dir  = "/global/cfs/cdirs/desi/cosmosim/Abacus/"
base_dir  = "/global/cscratch1/sd/sbose/subsample_B_particles/"
#base_dir  = disk_dir

if "small" in sim:
	disk_dir += "/small/"
	base_dir += "/small/"

cat_dir   = base_dir + sim + "/halos/"  + zout + "/halo_info/"

merge_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/mergerhistory/%s/"%(sim) + zout
#merge_dir = "/global/cscratch1/sd/sbose/python/abacus_mergertree/postprocess_trees/test_codes/mergerhistory/%s/"%(sim)+zout
#merge_dir = "./mergerhistory/%s/"%(sim) + zout
if "small" in sim:
	clean_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/cleaned_halos/small/%s/halos/"%(sim) + zout
else:
	clean_dir = "/global/cfs/cdirs/desi/cosmosim/Abacus/cleaned_halos/%s/halos/"%(sim) + zout
#clean_dir = "/global/cscratch1/sd/sbose/python/abacus_mergertree/postprocess_trees/test_codes/cleaned_halos_new/%s/halos/"%(sim) + zout
#clean_dir = "./cleaned_halos/%s/halos/"%(sim) + zout

# Test
#disk_dir = base_dir
#merge_dir = "/global/cscratch1/sd/sbose/subsample_B_particles/mergerhistory/%s/"%(sim)+zout
#clean_dir = "/global/cscratch1/sd/sbose/subsample_B_particles/cleaned_halos/%s/halos/"%(sim)+zout

if not os.path.exists(clean_dir):
	os.makedirs(clean_dir, exist_ok=True)

unq_prev_files = sorted(glob.glob(merge_dir + "/temporary_mass_matches_z*.000.npy"))
Nsnapshot = len(unq_prev_files)
snapList  = sorted([float(sub.split('z')[-1][:-8]) for sub in unq_prev_files])

# Since some halo_info output times != association output times
halo_unique_files = glob.glob( disk_dir + sim + "/halos/z*" )
halo_snapList = sorted([float(sub.split('z')[-1]) for sub in halo_unique_files])
halo_snapList = np.array(halo_snapList)
argSnap   = np.argmin(abs(halo_snapList - snapin))
halo_snapList = halo_snapList[argSnap+1:] # +1 as we want all preceding times
halo_snapList = halo_snapList[:len(snapList)]

#size      = MPI.COMM_WORLD.Get_size()
#myrank    = MPI.COMM_WORLD.Get_rank()

z_primary  = [0.100, 0.200, 0.300, 0.400, 0.500, 0.800, 1.100, 1.400, 1.700, 2.000, 2.500, 3.000]

# Check if this is a primary redshift or not
if os.path.isdir(base_dir + sim + "/halos/"  + zout + "/halo_rv_A"):
	#assert snapin in z_primary
	print("Primary redshift.")
	isPrimary = True
else:
	#assert snapin not in z_primary
	print("Secondary redshift.")
	isPrimary = False

sys.stdout.flush()

def return_search_list(string_name, counter):
	search_list_next = []
	F = sorted(glob.glob(string_name))
	nfiles = len(F)
	if nfiles == 1: return F
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

@njit
def combine_mainprog(extra_halo_index, N_mainprog_host_initial, N_mainprog):
	extra_counter = 1
	N_mainprog_combined = np.zeros((len(extra_halo_index)+1, N_mainprog_host_initial.size))
	N_mainprog_combined[0] = N_mainprog_host_initial
	for extra in extra_halo_index:
		N_mainprog_extra = N_mainprog[extra]
		N_mainprog_combined[extra_counter] = N_mainprog_extra
		extra_counter += 1
		#N_mainprog[extra] = 0
	N_mainprog_final = np.zeros(N_mainprog_combined.shape[1])
	for kk in range(len( N_mainprog_final)):
		N_mainprog_final[kk] = np.sum(np.unique(N_mainprog_combined[:,kk]))
	return N_mainprog_final

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
	#("N_merge", np.uint32),
	#("N_mainprog", np.uint32, Nsnapshot),
	#("vcirc_max_L2com_mainprog", np.float32, Nsnapshot),
	#("sigmav3d_L2com_mainprog", np.float32, Nsnapshot),
	("haloindex_mainprog", np.int64),
	("v_L2com_mainprog", np.float32, 3),
	#("npstartA_merge", np.int64),
	#("npoutA_merge", np.uint32),
	#("npoutA_L0L1_merge", np.uint32),
	#("npstartB_merge", np.int64),
	#("npoutB_merge", np.uint32),
	#("npoutB_L0L1_merge", np.uint32),
	], align=True)

mass_dt = np.dtype([
	("N_mainprog", np.uint32, Nsnapshot),
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

@njit
def unpack_inds(halo_ids):
	id_factor   = 1000000000000 # 1e12
	slab_factor = 1000000000 # 1e9
	index       = halo_ids%slab_factor
	slab_number = (halo_ids%id_factor-index)//slab_factor
	return slab_number, index

@njit
def compute_merge(haloid, counter_A, counter_B, extra_halo_global_index, N, noutA, noutB, N_merge, npstartA_merge, npstartB_merge, npoutA_merge, npoutB_merge):

	extra_npart             = np.sum(N[extra_halo_global_index])
	extra_subsampA          = np.sum(noutA[extra_halo_global_index])
	extra_subsampB          = np.sum(noutB[extra_halo_global_index])

	# Update indices
	N_merge[haloid]        = extra_npart
	npstartA_merge[haloid] = counter_A
	npstartB_merge[haloid] = counter_B
	npoutA_merge[haloid]   = extra_subsampA
	npoutB_merge[haloid]   = extra_subsampB

	return

@njit
def fill_particles(extra_halo_global_index, counter_A, counter_B, rvint_A, rvint_B, pid_A, pid_B, RV, PID, nstartA, noutA, nstartB, noutB, isPrimary=True):
	if isPrimary:
		for nn in range(len(extra_halo_global_index)):
			halo_now = extra_halo_global_index[nn]
			rvint_A[counter_A:counter_A+noutA[halo_now]] = RV[nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
			# Add subsample A pids
			pid_A[counter_A:counter_A+noutA[halo_now]] = PID[nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
			counter_A += noutA[halo_now]
			# Add subsample B rvints
			rvint_B[counter_B:counter_B+noutB[halo_now]] = RV[nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
			# Add subsample B pids
			pid_B[counter_B:counter_B+noutB[halo_now]] = PID[nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
			counter_B += noutB[halo_now]
	else:
		for nn in range(len(extra_halo_global_index)):
			halo_now = extra_halo_global_index[nn]
			# Add subsample A pids
			pid_A[counter_A:counter_A+noutA[halo_now]] = PID[nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
			counter_A += noutA[halo_now]
			# Add subsample B pids
			pid_B[counter_B:counter_B+noutB[halo_now]] = PID[nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
			counter_B += noutB[halo_now]
	return counter_A, counter_B


nfiles_to_do = len(glob.glob(merge_dir + "/MergerHistory_Final*.asdf"))

tbegin = time.time()

for i, ii in enumerate(range(nfiles_to_do)):
#for i, ii in enumerate(range(0, 1)):

	#if i % size != myrank: continue
	#print("Superslab number : %d (of %d) being done by processor %d"%(ii, nfiles_to_do, myrank))
	print("Superslab number: %d (of %d)"%(ii, nfiles_to_do))
	sys.stdout.flush()

	#tstart = time.time()

	# Load cleaned catalogue files
	clean_list = return_search_list(merge_dir + "/MergerHistory_Final*.asdf", ii)
	clean_cat  = read_multi_cat(clean_list)
	merged_to  = clean_cat["IsMergedTo"].data
	#N_total    = clean_cat["N_total"]
	global_ind = clean_cat["HaloGlobalIndex"].data

	# Load halo info files
	info_list  = return_search_list(cat_dir + "halo_info*.asdf", ii)
	if isPrimary:
		cat        = CompaSOHaloCatalog(info_list, clean_path=None, cleaned_halos=False, load_subsamples="AB_halo_pidrvint", convert_units=True, fields=["N", "npstartA", "npstartB", "npoutA", "npoutB"], unpack_bits=False)
	else:
		cat        = CompaSOHaloCatalog(info_list, clean_path=None, cleaned_halos=False, load_subsamples="AB_halo_pid", convert_units=True, fields=["N", "npstartA", "npstartB", "npoutA", "npoutB"], unpack_bits=False)
	#cat        = CompaSOHaloCatalog(info_list, clean_path=None, cleaned_halos=False, load_subsamples=False, convert_units=True, fields=["N", "npstartA", "npstartB", "npoutA", "npoutB"], unpack_bits=False)
	N          = cat.halos["N"].data
	nstartA    = cat.halos["npstartA"].data
	nstartB    = cat.halos["npstartB"].data
	noutA      = cat.halos["npoutA"].data
	noutB      = cat.halos["npoutB"].data
	header     = cat.header
	numhalos   = cat.numhalos # Objects in superslab of interest run from [numhalos[0]:numhalos[0]+numhalos[1]]
	#pidrvint_sub  = cat.subsamples

	# For single superslab case
	nslab = len(numhalos)
	if len(numhalos) == 1: numhalos = np.array([0, numhalos[0], 0])

	PID        = cat.subsamples["pid"].data

	if isPrimary:
		RV         = cat.subsamples["rvint"].data
	else:
		RV         = np.array([0, 0, 0])

	num_subsamples = np.array([len(PID)], dtype=np.uint64)
	cat = []; clean_cat=[]
	# Let's redefine noutA, noutB so that we also include L0 particles
	#A_start_indices = np.append(nstartA, nstartB[0])
	#B_start_indices = np.append(nstartB, num_subsamples)
	#noutA_L0L1 = np.diff(A_start_indices).astype(np.uint32)
	#noutB_L0L1 = np.diff(B_start_indices).astype(np.uint32)

	#tend = time.time()
	#print('Initial read: %4.2fs'%(tend-tstart))

	indices_that_merge   = np.where(merged_to != -1)[0]
	nhalos_this_slab     = numhalos[1]

	header["NumTimeSliceRedshiftsPrev"] = Nsnapshot
	header["TimeSliceRedshiftsPrev"]    = list(halo_snapList)

	# In some rare instances, objects that are deleted have received merged particles. We need to reassign these particles.
	# First, find the "troublesome" objects that are marked with 0 particles but also receive merged particles
	mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
	hosts_to_reassign   = indices_that_merge[mask_to_reassign]
	indices_this_slab   = np.arange(numhalos[0], numhalos[0]+numhalos[1])

	# MEMORY
	#current, peak = tracemalloc.get_traced_memory()
	#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	while len(hosts_to_reassign) > 0:
		# Now, find the halos whose hosts are the "troublesome" objects
		matches = ms.match(merged_to, global_ind[hosts_to_reassign], arr2_sorted=False)
		halos_to_reassign   = np.where(matches != -1)[0]
		print("Found %d halos to reassign."%(len(halos_to_reassign)))
		sys.stdout.flush()

		# Reassign their hosts as the hosts of the "troublesome" objects themselves
		merged_to[halos_to_reassign] = merged_to[hosts_to_reassign][matches[halos_to_reassign]]

		mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
		hosts_to_reassign   = indices_that_merge[mask_to_reassign]

		# Remove objects that are now marked as merging onto "themselves"
		index_diff = global_ind[hosts_to_reassign]-merged_to[hosts_to_reassign]
		mask_remove = np.where(index_diff == 0)[0]
		hosts_to_reassign = np.delete(hosts_to_reassign, mask_remove)

	matches = []
	# Check one more level
	#mask_to_reassign    = np.isin(global_ind[indices_that_merge], merged_to)
	#hosts_to_reassign   = indices_that_merge[mask_to_reassign]
	#if not len(hosts_to_reassign) == 0:
	#    matches = ms.match(merged_to, global_ind[hosts_to_reassign], arr2_sorted=False)
	#    halos_to_reassign   = np.where(matches != -1)[0]
	#    merged_to[halos_to_reassign] = merged_to[hosts_to_reassign][matches[halos_to_reassign]]

	# Particle list will never be bigger than this
	try:
		merge_index_this_slab_min    = np.argmin(abs(indices_that_merge-indices_this_slab[0]))
		merge_index_this_slab_max    = np.argmin(abs(indices_that_merge-indices_this_slab[-1]))
		indices_that_merge_this_slab = indices_that_merge[merge_index_this_slab_min:merge_index_this_slab_max]
		tot_particles_to_add = int(np.sum(noutB[indices_that_merge_this_slab]) * 1.2) # 20% buffer
	except ValueError: # when no indices_that_merge
		tot_particles_to_add = 1

	'''
	# Create Table for particle list
	if isPrimary:
		cols       = {col:np.empty((tot_particles_to_add, 3), dtype=part_dt[col]) for col in ["rvint_A", "rvint_B"]}
		particles  = Table(cols, copy=False)
		particles.add_column(np.empty(tot_particles_to_add, dtype=np.uint64), name="pid_A", copy=False)
		particles.add_column(np.empty(tot_particles_to_add, dtype=np.uint64), name="pid_B", copy=False)
	else:
		cols       = {col:np.empty(tot_particles_to_add, dtype=np.uint64) for col in ["pid_A", "pid_B"]}
		particles  = Table(cols, copy=False)
	'''

	# Create arrays for particle lists
	if isPrimary:
		rvint_A = np.empty((tot_particles_to_add, 3), dtype=np.int32)
		rvint_B = np.empty((tot_particles_to_add, 3), dtype=np.int32)
		pid_A   = np.empty(tot_particles_to_add, dtype=np.uint64)
		pid_B   = np.empty(tot_particles_to_add, dtype=np.uint64)
	else:
		pid_A   = np.empty(tot_particles_to_add, dtype=np.uint64)
		pid_B   = np.empty(tot_particles_to_add, dtype=np.uint64)
		rvint_A = np.empty((1, 3), dtype=np.int32)
		rvint_B = np.empty((1, 3), dtype=np.int32)

	# Create Table for indexing
	cols_index = {col:np.zeros(nhalos_this_slab, dtype=index_dt[col]) for col in index_dt.names}
	p_indexing = Table(cols_index, copy=False)

	# Make the index columns sufficiently different so that there are no confusions with indexing
	#p_indexing["npstartA_merge"][:] = -999
	#p_indexing["npstartB_merge"][:] = -999
	N_merge        = np.zeros(nhalos_this_slab, dtype=np.uint32)
	npstartA_merge = np.zeros(nhalos_this_slab, dtype=np.int32) - 999
	npstartB_merge = np.zeros(nhalos_this_slab, dtype=np.int32) - 999
	npoutA_merge   = np.zeros(nhalos_this_slab, dtype=np.uint32)
	npoutB_merge   = np.zeros(nhalos_this_slab, dtype=np.uint32)

	merged_to_this_slab = merged_to[indices_this_slab]
	deleted_halos = np.where(merged_to_this_slab != -1)[0]
	#p_indexing["is_deleted"][deleted_halos] = 1

	print("Done reading all data. Now sorting indices...")
	sort       = np.argsort(merged_to)
	merged_to_sorted = merged_to[sort]

	#tstart = time.time()

	print("Finding unique indices...")
	unq_haloidx, array_idx, ncount = np.unique(merged_to_sorted, return_counts=True, return_index=True)
	sys.stdout.flush()

	#tend = time.time()
	#print('Took %4.2fs'%(tend-tstart))

	# MEMORY
	#current, peak = tracemalloc.get_traced_memory()
	#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	#tstart = time.time()

	vmax_flist = sorted(glob.glob(merge_dir + "/temporary_vmax_matches_z*.%03d.npy"%ii))
	vrms_flist = sorted(glob.glob(merge_dir + "/temporary_vrms_matches_z*.%03d.npy"%ii))
	vcirc_max_L2com_mainprog = multithreaded_read(vmax_flist, 6)
	sigmav3d_L2com_mainprog  = multithreaded_read(vrms_flist, 6)
	if len(vmax_flist) > 0:
		vcirc_max_L2com_mainprog = float_trunc(np.column_stack(vcirc_max_L2com_mainprog), 12)
		vcirc_max_L2com_mainprog[deleted_halos] = 0.0
	if len(vrms_flist) > 0:
		sigmav3d_L2com_mainprog  = float_trunc(np.column_stack(sigmav3d_L2com_mainprog), 12)
		sigmav3d_L2com_mainprog[deleted_halos]  = 0.0


	'''
	# Finally, we want to fill out the progenitor information
	for mm in range(Nsnapshot):
		#mass_mainprog = np.load(merge_dir + "temporary_mass_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
		vmax_mainprog = np.load(merge_dir + "/temporary_vmax_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
		vrms_mainprog = np.load(merge_dir + "/temporary_vrms_matches_z%4.3f.%03d.npy"%(snapList[mm], ii))
		#mass_mainprog[deleted_halos] = 0
		vmax_mainprog[deleted_halos] = 0.0
		vrms_mainprog[deleted_halos] = 0.0
		#data_tree["data"]["N_mainprog_z%4.3f"%(snapList[mm])] = mass_mainprog
		#data_tree["data"]["vcirc_max_L2com_mainprog_z%4.3f"%(snapList[mm])] = vmax_mainprog
		#data_tree["data"]["sigmav3d_L2com_mainprog_z%4.3f"%(snapList[mm])] = vrms_mainprog
		#p_indexing["N_mainprog"][:, mm] = mass_mainprog
		vmax_mainprog = float_trunc(vmax_mainprog, 12)
		vrms_mainprog = float_trunc(vrms_mainprog, 12)
		try:
			p_indexing["vcirc_max_L2com_mainprog"][:, mm] = vmax_mainprog
			p_indexing["sigmav3d_L2com_mainprog"][:, mm] = vrms_mainprog
		except IndexError:
			p_indexing["vcirc_max_L2com_mainprog"][:] = vmax_mainprog
			p_indexing["sigmav3d_L2com_mainprog"][:] = vrms_mainprog
	'''
	if Nsnapshot > 0:
		# Get the HaloIndex of the MainProgenitor from the previous snapshot
		index_mainprog = np.load(merge_dir + "/temporary_index_matches_z%4.3f.%03d.npy"%(snapList[0], ii))
		index_mainprog[deleted_halos] = 0
		p_indexing["haloindex_mainprog"][:] = index_mainprog
		vel_mainprog = np.load(merge_dir + "/temporary_vel_matches_z%4.3f.%03d.npy"%(snapList[0], ii))
		vel_mainprog[deleted_halos,:] = 0.0
		vel_mainprog = float_trunc(vel_mainprog, 12)
		p_indexing["v_L2com_mainprog"][:] = vel_mainprog

	#tend = time.time()
	#print("Middle progenitor fill time: %4.2fs"%(tend-tstart))

	# The first ones are all going to be minus -1
	assert unq_haloidx[0] == -1
	counter_A = 0; counter_B = 0

	#sys.exit()
	first = 0
	second = 0

	slabid_list = []; haloid_list = []
	for jj in range(1, len(unq_haloidx)):
		#t0 = time.time()
		# Get the slab number
		slabid, haloid = unpack_inds(unq_haloidx[jj])
		slabid_list.append(slabid); haloid_list.append(haloid)
		if slabid != ii: continue

		extra_halo              = array_idx[jj]
		extra_halo_global_index = sort[extra_halo:extra_halo+ncount[jj]]

		compute_merge(haloid, counter_A, counter_B, extra_halo_global_index, N, noutA, noutB, N_merge, npstartA_merge, npstartB_merge, npoutA_merge, npoutB_merge)

		'''
		extra_npart             = np.sum(N[extra_halo_global_index])
		#extra_subsampA_L0L1     = np.sum(noutA_L0L1[extra_halo_global_index])
		#extra_subsampB_L0L1     = np.sum(noutB_L0L1[extra_halo_global_index])
		extra_subsampA          = np.sum(noutA[extra_halo_global_index])
		extra_subsampB          = np.sum(noutB[extra_halo_global_index])

		# Update indices
		p_indexing["N_merge"][haloid]        = extra_npart
		p_indexing["npstartA_merge"][haloid] = counter_A
		p_indexing["npstartB_merge"][haloid] = counter_B
		p_indexing["npoutA_merge"][haloid]   = extra_subsampA
		p_indexing["npoutB_merge"][haloid]   = extra_subsampB
		#p_indexing["npoutA_L0L1_merge"][haloid]   = extra_subsampA_L0L1
		#p_indexing["npoutB_L0L1_merge"][haloid]   = extra_subsampB_L0L1
		'''

		#t1 = time.time()

		#first += (t1-t0)

		#t2 = time.time()

		counter_A, counter_B = fill_particles(extra_halo_global_index, counter_A, counter_B, rvint_A, rvint_B, pid_A, pid_B, RV, PID, nstartA, noutA, nstartB, noutB, isPrimary=isPrimary)

		'''
		if isPrimary:
			for nn in range(len(extra_halo_global_index)):
				halo_now = extra_halo_global_index[nn]
				# Add subsample A rvints
				particles["rvint_A"][counter_A:counter_A+noutA[halo_now]] = pidrvint_sub["rvint"][nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
				# Add subsample A pids
				particles["pid_A"][counter_A:counter_A+noutA[halo_now]] = pidrvint_sub["pid"][nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
				counter_A += noutA[halo_now]
				# Add subsample B rvints
				particles["rvint_B"][counter_B:counter_B+noutB[halo_now]] = pidrvint_sub["rvint"][nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
				# Add subsample B pids
				particles["pid_B"][counter_B:counter_B+noutB[halo_now]] = pidrvint_sub["pid"][nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
				counter_B += noutB[halo_now]
		else:
			for nn in range(len(extra_halo_global_index)):
				halo_now = extra_halo_global_index[nn]
				# Add subsample A pids
				particles["pid_A"][counter_A:counter_A+noutA[halo_now]] = pidrvint_sub["pid"][nstartA[halo_now]:nstartA[halo_now]+noutA[halo_now]]
				counter_A += noutA[halo_now]
				# Add subsample B pids
				particles["pid_B"][counter_B:counter_B+noutB[halo_now]] = pidrvint_sub["pid"][nstartB[halo_now]:nstartB[halo_now]+noutB[halo_now]]
				counter_B += noutB[halo_now]
		'''
		#t3 = time.time()
		#second += (t3-t2)

	#print('First: %4.2fs, Second: %4.2fs'%(first,second))

	#p_indexing["N_total"] = N[numhalos[0]:numhalos[0]+numhalos[1]] + p_indexing["N_merge"]
	p_indexing["N_total"] = N[numhalos[0]:numhalos[0]+numhalos[1]] + N_merge
	p_indexing["N_total"][deleted_halos] = 0
	PID = []; RV = []
	gc.collect()

	#tstart = time.time()
	#sys.exit()
	# For N_mainprog, we need a Table whose size = numhalos.sum()
	#cols_index = {col:np.zeros(numhalos.sum(), dtype=mass_dt[col]) for col in ["N_mainprog"]}
	#mass_indexing = Table(cols_index, copy=False)

	# MEMORY
	#current, peak = tracemalloc.get_traced_memory()
	#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	# We have to fill mass_indexing from three superslabs, and for Nsnapshot
	# First, grab the superslabs of interest
	slabs = [sub.split('.')[-2] for sub in clean_list]

	N_mainprog = np.zeros(numhalos.sum(), dtype=mass_dt["N_mainprog"])
	if nslab > 1:
		for nn in range(3):
			if nn == 0:
				start = 0
				end   = numhalos[0]
			elif nn == 1:
				start = numhalos[0]
				end   = numhalos[0]+numhalos[1]
			else:
				start = numhalos[0]+numhalos[1]
				end   = None
			flist = sorted(glob.glob(merge_dir + "/temporary_mass_matches_z*.%s.npy"%(slabs[nn])))
			if len(flist) > 0:
				mass_mainprog = np.column_stack(multithreaded_read(flist, 6))
				try:
					N_mainprog[start:end,:] = mass_mainprog
				except IndexError:
					assert mass_mainprog.shape[1] == 1
					N_mainprog[start:end]   = mass_mainprog[:,0]
	else:
		flist = sorted(glob.glob(merge_dir + "/temporary_mass_matches_z*.%s.npy"%(slabs[0])))
		if len(flist) > 0:
			mass_mainprog = np.column_stack(multithreaded_read(flist, 6))
			try:
				N_mainprog[:] = mass_mainprog
			except ValueError:
				assert mass_mainprog.shape[1] == 1
				N_mainprog[:] = mass_mainprog[:,0]

	mass_mainprog = []
	'''
	for nn in range(3):
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
			mass_mainprog = np.load(merge_dir + "/temporary_mass_matches_z%4.3f.%s.npy"%(snapList[mm], slabs[nn]))

			try:
				mass_indexing["N_mainprog"][start:end, mm] = mass_mainprog
			except IndexError:
				mass_indexing["N_mainprog"][start:end] = mass_mainprog

	N_mainprog = mass_indexing["N_mainprog"].data
	'''
	mass_indexing=[]

	#tend = time.time()

	#print('Main prog filling: %4.2fs'%(tend-tstart))
	#sys.exit()

	for jj in range(1, len(unq_haloidx)):
		# Get the slab number
		#slabid, haloid = unpack_inds(unq_haloidx[jj])
		slabid, haloid = slabid_list[jj-1], haloid_list[jj-1]
		if slabid != ii: continue

		extra_halo              = array_idx[jj]
		extra_halo_global_index = sort[extra_halo:extra_halo+ncount[jj]]

		N_mainprog_thishalo_initial = np.copy(N_mainprog[indices_this_slab[haloid]])
		# Correct main progenitor info
		N_mainprog[indices_this_slab[haloid]] = combine_mainprog(extra_halo_global_index, N_mainprog_thishalo_initial, N_mainprog)
		for extra in extra_halo_global_index:
			N_mainprog[extra] = 0

	N_mainprog[indices_this_slab][deleted_halos] = 0
	#break
	#sys.exit()
	# MEMORY
	#current, peak = tracemalloc.get_traced_memory()
	#print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	# Free memory
	N=[]; nstartA=[]; nstartB=[]; noutA=[]; noutB=[]; sort=[]
	gc.collect()

	#tstart = time.time()

	# Done looping over haloes in this file. Time to write out data
	data_tree = {"data":{
		"is_merged_to": merged_to[numhalos[0]:numhalos[0]+numhalos[1]],
		"haloindex": global_ind[numhalos[0]:numhalos[0]+numhalos[1]],
		"N_total": p_indexing["N_total"].data.copy(),
		#"N_merge": p_indexing["N_merge"].data.copy(),
		"N_merge": N_merge,
		"N_mainprog": N_mainprog[indices_this_slab],
		#"vcirc_max_L2com_mainprog": p_indexing["vcirc_max_L2com_mainprog"].data.copy(),
		#"sigmav3d_L2com_mainprog": p_indexing["sigmav3d_L2com_mainprog"].data.copy(),
		"vcirc_max_L2com_mainprog": vcirc_max_L2com_mainprog,
		"sigmav3d_L2com_mainprog": sigmav3d_L2com_mainprog,
		"haloindex_mainprog": p_indexing["haloindex_mainprog"].data.copy(),
		"v_L2com_mainprog": p_indexing["v_L2com_mainprog"].data.copy(),
		#"npoutA_L0L1": noutA_L0L1[numhalos[0]:numhalos[0]+numhalos[1]],
		#"npoutB_L0L1": noutB_L0L1[numhalos[0]:numhalos[0]+numhalos[1]],
		#"npstartA_merge": p_indexing["npstartA_merge"].data.copy(),
		#"npoutA_merge": p_indexing["npoutA_merge"].data.copy(),
		"npstartA_merge": npstartA_merge,
		"npoutA_merge": npoutA_merge,
		#"npoutA_L0L1_merge": p_indexing["npoutA_L0L1_merge"].data,
		#"npstartB_merge": p_indexing["npstartB_merge"].data.copy(),
		#"npoutB_merge": p_indexing["npoutB_merge"].data.copy()},
		"npstartB_merge": npstartB_merge,
		"npoutB_merge": npoutB_merge},
		#"npoutB_L0L1_merge": p_indexing["npoutB_L0L1_merge"].data,
		"header": header
	}

	print("Writing out data....")
	sys.stdout.flush()
	#outfile = asdf.AsdfFile(data_tree)
	#outfile.write_to(clean_dir + "/cleaned_halo_info_%03d.asdf"%ii)
	#outfile.close()
	asdf_fn = clean_dir + "/cleaned_halo_info_%03d.asdf"%ii
	with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
		af.write_to(fp, all_array_compression="blsc")

	# Free memory
	data_tree=[]; p_indexing=[]; N_mainprog=[]; merged_to=[]; global_ind=[]; N_merge=[]; npstartA_merge=[]; npstartB_merge=[]; npoutA_merge=[]; npoutB_merge=[]
	vcirc_max_L2com_mainprog=[]; sigmav3d_L2com_mainprog=[]
	gc.collect()

	# Write out new particles

	if isPrimary:
		data_tree = {"data":{
		"packedpid_A": pid_A[:counter_A].copy(),
		"packedpid_B": pid_B[:counter_B].copy(),
		"rvint_A": rvint_A[:counter_A].copy(),
		"rvint_B": rvint_B[:counter_B].copy()},
		"header": header
		}
	else:
		data_tree = {"data":{
			"packedpid_A": pid_A[:counter_A].copy(),
			"packedpid_B": pid_B[:counter_B].copy()},
			"header": header
		}

	'''
	if isPrimary:
		data_tree = {"data":{
			"packedpid_A": particles["pid_A"][:counter_A].data.copy(),
			"packedpid_B": particles["pid_B"][:counter_B].data.copy(),
			"rvint_A": particles["rvint_A"][:counter_A].data.copy(), # Trim to last filled value
			"rvint_B": particles["rvint_B"][:counter_B].data.copy()},
			"header": header
		}
	else:
		data_tree = {"data":{
			"packedpid_A": particles["pid_A"][:counter_A].data.copy(),
			"packedpid_B": particles["pid_B"][:counter_B].data.copy()},
			"header": header
		}
	'''
	#outfile = asdf.AsdfFile(data_tree)
	#outfile.write_to(clean_dir + "/cleaned_rvpid_%03d.asdf"%ii)
	#outfile.close()
	asdf_fn = clean_dir + "/cleaned_rvpid_%03d.asdf"%ii
	with asdf.AsdfFile(data_tree) as af, CksumWriter(asdf_fn) as fp:
		af.write_to(fp, all_array_compression="blsc")

	#tend = time.time()
	#print("Writing took: %4.2fs"%(tend-tstart))

	# Free memory
	#N=[]; nstartA=[]; nstartB=[]; noutA=[]; noutB=[]; pidrvint_sub=[]; N_mainprog=[]
	#global_ind=[]; merged_to=[]
	particles=[]; data_tree=[]; pid_A=[]; pid_B=[]; rvint_A=[]; rvint_B=[]

	gc.collect()

print("Deleting temporary files...")

#tracemalloc.stop()


for f in glob.glob(merge_dir+"/*.npy"):
	os.remove(f)

print("Combining checksums...")
os.system("$ABACUS/external/fast-cksum/bin/merge_checksum_files.py --delete %s/*.crc32 > %s/checksums.crc32"%(merge_dir, merge_dir))
os.system("$ABACUS/external/fast-cksum/bin/merge_checksum_files.py --delete %s/*.crc32 > %s/checksums.crc32"%(clean_dir, clean_dir))


tfinish = time.time()
print("Took %4.2fs."%(tfinish-tbegin))
sys.stdout.flush()
