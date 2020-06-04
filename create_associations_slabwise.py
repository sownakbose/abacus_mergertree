#!/usr/bin/env python
#! Filename: create_associations_slabwise.py

from __future__ import division
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from numba import jit
from tqdm import *
import numpy as np
import h5py as h5
import tempfile
import warnings
import argparse
import shutil
import numba
import glob
import asdf
import time
import sys
import os

# Load merger tree libraries

from merger_tree_library import read_halo_catalogue, filled_array, indxxHaloSlabwise

warnings.filterwarnings("ignore")

# Create the parser

my_parser = argparse.ArgumentParser(description="Code to create halo time slice associations from abacus outputs",\
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Add the arguments

my_parser.add_argument("-inputdir",\
 	action="store",\
	type=str,\
	help="Path to where the simulation outputs are stored",\
	default="/mnt/store/AbacusSummit/")
my_parser.add_argument("-simname",\
 	action="store",\
	type=str,\
	help="Simulation name",\
	default="AbacusSummit_highbase_c000_ph100")
my_parser.add_argument("-numhalofiles",\
 	action="store",\
	type=int,\
	help="Number of halo_info files per output time",\
	default=16)
my_parser.add_argument("-num_cores",\
 	action="store",\
	type=int,\
	help="Number of cores for parallelisation",\
	default=16)
my_parser.add_argument("-outputdir",\
 	action="store",\
	type=str,\
	help="Path to where the associations outputs will be saved",\
	default="../data/")

args = my_parser.parse_args()

# Initialise

num_cores    = args.num_cores
pre_dispatch = "4*n_jobs"
batch_size   = "auto"
halo_type    = "Abacus_slabwise" # "Abacus_FOF" or "Abacus_SO" or "Rockstar" or "Abacus_Cosmos" or "asdf"
start_snap   = 0
end_snap     = None

# Slice spacing to build tree for
dn = 1
file_nchunks = args.numhalofiles

# The following parameters probably shouldn't be changed

# Match fraction to be considered a progenitor
mfrac    = 0.49 # should be = 0.5, but allow a small error
# Minimum number of particles in halo to consider for a candidate merger tree root
npmin    = 50
# Minimum number of subsampled particles a halo needs to have to consider for associations
ntagmin  = 5
# Minimum number of subsampled particles a halo needs to have to be considered a matching candidate
lowlim   = int(mfrac*ntagmin)
# Threshold halo mass for building tree of proximate massive haloes ( needed for search_rad parameter)
massive_threshold = 1e14

# Output name

simName = args.simname #"AbacusSummit_highbase_c000_ph100"

# Load initial catalogue

base   = args.inputdir + "/%s"%(simName)
base  += "/halos/z*"

odir   = args.outputdir + "/%s/"%(simName)

if not os.path.exists(odir):
	os.makedirs(odir)

if (halo_type == "Abacus_Cosmos") or (halo_type == "Abacus_asdf") or (halo_type == "Abacus_slabwise"):
	steps  = sorted(glob.glob(base))
else:
	steps  = sorted(glob.glob(base+"Step*"))

if (halo_type == "Abacus_SO") or (halo_type == "Abacus_FOF"):
		# Reverse list in descending order of slice Nr
		steps  = steps[::-1]
elif (halo_type == "Rockstar") or (halo_type == "Abacus_Cosmos") or (halo_type == "Abacus_asdf") or (halo_type == "Abacus_slabwise"):
		# List in descending order of redshift already
		steps  = steps

steps    = steps[:end_snap]
stepsAll = steps[::dn]
steps    = stepsAll[start_snap:]

# Routine for looping through candidate haloes and matching IDs

read_time       = 0.0
write_time      = 0.0
tree_build_time = 0.0
tree_query_time = 0.0
loop_time       = 0.0

@jit(nopython=True, fastmath=True)
def surf_halo(iter, neigh, mainProgArray, mainProgFracArray):

	halo_index   = mask_eligible[iter]

	progs = []

	if neigh[0] == -999:
		mainProgArray[halo_index]  = -999
		return [0]

	# Collect the PIDs and densities associated with the 10% subsamples for this halo
	ids_this_haloA = pids[nstartA[halo_index] : nstartA[halo_index]+ntagA[halo_index]]
	ids_this_haloB = pids[nstartB[halo_index] : nstartB[halo_index]+ntagB[halo_index]]
	ids_this_halo  = np.append(ids_this_haloA, ids_this_haloB)
	nump_this_halo = len(ids_this_halo)
	rho_this_haloA = rho[nstartA[halo_index] : nstartA[halo_index]+ntagA[halo_index]]
	rho_this_haloB = rho[nstartB[halo_index] : nstartB[halo_index]+ntagB[halo_index]]
	rho_this_halo  = np.append(rho_this_haloA, rho_this_haloB)
	rho_initial    = np.sum(rho_this_halo)

	ncontr_max   = 0.0 # weighted by density
	id_contr_max = -999
	frac_now     = 0.0

	# Loop over list of neighbours
	for	j in range(len(neigh)):

		idx_this_cand  = neigh[j]

		if (ntag_next[idx_this_cand] == 0) or (ntag_next[idx_this_cand] <= lowlim):
			continue

		# Collect PIDs associated with the 10% subsamples for this candidate match
		ids_tmp_candA  = pids_next[nstartA_next[idx_this_cand] : nstartA_next[idx_this_cand]+ntagA_next[idx_this_cand]]
		ids_tmp_candB  = pids_next[nstartB_next[idx_this_cand] : nstartB_next[idx_this_cand]+ntagB_next[idx_this_cand]]
		ids_tmp_cand   = np.append(ids_tmp_candA, ids_tmp_candB)
		ids_tmp_cand.sort()

		# Match IDs
		# Custom version of sorted match
		ptr             = np.searchsorted(ids_tmp_cand, ids_this_halo)
		ptr[ptr>=len(ids_tmp_cand)] = 0
		ptr[ptr<0]      = 0
		mask            = np.where(ids_tmp_cand[ptr] == ids_this_halo)[0]

		#print(len(mask), indxx_next[idx_this_cand], '%3.2e'%mhalo_next[idx_this_cand], mhalo[halo_index]/mhalo_next[idx_this_cand])

		# The contribution of this candidate is the number of matched particles weighted by density
		weighted_count = np.sum(rho_this_halo[mask])
		if not len(mask) == 0:
			pids_frac_this = len(mask) / float(nump_this_halo)
			pids_frac_cand = len(mask) / float(len(ids_tmp_cand))
			#print(len(mask), pids_frac_this, pids_frac_cand, weighted_count)
			if (pids_frac_cand >= mfrac) or (pids_frac_this >= mfrac):
				progs.append(indxx_next[idx_this_cand])
				if weighted_count > ncontr_max:
					ncontr_max   = weighted_count#len(mask)
					id_contr_max = indxx_next[idx_this_cand]
					frac_now     = pids_frac_cand

		# Remove matched particles
		ids_this_halo = np.delete(ids_this_halo, mask)
		rho_this_halo = np.delete(rho_this_halo, mask)

		#if (len(ids_this_halo) <= lowlim):
		#	break

	mainProgArray[halo_index]     = id_contr_max
	mainProgFracArray[halo_index] = frac_now

	return progs

@jit(nopython=True, fastmath=True)
def surf_halo_dnext(iter, neigh, dmainProgArray, dmainProgFracArray):

	halo_index = mask_eligible[iter]

	if neigh[0] == -999:
		dmainProgArray[halo_index]  = -999
		return

	# Collect the PIDs and densities associated with the 10% subsamples for this halo
	ids_this_haloA = pids[nstartA[halo_index] : nstartA[halo_index]+ntagA[halo_index]]
	ids_this_haloB = pids[nstartB[halo_index] : nstartB[halo_index]+ntagB[halo_index]]
	ids_this_halo  = np.append(ids_this_haloA, ids_this_haloB)
	nump_this_halo = len(ids_this_halo)
	rho_this_haloA = rho[nstartA[halo_index] : nstartA[halo_index]+ntagA[halo_index]]
	rho_this_haloB = rho[nstartB[halo_index] : nstartB[halo_index]+ntagB[halo_index]]
	rho_this_halo  = np.append(rho_this_haloA, rho_this_haloB)
	rho_initial    = np.sum(rho_this_halo)

	ncontr_max   = 0.0 # weighted by density
	id_contr_max = -999
	frac_now     = 0.0

	for	j in range(len(neigh)):

		idx_this_cand  = neigh[j]
		if (ntag_dnext[idx_this_cand] == 0) or (ntag_dnext[idx_this_cand] <= lowlim):
			continue

		ids_tmp_candA  = pids_dnext[nstartA_dnext[idx_this_cand] : nstartA_dnext[idx_this_cand]+ntagA_dnext[idx_this_cand]]
		ids_tmp_candB  = pids_dnext[nstartB_dnext[idx_this_cand] : nstartB_dnext[idx_this_cand]+ntagB_dnext[idx_this_cand]]
		ids_tmp_cand   = np.append(ids_tmp_candA, ids_tmp_candB)
		ids_tmp_cand.sort()

		# Match IDs
		# Custom version of sorted match
		ptr             = np.searchsorted(ids_tmp_cand, ids_this_halo)
		ptr[ptr>=len(ids_tmp_cand)] = 0
		ptr[ptr<0]      = 0
		mask            = np.where(ids_tmp_cand[ptr] == ids_this_halo)[0]

		# The contribution of this candidate is the number of matched particles weighted by density
		weighted_count = np.sum(rho_this_halo[mask])

		if not len(mask) == 0:
			pids_frac_this = len(mask) / float(nump_this_halo)
			pids_frac_cand = len(mask) / float(len(ids_tmp_cand))
			if (pids_frac_cand >= mfrac) or (pids_frac_this >= mfrac):
				if weighted_count > ncontr_max:
					ncontr_max   = weighted_count#len(mask)
					id_contr_max = indxx_dnext[idx_this_cand]
					frac_now     = pids_frac_cand

		# Remove matched particles
		ids_this_halo = np.delete(ids_this_halo, mask)
		rho_this_halo = np.delete(rho_this_halo, mask)

		# Since we're only interested in the main progenitor in the dnext step,
		# we need only look through a few candidates at most
		if (np.sum(rho_this_halo) < 0.5*rho_initial):
			break

	dmainProgArray[halo_index]     = id_contr_max
	dmainProgFracArray[halo_index] = frac_now
	return

# Master function that executes both surf_halo and surf_halo_dnext
def surf_halo_tot(iter, counter, mainProgArray, mainProgFracArray, dmainProgArray, dmainProgFracArray):

		# Define neigh and dneigh here!!
		neigh  = neighbours[counter]
		dneigh = dneighbours[counter]

		try:
			progenitors = surf_halo(iter, neigh, mainProgArray, mainProgFracArray)
		except ValueError:
			progenitors = surf_halo(iter, [-999], mainProgArray, mainProgFracArray)#[0]

		if len(progenitors)==0:
			progenitors = [0]

		try:
			surf_halo_dnext(iter, dneigh, dmainProgArray, dmainProgFracArray)
		except ValueError:
			surf_halo_dnext(iter, [-999], dmainProgArray, dmainProgFracArray)

		return progenitors

# Master function for final step of associations code when only surf_halo needs to be run
def surf_halo_final(iter, counter, mainProgArray, mainProgFracArray):

	neigh = neighbours[counter]

	try:
		progenitors = surf_halo(iter, neigh, mainProgArray, mainProgFracArray)
	except ValueError:
		progenitors = surf_halo(iter, [-999], mainProgArray, mainProgFracArray)#[0]

	if len(progenitors)==0:
		progenitors = [0]

	return progenitors


start_time = time.time()

# Being loop over timesteps
for jj in range(len(steps)-1):
#for jj in range(1):

	print("Step %d of %d"%(jj+1,len(steps)-1))
	sys.stdout.flush()
	step      = steps[jj]
	step_next = steps[jj+1]

	if jj == 0:
		is_first_step = True
	else:
		is_first_step = False

	if not jj == len(steps)-2:
		do_dnext = True
	else:
		do_dnext = False

	if do_dnext:
		step_dnext = steps[jj+2]

	# First, get the list of halo files in this output step
	step_list  = sorted(glob.glob(step + "/halo_info/halo_info*"))
	num_files  = len(step_list)
	chunk_list = np.array_split(step_list, file_nchunks)

	# Next, get the list of halo files in the next_output
	step_next_list = sorted(glob.glob(step_next + "/halo_info/halo_info*"))
	num_files_next = len(step_next_list)

	if do_dnext:
		# Get the list of halo files in the dnext_output
		step_dnext_list = sorted(glob.glob(step_dnext + "/halo_info/halo_info*"))
		num_files_dnext = len(step_dnext_list)

	t_step_start = time.time()

	for ifile_counter in range(file_nchunks):

		file_num_min = int(chunk_list[ifile_counter][0][-8:-5]) # Remove the .asdf trailing at the end
		file_num_max = int(chunk_list[ifile_counter][-1][-8:-5])

		# Begin reading halo_info files and 10% subsamples
		t_read_0 = time.time()
		if ifile_counter == 0:
			header, box, nslice, z, numhalos, nphalo, mhalo, pos, r100, vmax, nstartA, ntagA, nstartB, ntagB, ntag, pids, rho = read_halo_catalogue(chunk_list[ifile_counter], halo_type, return_header = True)
			z = header["Redshift"]
		else:
			box, nslice, z, numhalos, nphalo, mhalo, pos, r100, vmax, nstartA, ntagA, nstartB, ntagB, ntag, pids, rho = read_halo_catalogue(chunk_list[ifile_counter], halo_type, return_header = False)
		t_read_1 = time.time()

		read_time += (t_read_1 - t_read_0)

		file_ext_nums = np.zeros(len(numhalos), dtype = "int")
		for i in range(len(numhalos)):
			file_ext_nums[i] = int(chunk_list[ifile_counter][i][-8:-5])

		# Now, index these haloes
		indxx = indxxHaloSlabwise(nslice, numhalos, file_ext_nums)

		if jj == 0:
			half_box = 0.5*box

		search_list_next = []

		# Collect the halo_info filenames on either side of current in the preceding timestep
		if file_num_max == len(step_next_list)-1:
			search_list_next.append(step_next_list[file_num_min:file_num_max+1])
			search_list_next = list(np.array(search_list_next).flat)
			search_list_next.append(step_next_list[0])
		else:
			search_list_next.append(step_next_list[file_num_min:file_num_max+2])
			search_list_next = list(np.array(search_list_next).flat)
		search_list_next.insert(0, step_next_list[file_num_min-1])

		# Else, if we're given the entire list
		if (file_num_min == 0) and (file_num_max == len(step_next_list)-1):
			search_list_next = step_next_list

		# Read the halo_info files and 10% subsamples
		t_read_0 = time.time()
		box, nslice_next, z_next, numhalos_next, nphalo_next, mhalo_next, pos_next, r100_next, vmax_next, nstartA_next, ntagA_next, nstartB_next, ntagB_next, ntag_next, pids_next, rho_next = read_halo_catalogue(search_list_next, halo_type, return_header = False)
		t_read_1 = time.time()

		read_time += (t_read_1 - t_read_0)

		file_ext_nums = np.zeros(len(numhalos_next), dtype = "int")
		for i in range(len(numhalos_next)):
			file_ext_nums[i] = int(search_list_next[i][-8:-5])

		# Now, index these haloes
		indxx_next = indxxHaloSlabwise(nslice_next, numhalos_next, file_ext_nums)

		# Build a tree of halo positions
		print("Building tree 1 of 3.")
		sys.stdout.flush()
		t_build_1 = time.time()
		tree = cKDTree(pos_next+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)
		t_build_2 = time.time()
		tbuild    = t_build_2-t_build_1
		print("Tree build time: %4.2fs"%(tbuild))
		sys.stdout.flush()

		tree_build_time += tbuild

		if do_dnext:

			search_list_dnext = []

			# Collect the halo_info filenames on either side of current in the second preceding timestep
			if file_num_max == len(step_dnext_list)-1:
				search_list_dnext.append(step_dnext_list[file_num_min:file_num_max+1])
				search_list_dnext = list(np.array(search_list_dnext).flat)
				search_list_dnext.append(step_dnext_list[0])
			else:
				search_list_dnext.append(step_dnext_list[file_num_min:file_num_max+2])
				search_list_dnext = list(np.array(search_list_dnext).flat)
			search_list_dnext.insert(0, step_dnext_list[file_num_min-1])

			# Else, if we're given the entire list
			if (file_num_min == 0) and (file_num_max == len(step_dnext_list)-1):
				search_list_dnext = step_dnext_list

			# Read in the halo_info files
			t_read_0 = time.time()
			box, nslice_dnext, z_dnext, numhalos_dnext, nphalo_dnext, mhalo_dnext, pos_dnext, r100_dnext, vmax_dnext, nstartA_dnext, ntagA_dnext, nstartB_dnext, ntagB_dnext, ntag_dnext, pids_dnext, rho_dnext = read_halo_catalogue(search_list_dnext, halo_type, return_header = False)
			t_read_1 = time.time()

			read_time += (t_read_1 - t_read_0)

			file_ext_nums = np.zeros(len(numhalos_dnext), dtype = "int")
			for i in range(len(numhalos_dnext)):
				file_ext_nums[i] = int(search_list_dnext[i][-8:-5])

			# Now, index these haloes
			indxx_dnext = indxxHaloSlabwise(nslice_dnext, numhalos_dnext, file_ext_nums)

			# Build a tree of halo positions
			print("Building tree 2 of 3.")
			sys.stdout.flush()
			t_build_1 = time.time()
			tree_dnext = cKDTree(pos_dnext+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)
			t_build_2 = time.time()
			tbuild    = t_build_2-t_build_1
			print("Tree build time: %4.2fs"%(tbuild))
			sys.stdout.flush()

			tree_build_time += tbuild

		mask_eligible = np.where( (nphalo >= npmin) & (ntag >= ntagmin) )[0]

		# Find out which haloes in current timestep are close to a massive cluster. These objects
		# need an especially large search_rad
		print("Building tree 3 of 3.")
		sys.stdout.flush()
		t_build_1     = time.time()
		tree_massive  = cKDTree(pos[mhalo>=massive_threshold]+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)
		#tree_original = cKDTree(pos[mask_eligible]+half_box, boxsize=box+1e-6, compact_nodes = False, balanced_tree = False)
		t_build_2     = time.time()
		tbuild        = t_build_2-t_build_1
		print("Tree build time: %4.2fs"%(tbuild))
		sys.stdout.flush()

		tree_build_time += tbuild

		t_query_1  = time.time()
		dist, indx = tree_massive.query(pos[mask_eligible]+half_box, k = 1, distance_upper_bound = 3.5, n_jobs = -1)
		t_query_2  = time.time()
		tquery     = t_query_2-t_query_1
		print("Took %4.2fs to query all neighbours."%(tquery))
		sys.stdout.flush()

		objs_with_massive_neighbour = np.isfinite(dist)

		# Build the array of search_rad
		search_rad                 = r100[mask_eligible] * 4.0
		search_rad[search_rad>3.0] = 3.0
		search_rad[search_rad<2.0] = 2.0
		search_rad[objs_with_massive_neighbour] = r100[mhalo>=massive_threshold][indx[objs_with_massive_neighbour]] * 4.0

		if num_cores == 1:
			sort_index    = np.argsort(ntag[mask_eligible])[::-1]
			mask_eligible = mask_eligible[sort_index]

		# Now, we need to find the list of neighbours in the next_output step
		t_query_1  = time.time()
		#neighbours = tree_original.query_ball_tree(tree, r = search_rad, eps = 0.1)
		neighbours = tree.query_ball_point(pos[mask_eligible]+half_box, r = search_rad, return_sorted = True)
		t_query_2  = time.time()
		tquery     = t_query_2-t_query_1
		print("Took %4.2fs to query all neighbours."%(tquery))
		sys.stdout.flush()

		tree_query_time += tquery

		if do_dnext:
			print("Finding neighbours for subsequent catalogue.")
			sys.stdout.flush()
			t_query_1   = time.time()
			#dneighbours = tree_original.query_ball_tree(tree_dnext, r = 0.5*search_rad, eps = 0.1)
			dneighbours = tree_dnext.query_ball_point(pos[mask_eligible]+half_box, r = 2.0, return_sorted = True)
			t_query_2   = time.time()
			tquery      = t_query_2-t_query_1
			print("Took %4.2fs to query all neighbours."%(tquery))
			sys.stdout.flush()

			tree_query_time += tquery

		# Create some temporary memory spaces for multiprocessing
		folder      = tempfile.mkdtemp()
		mpc_name    = os.path.join(folder, "mpc")
		mpfc_name   = os.path.join(folder, "mpfc")
		desc_name   = os.path.join(folder, "desc")

		dmpc_name  = os.path.join(folder, "dmpc")
		dmpfc_name = os.path.join(folder, "dmpfc")

		MAIN_PROG     = np.memmap(mpc_name, "int", shape = len(mhalo), mode = "w+")
		MPMATCH_FRAC  = np.memmap(mpfc_name, "float", shape = len(mhalo), mode = "w+")

		DMAIN_PROG    = np.memmap(dmpc_name, "int", shape = len(mhalo), mode = "w+")
		DMPMATCH_FRAC = np.memmap(dmpfc_name, "float", shape = len(mhalo), mode = "w+")

		IS_ASSOC                = np.zeros(len(mhalo), dtype = "int")
		IS_ASSOC[mask_eligible] = 1

		t_loop_start  = time.time()

		'''
		t_sort_start = time.time()
		sort_next  = np.argsort(pids_next)
		sort_dnext = np.argsort(pids_dnext)
		pids_next_sorted  = pids_next[sort_next]
		pids_dnext_sorted = pids_dnext[sort_next]
		t_sort_end = time.time()
		print("PID sorting took: %4.2fs"%(t_sort_end-t_sort_start))
		'''

		if do_dnext:
			with Parallel(n_jobs = num_cores, batch_size = batch_size, pre_dispatch = pre_dispatch, backend = "multiprocessing") as parallel:
				PROG_INDX = parallel(delayed(surf_halo_tot)(i, counter, MAIN_PROG, MPMATCH_FRAC, DMAIN_PROG, DMPMATCH_FRAC) for i, counter in zip(range(len(mask_eligible)), trange(len(mask_eligible)))     )

		else:
			with Parallel(n_jobs = num_cores, batch_size = batch_size, pre_dispatch = pre_dispatch, backend = "multiprocessing") as parallel:
				PROG_INDX = parallel(delayed(surf_halo_final)(i, counter, MAIN_PROG, MPMATCH_FRAC) for i, counter in zip(range(len(mask_eligible)), trange(len(mask_eligible)))     )

		t_loop_finish = time.time()
		loop_time += (t_loop_finish-t_loop_start)

		t0 = time.time()

		# We need to flatten the list of progenitor indices
		PROG_INDX_OUT = [[0]] * len(mhalo)
		for i in range(len(mask_eligible)):
			ind = mask_eligible[i]
			PROG_INDX_OUT[ind] = PROG_INDX[i]

		NUM_PROG      = [len(item) for item in PROG_INDX_OUT]
		PROG_INDX_OUT = [item for sublist in PROG_INDX_OUT for item in sublist]
		PROG_INDX_OUT = np.array(PROG_INDX_OUT)
		NUM_PROG      = np.array(NUM_PROG)

		# Create data tree structure
		data_tree ={
		"data": {
		"HaloIndex": indxx,
		"HaloMass": mhalo,
		"HaloVmax": vmax,
		"Position": pos,
		"IsAssociated": IS_ASSOC,
		"Progenitors": PROG_INDX_OUT,
		"NumProgenitors": NUM_PROG,
		"MainProgenitor": MAIN_PROG,
		"MainProgenitorFrac": MPMATCH_FRAC,
		"MainProgenitorPrec": DMAIN_PROG,
		"MainProgenitorPrecFrac": DMPMATCH_FRAC},
		"header": header,
		}

		# Save the data
		output_file = asdf.AsdfFile(data_tree)
		output_file.write_to(odir + "associations_z%3.2f.%d.asdf"%(z, ifile_counter))
		del PROG_INDX, PROG_INDX_OUT, NUM_PROG, MAIN_PROG, DMAIN_PROG, MPMATCH_FRAC, DMPMATCH_FRAC, IS_ASSOC

		t1 = time.time()
		print("Total write time: %4.2fs"%(t1-t0))

		write_time += (t1-t0)

		# Remove temporary folder
		shutil.rmtree(folder)

	t_step_finish = time.time()
	print("Total step time: %4.2fs"%(t_step_finish-t_step_start))
	print("Total time spent building trees until now: %4.2fs"%(tree_build_time))
    print("Total time spent querying trees until now: %4.2fs"%(tree_query_time))
	print("Total time spent reading until now: %4.2fs"%(read_time))
    print("Total time spent writing until now: %4.2fs"%(write_time))
	print("Total time spent matching haloes until now: %4.2fs"%(loop_time))

	sys.stdout.flush()

t_complete = time.time()
t_total    = t_complete-start_time

print("===============================================")
print("Merger tree associations complete. Final stats:")
print("The entire calculation took: %4.2fs"%(t_total))
print("Total reading time: %4.2fs"%(read_time))
print("Total writing time: %4.2fs"%(write_time))
print("Total tree building time: %4.2fs"%(tree_build_time))
print("Total tree query time: %4.2fs"%(tree_query_time))
print("Total halo matching loop time: %4.2fs"%(loop_time))
print("===============================================")
sys.stdout.flush()