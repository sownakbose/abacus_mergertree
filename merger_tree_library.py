#!/usr/bin/env python
#! Filename: associate_particle_slices_v5.0.py

from __future__ import division
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from numba import jit
from tqdm import *
import numpy as np
import h5py as h5
import tempfile
import shutil
import numba
import glob
import time
import sys
import os

def read_halo_catalogue(step_now, halo_type, return_header = False):

	print("Reading halo catalogue from step %s"%(step_now))
	sys.stdout.flush()

	if "Abacus" in halo_type:
		if "FOF" in halo_type:
				import Groups
				cat    = Groups.GroupCatalog(step_now, load_subsamples="pids", convert_units=True)
				halo   = cat.halos
				pids   = cat.tagged_pids
				nstart = halo[:]["taggedstart"]
				ntag   = halo[:]["ntagged"]
				pos    = halo[:]["pos"]
				nphalo = halo[:]["N"]
		elif "SO" in halo_type:
				from abacus_halo_catalog import DietAbacusHaloCatalog
				cat    = Groups.DietGroupCatalog(step_now, load_subsamples="A_pids", convert_units=True)
				halo   = cat.halos
				pids   = cat.pids
				rho    = Groups.get_density(pids)
				pids   = Groups.get_id(pids)
				nstart = halo[:]["npstartA"]
				ntag   = halo[:]["npoutA"]
				pos    = halo[:]["x_L2com"]
				vmax   = halo[:]["vcirc_max_L2com"]
				nphalo = halo[:]["N"]
		elif "slabwise" in halo_type:
				cat      = CompaSOHaloCatalog(step_now, load_subsamples="AB_halo_pid", convert_units=True, fields = "merger", unpack_bits = "merger")
				pids     = cat.subsamples[:]["pid"].data
				rho      = cat.subsamples[:]["density"].data
				nstartA  = cat.halos[:]["npstartA"].data
				ntagA    = cat.halos[:]["npoutA"].data
				nstartB  = cat.halos[:]["npstartB"].data
				ntagB    = cat.halos[:]["npoutB"].data
				pos      = cat.halos[:]["x_L2com"].data
				vmax     = cat.halos[:]["vcirc_max_L2com"].data
				r100     = cat.halos[:]["r100_L2com"].data
				nphalo   = cat.halos[:]["N"].data
				numhalos = cat.numhalos
				ntag     = ntagA + ntagB
		elif "asdf" in halo_type:
				cat    = DietAbacusHaloCatalog(step_now, load_subsamples="B_halo_pid", convert_units=True, unpack_bits=True)
				pids   = cat.subsamples[:]["pid"].data
				rho    = cat.subsamples[:]["density"].data
				nstart = cat.halos[:]["npstartB"].data
				ntag   = cat.halos[:]["npoutB"].data
				pos    = cat.halos[:]["x_L2com"].data
				vmax   = cat.halos[:]["vcirc_max_L2com"].data
				nphalo = cat.halos[:]["N"].data
		elif "Cosmos" in halo_type:
				from AbacusCosmos import Halos
				cat    = Halos.make_catalog_from_dir(dirname = step_now, load_pids = True, load_subsamples=True)
				halo   = cat.halos
				parts  = cat.subsamples
				nstart = halo[:]["subsamp_start"]
				ntag   = halo[:]["subsamp_len"]
				pos    = halo[:]["pos"]
				pids   = parts["pid"]
				rho    = np.ones(len(pids))*1.
				vmax   = halo[:]["vcirc_max"]
				nphalo = halo[:]["N"]
		nslice = cat.header["FullStepNumber"]
		mpart  = cat.header["ParticleMassHMsun"]
		z      = cat.header["Redshift"]
		box    = cat.header["BoxSizeHMpc"]
		mhalo  = nphalo * mpart
	elif "Rockstar" in halo_type:
		import hdf5_multi_file as hmf
		nfiles = len(glob.glob(step_now + "/halos_0*.h5"))
		cat    = hmf.MultiFile(step_now + "/halos_0.%d.h5", (0, nfiles), allow_missing = True)
		halo   = cat["halos"]
		nstart = halo["subsamp_start"]
		ntag   = halo["subsamp_len"]
		pos    = halo["pos"]
		nphalo = halo["N"]
		mhalo  = halo["m"]
		vmax   = halo["vmax"]
		#parent = halo["parent_id"]
		ftmp   = h5.File(step_now + "/halos_0.0.h5", "r")
		nslice = ftmp["halos"].attrs["FullStepNumber"]
		z      = ftmp["halos"].attrs["Redshift"]
		box    = ftmp["halos"].attrs["BoxSizeHMpc"]
		ftmp.close(); cat.close()
		part_f = hmf.MultiFile(step_now + "/particles_0.%d.h5", (0, nfiles), allow_missing = True)
		parts  = part_f["particles"]
		pids   = parts["id"]
		rho    = np.ones(len(pids))*1.
		part_f.close()

	if return_header:
		if not "slabwise" in halo_type:
			return cat.header, box, nslice, z, nphalo, mhalo, pos, vmax, nstart, ntag, pids, rho
		else:
			return cat.header, box, nslice, z, numhalos, nphalo, mhalo, pos, r100, vmax, nstartA, ntagA, nstartB, ntagB, ntag, pids, rho
	else:
		if not "slabwise" in halo_type:
			return box, nslice, z, nphalo, mhalo, pos, vmax, nstart, ntag, pids, rho
		else:
			return box, nslice, z, numhalos, nphalo, mhalo, pos, r100, vmax, nstartA, ntagA, nstartB, ntagB, ntag, pids, rho

# It's probably worth giving these haloes a unique identifier

def indxxHalo(Nslice, Nhaloes):
	indxx  = Nslice * 1e12 + np.arange(Nhaloes)
	indxx  = indxx.astype(int)
	return indxx

@jit(nopython=True)
def indxxHaloSlabwise(Nslice, numhaloArray, filextArray):
	indxxArray = np.zeros(np.sum(numhaloArray), dtype = numba.int64)
	for nn in range(len(numhaloArray)):
		if nn == 0:
			startindex = 0
			endindex = startindex + numhaloArray[nn]
		else:
			startindex = np.sum(numhaloArray[:nn])
			endindex = startindex + numhaloArray[nn]
		indxxArray[startindex : endindex] = Nslice*1e12 + filextArray[nn]*1e9 + np.arange(numhaloArray[nn])
	return indxxArray

# Create a utility for resizing/appending to an existing HDF5 array

def writeDset(dataset, appendArray):
	oldLen = dataset.len()
	newLen = oldLen + len(appendArray)
	try:
		dataset.resize((newLen,appendArray.shape[1]))
	except IndexError:
		dataset.resize((newLen,))
	dataset[oldLen:newLen] = appendArray

# Function to associate particle IDs with a unique halo ID
def filled_array(i, start, end, length):
	out = np.zeros((length), dtype=int)
	np.add.at(out,start,i)
	np.add.at(out,end,-i)
	return out.cumsum()
