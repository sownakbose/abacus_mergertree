"""
The ``compaso_halo_catalog`` module loads halo catalogs from CompaSO, Abacus's
on-the-fly halo finder.  The module defines one class, ``CompaSOHaloCatalog``,
whose constructor takes the path to a halo catalog as an argument.
Users should use this class as the primary interface to load
and manipulate halo catalogs.

The halo catalogs and particle subsamples are stored on disk in
ASDF files and are loaded into memory as Astropy tables.  Each
column of an Astropy table is essentially a Numpy array and can
be accessed with familiar Numpy-like syntax.  More on Astropy
tables here: http://docs.astropy.org/en/stable/table/

Beyond just loading the halo catalog files into memory, this
module performs a few other manipulations.  Many of the halo
catalog columns are stored in bit-packed formats (e.g.
floats are scaled to a ratio from 0 to 1 then packed in 16-bit
ints), so these columns are unpacked as they are loaded.

Furthermore, the halo catalogs for big simulations are divided
across a few dozen files.  These files are transparently loaded
into one monolithic Astropy table if one passes a directory
to ``CompaSOHaloCatalog``; to save memory by loading only one file,
pass just that file as the argument to ``CompaSOHaloCatalog``.

Importantly, because ASDF and Astropy tables are both column-
oriented, it can be much faster to load only the subset of
halo catalog columns that one needs, rather than all 60-odd
columns.  Use the ``fields`` argument to the ``CompaSOHaloCatalog``
constructor to specify a subset of fields to load.  Similarly, the
particles can be quite large, and one can use the ``load_subsamples``
argument to restrict the particles to the subset one needs.

Some brief examples and technical details about the halo catalog
layout are presented below, followed by the full module API.
Examples of using this module to work with AbacusSummit data can
be found on the AbacusSummit website here:
https://abacussummit.readthedocs.io


Short Example
=============
>>> from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
>>> # Load the RVs and PIDs for particle subsample A
>>> cat = CompaSOHaloCatalog('/storage/AbacusSummit/AbacusSummit_base_c000_ph000/halos/z0.100', load_subsamples='A_all')
>>> print(cat.halos[:5])  # cat.halos is an Astropy Table, print the first 5 rows
   id    npstartA npstartB ... sigmavrad_L2com sigmavtan_L2com rvcirc_max_L2com
-------- -------- -------- ... --------------- --------------- ----------------
25000000        0        2 ...       0.9473971      0.96568024      0.042019103
25000001       11       12 ...      0.86480814       0.8435805      0.046611086
48000000       18       15 ...      0.66734606      0.68342227      0.033434115
58000000       31       18 ...      0.52170926       0.5387341      0.042292822
58001000       38       23 ...       0.4689916      0.40759262      0.034498636
>>> print(cat.halos['N','x_com'][:5])  # print the position and mass of the first 5 halos
  N         x_com [3]
--- ------------------------
278 -998.88525 .. -972.95404
 45  -998.9751 .. -972.88416
101   -999.7485 .. -947.8377
 82    -998.904 .. -937.6313
 43   -999.3252 .. -937.5813
>>> # Examine the particle subsamples associated with the 5th halo
>>> h5 = cat.halos[4]
>>> print(cat.subsamples['pos'][h5['npstartA']:h5['npstartA'] + h5['npoutA']])
        pos [3]
------------------------
  -999.3019 .. -937.5229
 -999.33435 .. -937.5515
-999.38965 .. -937.58777
>>> # At a glance, the pos fields match that of the 5th halo above, so it appears we have indexed correctly!


Catalog Structure
=================
The catalogs are stored in a directory structure that looks like:

.. code-block:: none

    - SimulationName/
        - halos/
            - z0.100/
                - halo_info/
                    halo_info_000.asdf
                    halo_info_001.asdf
                    ...
                - halo_rv_A/
                    halo_rv_A_000.asdf
                    halo_rv_A_001.asdf
                    ...
                - <field & halo, rv & PID, subsample A & B directories>
            - <other redshift directories, some with particle subsamples, others without>

The file numbering roughly corresponds to a planar chunk of the simulation
(all y and z for some range of x).  The matching of the halo_info file numbering
to the particle file numbering is important; halo_info files index into the
corresponding particle files.

The halo catalogs are stored on disk in ASDF files (https://asdf.readthedocs.io/).
The ASDF files start with a human-readable header that describes
the data present in the file and metadata about the simulation
from which it came (redshift, cosmology, etc).  The rest of
the file is binary blobs, each representing a column (i.e.
a halo property).

Internally, the ASDF binary portions are usually compressed.  This should
be transparent to users, although you may be prompted to install the
blosc package if it is not present.  Decompression should be fast,
at least 500 MB/s per core.


Particle Subsamples
===================
We define two disjoint sets of "subsample" particles, called "subsample A" and
"subsample B".  Subsample A is a few percent of all particles, with subsample B
a few times larger than that.  Particles membership in each group is a function
of PID and is thus consistent across redshift.

At most redshifts, we only output halo catalogs and halo subsample particle PIDs.
This aids with construction of merger trees.  At a few redshifts, we provide
subsample particle positions as well as PIDs, for both halo particles and
non-halo particles, called "field" particles.


Halo File Types
===============
Each file type (for halos, particles, etc) is grouped into a subdirectory.
These subdirectories are:

- ``halo_info/``
    The primary halo catalog files.  Contains stats like
    CoM positions and velocities and moments of the particles.
    Also indicates the index and count of subsampled particles in the
    ``halo_pid_A/B`` and ``halo_rv_A/B`` files.

- ``halo_pid_A/`` and ``halo_pid_B/``
    The 64-bit particle IDs of particle subsamples A and B.  The PIDs
    contain information about the Lagrangian position of the particles,
    whether they are tagged, and their local density.

The following subdirectories are only present for the redshifts for which
we output particle subsamples and not just halo catalogs:

- ``halo_rv_A/`` and ``halo_rv_B/``
    The positions and velocities of the halo subsample particles, in "RVint"
    format. The halo associations are recoverable with the indices in the
    ``halo_info`` files.

- ``field_rv_A/`` and ``field_rv_B/``
    Same as ``halo_rv_<A|B>/``, but only for the field (non-halo) particles.

- ``field_pid_A/`` and ``field_pid_B/``
    Same as ``halo_pid_<A|B>/``, but only for the field (non-halo) particles.


Bit-packed Formats
==================
The "RVint" format packs six fields (x,y,z, and vx,vy,vz) into three ints (12 bytes).
Positions are stored to 20 bits (global), and velocities 12 bits (max 6000 km/s).

The PIDs are 8 bytes and encode a local density estimate, tag bits for merger trees,
and a unique particle id, the last of which encodes the Lagrangian particle coordinate.

These are described in more detail on the :doc:`AbacusSummit Data Model page <summit:data-products>`.

Field Subset Loading
====================
Because the ASDF files are column-oriented, it is possible to load just one or a few
columns (halo catalog fields) rather than the whole file.  This can save huge amounts
of IO, memory, and CPU time (due to the decompression).  Use the `fields` argument
to the ``CompaSOHaloCatalog`` constructor to specify the list of columns you want.

In detail, some columns are stored as ratios to other columns.  For example, ``r90``
is stored as a ratio relative to ``r100``.  So to properly unpack
``r90``, the ``r100`` column must also be read.  ``CompaSOHaloCatalog`` knows about
these dependencies and will load the minimum set necessary to return the requested
columns to the user.  However, this may result in more IO than expected.  The ``verbose``
constructor flag or the ``dependency_info`` field of the ``CompaSOHaloCatalog``
object may be useful for diagnosing exactly what data is being loaded.

Despite the potential extra IO and CPU time, the extra memory usage is granular
at the level of individual files.  In other words, when loading multiple files,
the concatenated array will never be constructed for columns that only exist for
dependency purposes.

Multi-threaded Decompression
============================
The Blosc compression we use inside the ASDF files supports multi-threaded
decompression.  We have packed AbacusSummit files with 4 Blosc blocks (each ~few MB)
per ASDF block, so 4 Blosc threads is probably the optimal value.  This is the
default value, unless fewer cores are available (as determined by the process
affinity mask).

.. note::

    Loading a CompaSOHaloCatalog will use 4 decompression threads by default.

You can control the number of decompression threads with:

.. code-block:: python

    import asdf.compression
    asdf.compression.set_decompression_options(nthreads=N)
"""

from glob import glob
import os
import os.path
from os.path import join as pjoin, dirname, basename, isdir, isfile, normpath, abspath, samefile
import re
import gc

# SOWNAK
from collections import defaultdict

# Stop astropy from trying to download time data; nodes on some clusters are not allowed to access the internet directly
from astropy.utils import iers
iers.conf.auto_download = False

import numpy as np
import numba as nb
import astropy.table
from astropy.table import Table
import asdf

import asdf.compression
try:
    asdf.compression.validate('blsc')
except:
    # Note: this is a temporary solution until blosc is integrated into ASDF, or until we package a pluggable decompressor
    exit('Error: your ASDF installation does not support Blosc compression.  Please install the fork with Blosc support with the following command: "pip install git+https://github.com/lgarrison/asdf.git"')

from bitpacked import unpack_rvint, unpack_pids, AUXPID

# Default to 4 decompression threads, or fewer if fewer cores are available
DEFAULT_BLOSC_THREADS = 4
DEFAULT_BLOSC_THREADS = max(1, min(len(os.sched_getaffinity(0)), DEFAULT_BLOSC_THREADS))
# asdf.compression.set_decompression_options(nthreads=DEFAULT_BLOSC_THREADS)
import abacusnbody.data.asdf
abacusnbody.data.asdf.set_nthreads(DEFAULT_BLOSC_THREADS)

class CompaSOHaloCatalog:
    """
    A halo catalog from Abacus's on-the-fly group finder.
    """

    # TODO: maybe this should just be a halotools catalog
    # TODO: optional progress meter for loading files
    # TODO: generator mode over chunks

    def __init__(self, path, clean_path, cleaned_halos=True, load_subsamples=False, convert_units=True, unpack_bits=False, fields='all', cleaned_fields='all', verbose=False):
        """
        Loads halos.  The ``halos`` field of this object will contain
        the halo records; and the ``subsamples`` field will contain
        the corresponding halo/field subsample positions and velocities and their
        ids (if requested via ``load_subsamples``).  The ``header`` field contains
        metadata about the simulation.

        Whether a particle is tagged or not is returned when loading the
        halo and field pids, as it is encoded for each in the 64-bit PID.
        The local density of the particle is also encoded in the PIDs
        and returned upon loading those.

        Parameters
        ----------
        path: str or list of str
            The halo catalog directory, like ``MySimulation/halos/z1.000/``.
            Or a single halo info file, or a list of halo info files.
            Will accept ``halo_info`` dirs or "redshift" dirs
            (e.g. ``z1.000/halo_info/`` or ``z1.000/``).

        load_subsamples: bool or str, optional
            Load halo particle subsamples.  True or False may be specified
            to load all particles or none, or a string in the following format
            may be specified:

            .. code-block:: none

                <A|B|AB>_[halo_|field_]<pid|rv|all>

            where fields in ``<>`` are mandatory and fields in ``[]`` are optional.
            So ``A_halo_rv`` would load the halo RVs from the A subsample and ``AB_pid``
            loads the halo and field PIDs from both subsamples.
            True is equivalent to ``AB_all``.
            False (the default) loads nothing.

        convert_units: bool, optional
            Convert positions from unit-box units to BoxSize-box units,
            velocities already come in km/s.  Default: True.

            #TODO: above velocity units right?

        unpack_bits: bool, optional
            Extract information from the PID field of each subsample particle
            info about its Lagrangian position, whether it is tagged,  and its
            current local density.  If False, only the particle ID part will
            be extracted.
            Default: False.

        fields: str or list of str, optional
            A list of field names/halo properties to load.  Selecting a small
            subset of fields can be substantially faster than loading all fields
            because the file IO will be limited to the desired fields.
            See ``compaso_halo_catalog.user_dt`` or the :doc:`AbacusSummit Data Model page <summit:data-products>`
            for a list of available fields.
            Default: 'all'

        verbose: bool, optional
            Print informational messages. Default: False

        """

        if type(path) is str:
            path = [path]  # dir or file
        else:
            # if list, must be all files
            for p in path:
                if os.path.exists(p) and not isfile(p):
                    raise ValueError(f'If passing a list of paths, all paths must be files, not dirs. Path "{p}" is not a file.')

        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f'Path "{p}" does not exist!')

        path = [abspath(p) for p in path]

        # SOWNAK
        if cleaned_halos:
            if type(clean_path) is str:
                clean_path = [clean_path]
            else:
                # if list, must be all files
                for p in clean_path:
                    if os.path.exists(p) and not isfile(p):
                        raise ValueError(f'If passing a list of paths, all paths must be files, not dirs. Path "{p}" is not a file.')
            for p in clean_path:
                if not os.path.exists(p):
                    raise FileNotFoundError(f'Path "{p}" does not exist!')

            clean_path = [abspath(p) for p in clean_path]

        # Allow users to pass halo_info dirs, even though redshift dirs remain canoncial
        for i,p in enumerate(path):
            if basename(p) == 'halo_info':
                path[i] = abspath(pjoin(p,os.pardir))

        # SOWNAK
        if cleaned_halos:
            for i,p in enumerate(clean_path):
                if basename(p) == 'cleaned_halo_info':
                    clean_path[i] = abspath(pjoin(p,os.pardir))

        # Can't mix files from different catalogs!
        if isfile(path[0]):
            self.groupdir = dirname(dirname(path[0]))
            for p in path:
                if not samefile(self.groupdir, dirname(dirname(p))):
                    raise ValueError("Can't mix files from different catalogs!")
                halo_fns = path  # path is list of one or more files

            for i,p in enumerate(path):
                for j,q in enumerate(path[i+1:]):
                    if samefile(p,q):
                        raise ValueError(f'Cannot pass duplicate halo_info files! Found duplicate "{p}" and at indices {i} and {i+j}')

        else:
            self.groupdir = path[0]  # path is a singlet of one dir
            globpat = pjoin(self.groupdir, 'halo_info', 'halo_info_*')
            halo_fns = sorted(glob(globpat))
            if len(halo_fns) == 0:
                raise FileNotFoundError(f'No halo_info files found! Search pattern was: "{globpat}"')

        del path  # use groupdir and halo_fns

        # SOWNAK
        if cleaned_halos:
            if isfile(clean_path[0]):
                self.cleandir = dirname(dirname(clean_path[0]))
                for p in clean_path:
                    if not samefile(self.cleandir, dirname(dirname(p))):
                        raise ValueError("Can't mix files from different catalogs!")
                    cleaned_halo_fns = clean_path

                for i,p in enumerate(clean_path):
                    for j,q in enumerate(clean_path[i+1:]):
                        if samefile(p,q):
                            raise ValueError(f'Cannot pass duplicate halo_info files! Found duplicate "{p}" and at indices {i} and {i+j}')

            else:
                self.cleandir = clean_path[0] # cleaned path is a singlet of one dir
                globpat = pjoin(self.cleandir, 'cleaned_halo_info', 'cleaned_halo_info_*')
                cleaned_halo_fns = sorted(glob(globpat))
                if len(cleaned_halo_fns) == 0:
                    raise FileNotFoundError(f'No halo_info files found! Search pattern was: "{globpat}"')
                del clean_path

        self.chunk_inds = np.array([int(hfn.split('_')[-1].strip('.asdf')) for hfn in halo_fns])
        self.data_key = 'data'
        self.convert_units = convert_units  # let's save, user might want to check later
        self.verbose = verbose

        if load_subsamples == False:
            # stub
            self.load_AB = []
            self.load_halofield = []
            self.load_pidrv = []
        else:
            # If user has not specified which subsamples, then assume user wants to load everything
            if load_subsamples == True:
                load_subsamples = "AB_all"
            if type(load_subsamples) != str:
                raise ValueError("`load_subsamples` argument must be string or bool")

            # Validate the user's `load_subsamples` option and figure out what subsamples we need to load
            # SOWNAK re.match --> re.fullmatch
            subsamp_match = re.fullmatch(r'(?P<AB>(A|B|AB))(_(?P<hf>halo|field))?_(?P<pidrv>all|pid|rv|pidrvint)', load_subsamples)
            if not subsamp_match:
                raise ValueError(f'Value "{load_subsamples}" for argument `load_subsamples` not understood')
            self.load_AB = subsamp_match.group('AB')
            self.load_halofield = subsamp_match.group('hf')
            self.load_halofield = [self.load_halofield] if self.load_halofield else ['halo','field']  # default is both
            self.load_pidrv = subsamp_match.group('pidrv')
            if self.load_pidrv == 'all':
                self.load_pidrv = ['pid','rv']
            # SOWNAK
            if self.load_pidrv == 'pidrvint':
                self.load_pidrv = ['pid', 'rvint']
        del load_subsamples  # use the parsed values


        # Open the first file, just to grab the header
        with asdf.open(halo_fns[0], lazy_load=True, copy_arrays=False) as af:
            # will also be available as self.halos.meta
            self.header = af['header']

        # Read and unpack the catalog into self.halos
        self._setup_halo_field_loaders()
        N_halo_per_file = self._read_halo_info(halo_fns, fields)
        if cleaned_halos:
            cleaned_N_halo_per_file = self._read_halo_info(cleaned_halo_fns, cleaned_fields, cleaned_halos=True)

            if (N_halo_per_file != cleaned_N_halo_per_file).any():
                raise RuntimeError('N_halo per superslab in primary halo files does not match N_halo per superslab in the cleaned files!')

        # TODO: the changes to the subsample loading procedure for the cleaned cats depends on the data model on disk we end up with
        self.subsamples = Table()  # empty table, to be filled with PIDs and RVs in the loading functions below

        # SOWNAK
        self.numhalos = N_halo_per_file

        # reindex subsamples if this is an L1 redshift
        # halo subsamples have not yet been reindexed
        self._reindexed = {'A': False, 'B': False}

        # Loading the particle information
        if "pid" in self.load_pidrv:
            self._load_pids(unpack_bits, N_halo_per_file)
        if "rv" in self.load_pidrv:
            self._load_RVs(N_halo_per_file)
        # SOWNAK
        if "rvint" in self.load_pidrv:
            self._load_RVs(N_halo_per_file, unpack=False)

    # SOWNAK
    def _read_halo_info(self, halo_fns, fields, cleaned_halos=False):
        # Remember, this function may be called twice
        # Once with the primary fields, then again with the cleaned fields

        # Open all the files, validate them, and count the halos
        # Lazy load, but don't use mmap
        afs = [asdf.open(hfn, lazy_load=True, copy_arrays=True) for hfn in halo_fns]

        # TODO: can't remember if keys() is an iterator
        # SOWNAK -- added list() to make it scriptable
        N_halo_per_file = np.array([len(af[self.data_key][list(af[self.data_key].keys())[0]]) for af in afs])
        N_halos = N_halo_per_file.sum()

        # TODO: maybe this logic should be in the calling function so we can differentiate between base and cleaned
        # SOWNAK
        if (fields == 'all') and not cleaned_halos:
            fields = list(user_dt.names)
        elif (fields == 'all') and cleaned_halos:
            fields = list(clean_dt.names)
        if type(fields) == str:
            fields = [fields]

        # Figure out what raw columns we need to read based on the fields the user requested
        # TODO: provide option to drop un-requested columns
        raw_dependencies, fields_with_deps, extra_fields = self._get_halo_fields_dependencies(fields)
        # save for informational purposes
        if not hasattr(self, 'dependency_info'):
            self.dependency_info = defaultdict(list)
        self.dependency_info['raw_dependencies'] += raw_dependencies
        self.dependency_info['fields_with_deps'] += fields_with_deps
        self.dependency_info['extra_fields'] += extra_fields

        if self.verbose:
            # TODO: going to be repeated in output
            print(f'{len(fields)} halo catalog fields requested. '
                f'Reading {len(raw_dependencies)} fields from disk. '
                f'Computing {len(extra_fields)} intermediate fields.')

        # Make an empty table for the concatenated, unpacked values
        # Note that np.empty is being smart here and creating 2D arrays when the dtype is a vector
        # SOWNAK
        if not cleaned_halos:
            cols = {col:np.empty(N_halos, dtype=user_dt[col]) for col in fields}
        else:
            cols = {col:np.empty(N_halos, dtype=clean_dt[col]) for col in fields}
        #cols = {col:np.full(N_halos, np.nan, dtype=user_dt[col]) for col in fields}  # nans for debugging
        if hasattr(self, 'halos'):
            # already exists
            # will throw error if duplicating a column
            self.halos.add_columns(list(cols.values()), names=list(cols.keys()), copy=False)
        else:
            # first time
            self.halos = Table(cols, copy=False)
            self.halos.meta.update(self.header)

        # Unpack the cats into the concatenated array
        # The writes would probably be more efficient if the outer loop was over column
        # and the inner was over cats, but wow that would be ugly
        N_written = 0
        for af in afs:
            # This is where the IO on the raw columns happens
            # There are some fields that we'd prefer to directly read into the concatenated table,
            # but ASDF doesn't presently support that, so this is the best we can do
            rawhalos = Table(data={field:af[self.data_key][field] for field in raw_dependencies}, copy=False)
            af.close()

            # `halos` will be a "pointer" to the next open space in the master table
            halos = self.halos[N_written:N_written+len(rawhalos)]
            N_written += len(rawhalos)  # actually not written yet, but let's aggregate the logic

            # For temporary (extra) columns, only need to construct the per-file version
            halos.add_columns([np.empty(len(rawhalos), dtype=user_dt[col]) for col in extra_fields], names=extra_fields, copy=False)
            #halos.add_columns([np.full(len(rawhalos), np.nan, dtype=user_dt[col]) for col in extra_fields], names=extra_fields, copy=False)

            loaded_fields = []
            for field in fields_with_deps:
                if field in loaded_fields:
                    continue
                loaded_fields += self._load_halo_field(halos, rawhalos, field)

            del rawhalos
            del halos
            gc.collect()

        return N_halo_per_file


    def _setup_halo_field_loaders(self):
        # Loaders is a dict of regex -> lambda
        # The lambda is responsible for unpacking the rawhalos field
        # The first regex that matches will be used, so they must be precise
        self.halo_field_loaders = {}

        if self.convert_units:
            box = self.header['BoxSize']
            # TODO: correct velocity units? There is an earlier comment claiming that velocities are already in km/s
            zspace_to_kms = self.header['VelZSpace_to_kms']
        else:
            box = 1.
            zspace_to_kms = 1.

        # The first argument to the following lambdas is the match object from re.match()
        # We will use m[0] to access the full match (i.e. the full field name)
        # Other indices, like m['com'], will access the sub-match with that group name

        # r10,r25,r33,r50,r67,r75,r90,r95,r98
        pat = re.compile(r'(?:r\d{1,2}|rvcirc_max)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']*raw['r100'+m['com']]/INT16SCALE*box

        # sigmavMin, sigmavMaj, sigmavrad, sigmavtan
        pat = re.compile(r'(?P<stem>sigmav(?:Min|Maj|rad|tan))(?P<com>_(?:L2)?com)')
        def _sigmav_loader(m,raw,halos):
            stem = m['stem'].replace('Maj','Max')
            return raw[stem+'_to_sigmav3d'+m['com']+'_i16']*raw['sigmav3d'+m['com']]/INT16SCALE*box
        self.halo_field_loaders[pat] = _sigmav_loader

        # sigmavMid
        pat = re.compile(r'sigmavMid(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: np.sqrt(raw['sigmav3d'+m['com']]*raw['sigmav3d'+m['com']]*box**2 \
                                                            - halos['sigmavMaj'+m['com']]**2 - halos['sigmavMin'+m['com']]**2)

        # sigmar
        pat = re.compile(r'sigmar(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']*raw['r100'+m['com']].reshape(-1,1)/INT16SCALE*box

        # sigman
        pat = re.compile(r'sigman(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]+'_i16']/INT16SCALE*box

        # x,r100 (box-scaled fields)
        pat = re.compile(r'(x|r100)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*box

        # v,sigmav,sigmav3d,meanSpeed,sigmav3d_r50,meanSpeed_r50,vcirc_max (vel-scaled fields)
        pat = re.compile(r'(v|sigmav3d|meanSpeed|sigmav3d_r50|meanSpeed_r50|vcirc_max)(?P<com>_(?:L2)?com)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*zspace_to_kms

        # id,npstartA,npstartB,npoutA,npoutB,ntaggedA,ntaggedB,N,L2_N,L0_N (raw/passthrough fields)
        # If ASDF could read into a user-provided array, could avoid these copies
        # SOWNAK
        pat = re.compile(r'id|npstartA|npstartB|npoutA|npoutB|ntaggedA|ntaggedB|N|L2_N|L0_N|N_total|N_merge|npstartA_merge|npstartB_merge|npoutA_merge|npoutB_merge|is_merged_to')  # TODO: can add other cleaned fields here
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]

        # SO_central_particle,SO_radius (and _L2max) (box-scaled fields)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_particle|_radius)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]*box

        # SO_central_density (and _L2max)
        pat = re.compile(r'SO(?:_L2max)?(?:_central_density)')
        self.halo_field_loaders[pat] = lambda m,raw,halos: raw[m[0]]

        # eigvecs loader
        pat = re.compile(r'(?P<rnv>sigma(?:r|n|v)_eigenvecs)(?P<which>Min|Mid|Maj)(?P<com>_(?:L2)?com)')
        def eigvecs_loader(m,raw,halos):
            minor,middle,major = unpack_euler16(raw[m['rnv']+m['com']+'_u16'])
            columns = {}

            minor_field = m['rnv'] + 'Min' + m['com']
            if minor_field in halos.colnames:
                columns[minor_field] = minor
            middle_field = m['rnv'] + 'Mid' + m['com']
            if middle_field in halos.colnames:
                columns[middle_field] = middle
            major_field = m['rnv'] + 'Maj' + m['com']
            if major_field in halos.colnames:
                columns[major_field] = major

            return columns

        self.halo_field_loaders[pat] = eigvecs_loader


    def _get_halo_fields_dependencies(self, fields):
        # Each of the loaders accesses some fields of the raw catalog
        # We can do automatic dependency generation by stubbing the raw catalog and recording accesses

        # TODO: define pre-set subsets of common fields

        class DepCapture:
            def __init__(self):
                self.keys = []
                self.colnames = []
            def __getitem__(self,key):
                self.keys += [key]
                return np.ones(1)  # a safe numeric value

        iter_fields = list(fields)  # make a copy

        raw_dependencies = []
        field_dependencies = []
        for field in iter_fields:
            have_match = False
            for pat in self.halo_field_loaders:
                # SOWNAK re.match --> re.fullmatch
                match = pat.fullmatch(field)
                if match:
                    if have_match:
                        raise KeyError(f'Found more than one way to load field "{field}"')
                    capturer,raw_capturer = DepCapture(),DepCapture()
                    self.halo_field_loaders[pat](match,raw_capturer,capturer)
                    raw_dependencies += raw_capturer.keys

                    # these are fields of `halos`
                    for k in capturer.keys:
                        # Add fields regardless of whether they have already been encountered
                        iter_fields += [k]
                        if k not in fields:
                            field_dependencies += [k]
                    have_match = True
                    #break  # comment out for debugging
            else:
                if not have_match:
                    raise KeyError(f"Don't know how to load halo field \"{field}\"")

        raw_dependencies = list(set(raw_dependencies))  # make unique
        # unique, preserve order, but using last occurrence
        # because nested dependencies need to be loaded in reverse order
        fields_with_deps = list(dict.fromkeys(iter_fields[::-1]))
        field_deps = list(dict.fromkeys(field_dependencies[::-1]))

        # All raw dependencies for all user-requested fields
        return raw_dependencies, fields_with_deps, field_deps


    def _load_halo_field(self, halos, rawhalos, field):
        # TODO: attach units to all these?

        # We must use the halos['field'][:] syntax in order to do an in-place update
        # We will enable column replacement warnings to make sure we don't make a mistake
        # Remember that "halos" here is a view into the self.halos table
        _oldwarn = astropy.table.conf.replace_warnings
        astropy.table.conf.replace_warnings = ['always']

        # Look for the loader for this field, should only match one
        have_match = False
        loaded_fields = []
        for pat in self.halo_field_loaders:
            # SOWNAK re.match --> re.fullmatch
            match = pat.fullmatch(field)
            if match:
                if have_match:
                    raise KeyError(f'Found more than one way to load field "{field}"')
                column = self.halo_field_loaders[pat](match,rawhalos,halos)

                # The loader is allowed to return a dict if it incidentally loaded multiple columns
                if type(column) == dict:
                    assert field in column
                    for k in column:
                        halos[k][:] = column[k]
                    loaded_fields += list(column)
                else:
                    halos[field][:] = column
                    loaded_fields += [field]

                have_match = True
                #break  # comment out for debugging
        else:
            if not have_match:
                raise KeyError(f"Don't know how to load halo field \"{field}\"")

        astropy.table.conf.replace_warnings = _oldwarn

        return loaded_fields


    def _reindex_subsamples(self, RVorPID, N_halo_per_file):
        if RVorPID == 'pid':
            asdf_col_name = 'packedpid'
        elif RVorPID == 'rv':
            asdf_col_name = 'rvint'
        else:
            raise ValueError(RVorPID)

        particle_AB_afs = []  # ASDF file handles for A+B
        np_total = 0
        np_per_file = []
        for AB in self.load_AB:
            # Open the ASDF file handles so we can query the size
            if 'halo' in self.load_halofield:
                halo_particle_afs = [asdf.open(pjoin(self.groupdir, f'halo_{RVorPID}_{AB}', f'halo_{RVorPID}_{AB}_{i:03d}.asdf'), lazy_load=True, copy_arrays=True)
                                        for i in self.chunk_inds]
            else:
                halo_particle_afs = []

            if 'field' in self.load_halofield:
                # a little repetitious, but perhaps it's better to be explicit
                field_particle_afs = [asdf.open(pjoin(self.groupdir, f'field_{RVorPID}_{AB}', f'field_{RVorPID}_{AB}_{i:03d}.asdf'), lazy_load=True, copy_arrays=True)
                                        for i in self.chunk_inds]
            else:
                field_particle_afs = []

            # Should have same number of files (1st subsample; 2nd L1), but note that empty slabs don't get files
            # TODO: double check these asserts
            try:
                assert len(N_halo_per_file) <= len(halo_particle_afs) or len(N_halo_per_file) <= len(field_particle_afs)
            except:
                assert len(N_halo_per_file) == len(halo_particle_afs) or len(self.halo_fns) == len(field_particle_afs)
            particle_afs = halo_particle_afs + field_particle_afs

            if not self._reindexed[AB] and 'halo' in self.load_halofield and 'npstart'+AB in self.halos.colnames:
                # Halos only index halo particles; no need to do this if we're just loading field particles!
                # Offset npstartB in case the user is loading both subsample A and B.  Also accounts for field particles.
                self.halos['npstart'+AB] += np_total
                _reindex_subsamples_from_asdf_size(self.halos['npstart'+AB],
                                                  [af[self.data_key][asdf_col_name] for af in halo_particle_afs],
                                                  N_halo_per_file)
                self._reindexed[AB] = True

            # total number of particles
            for af in particle_afs:
                np_per_file += [len(af[self.data_key][asdf_col_name])]
            np_total = np.sum(np_per_file)
            particle_AB_afs += particle_afs

        np_per_file = np.array(np_per_file)
        return particle_AB_afs, np_per_file


    def _load_pids(self, unpack_bits, N_halo_per_file, check_pids=False):
        # Even if unpack_bits is False, return the PID-masked value, not the raw value.

        pid_AB_afs, np_per_file = self._reindex_subsamples('pid', N_halo_per_file)

        start = 0
        np_total = np.sum(np_per_file)
        pids_AB = np.empty(np_total, dtype=np.uint64)
        for i,af in enumerate(pid_AB_afs):
            thisnp = np_per_file[i]
            if not unpack_bits:
                pids_AB[start:start+thisnp] = af[self.data_key]['packedpid'] & AUXPID
            else:
                pids_AB[start:start+thisnp] = af[self.data_key]['packedpid']
            start += thisnp

        # Could be expensive!  Off by default.  Probably faster ways to implement this.
        if check_pids:
            assert len(np.unique(pids_AB)) == len(pids_AB)

        if unpack_bits:
            # unpack_pids will do unit conversion if requested
            unpackbox = self.header['BoxSize'] if self.convert_units else 1.
            justpid, lagr_pos, tagged, density = unpack_pids(pids_AB, unpackbox, self.header['ppd'])
            self.subsamples.add_column(lagr_pos, name='lagr_pos', copy=False)
            self.subsamples.add_column(tagged, name='tagged', copy=False)
            self.subsamples.add_column(density, name='density', copy=False)
            self.subsamples.add_column(justpid, name='pid', copy=False)
            #self.subsamples.add_column(pids_AB, name='packedpid', copy=False)
        else:
            self.subsamples.add_column(pids_AB, name='pid', copy=False)


    def _load_RVs(self, N_halo_per_file, unpack=True):

        particle_AB_afs, np_per_file = self._reindex_subsamples('rv', N_halo_per_file)

        start = 0
        np_total = np.sum(np_per_file)
        particles_AB = np.empty((np_total,3),dtype=np.int32)
        for i,af in enumerate(particle_AB_afs):
            thisnp = np_per_file[i]
            particles_AB[start:start+thisnp] = af[self.data_key]['rvint']
            start += thisnp
        # SOWNAK
        if unpack:
            unpackbox = self.header['BoxSize'] if self.convert_units else 1.
            ppos_AB, pvel_AB = unpack_rvint(particles_AB, unpackbox)

            self.subsamples.add_column(ppos_AB, name='pos', copy=False)
            self.subsamples.add_column(pvel_AB, name='vel', copy=False)
        else:
            self.subsamples.add_column(particles_AB, name='rvint', copy=False)

def _reindex_subsamples_from_asdf_size(subsamp_start, particle_arrays, N_halo_per_file):
    '''
    For subsample redshifts where we have L1s followed by L0s in the halo_pids files,
    we need to reindex using the total number of PIDs in the file, not the npout fields,
    which only have the L1s.
    '''

    nh = 0
    for k,p in enumerate(particle_arrays):
        nh += N_halo_per_file[k]
        np_thisfile = len(p)
        subsamp_start[nh:] += np_thisfile

####################################################################################################
# The following constants and functions relate to unpacking our compressed halo and particle formats
####################################################################################################

# Constants
EULER_ABIN = 45
EULER_TBIN = 11
EULER_NORM = 1.8477590650225735122 # 1/sqrt(1-1/sqrt(2))

INT16SCALE = 32000.

# unpack the eigenvectors
def unpack_euler16(bin_this):
    N = bin_this.shape[0]
    minor = np.zeros((N,3))
    middle = np.zeros((N,3))
    major = np.zeros((N,3))

    cap = bin_this//EULER_ABIN
    iaz = bin_this - cap*EULER_ABIN   # This is the minor axis bin_this
    bin_this = cap
    cap = bin_this//(EULER_TBIN*EULER_TBIN)   # This is the cap
    bin_this = bin_this - cap*(EULER_TBIN*EULER_TBIN)

    it = (np.floor(np.sqrt(bin_this))).astype(int)
    its = np.sum(np.isnan(it))


    ir = bin_this - it*it

    t = (it+0.5)*(1.0/EULER_TBIN)   # [0,1]
    r = (ir+0.5)/(it+0.5)-1.0            # [-1,1]

    # We need to undo the transformation of t to get back to yy/zz
    t *= 1/EULER_NORM
    t = t * np.sqrt(2.0-t*t)/(1.0-t*t)   # Now we have yy/zz

    yy = t
    xx = r*t
    # and zz=1
    norm = 1.0/np.sqrt(1.0+xx*xx+yy*yy)
    zz = norm
    yy *= norm; xx *= norm;  # These are now a unit vector

    # TODO: legacy code, rewrite
    major[cap==0,0] = zz[cap==0]; major[cap==0,1] = yy[cap==0]; major[cap==0,2] = xx[cap==0];
    major[cap==1,0] = zz[cap==1]; major[cap==1,1] =-yy[cap==1]; major[cap==1,2] = xx[cap==1];
    major[cap==2,0] = zz[cap==2]; major[cap==2,1] = xx[cap==2]; major[cap==2,2] = yy[cap==2];
    major[cap==3,0] = zz[cap==3]; major[cap==3,1] = xx[cap==3]; major[cap==3,2] =-yy[cap==3];

    major[cap==4,1] = zz[cap==4]; major[cap==4,2] = yy[cap==4]; major[cap==4,0] = xx[cap==4];
    major[cap==5,1] = zz[cap==5]; major[cap==5,2] =-yy[cap==5]; major[cap==5,0] = xx[cap==5];
    major[cap==6,1] = zz[cap==6]; major[cap==6,2] = xx[cap==6]; major[cap==6,0] = yy[cap==6];
    major[cap==7,1] = zz[cap==7]; major[cap==7,2] = xx[cap==7]; major[cap==7,0] =-yy[cap==7];

    major[cap==8,2] = zz[cap==8]; major[cap==8,0] = yy[cap==8]; major[cap==8,1] = xx[cap==8];
    major[cap==9,2] = zz[cap==9]; major[cap==9,0] =-yy[cap==9]; major[cap==9,1] = xx[cap==9];
    major[cap==10,2] = zz[cap==10]; major[cap==10,0] = xx[cap==10]; major[cap==10,1] = yy[cap==10];
    major[cap==11,2] = zz[cap==11]; major[cap==11,0] = xx[cap==11]; major[cap==11,1] =-yy[cap==11];

    # Next, we can get the minor axis
    az = (iaz+0.5)*(1.0/EULER_ABIN)*np.pi
    xx = np.cos(az)
    yy = np.sin(az)
    # print("az = %f, %f, %f\n", az, xx, yy)
    # We have to derive the 3rd coord, using the fact that the two axes
    # are perpendicular.

    eq2 = (cap//4) == 2
    minor[eq2,0] = xx[eq2]; minor[eq2,1] = yy[eq2];
    minor[eq2,2] = (minor[eq2,0]*major[eq2,0]+minor[eq2,1]*major[eq2,1])/(-major[eq2,2])
    eq4 = (cap//4) == 0
    minor[eq4,1] = xx[eq4]; minor[eq4,2] = yy[eq4];
    minor[eq4,0] = (minor[eq4,1]*major[eq4,1]+minor[eq4,2]*major[eq4,2])/(-major[eq4,0])
    eq1 = (cap//4) == 1
    minor[eq1,2] = xx[eq1]; minor[eq1,0] = yy[eq1];
    minor[eq1,1] = (minor[eq1,2]*major[eq1,2]+minor[eq1,0]*major[eq1,0])/(-major[eq1,1])
    minor *= (1./np.linalg.norm(minor,axis=1).reshape(N,1))

    middle = np.zeros((minor.shape[0],3))
    middle[:,0] = minor[:,1]*major[:,2]-minor[:,2]*major[:,1]
    middle[:,1] = minor[:,2]*major[:,0]-minor[:,0]*major[:,2]
    middle[:,2] = minor[:,0]*major[:,1]-minor[:,1]*major[:,0]
    middle *= (1./np.linalg.norm(middle,axis=1).reshape(N,1))
    return minor, middle, major


"""
struct HaloStat {
    uint64_t id;    ///< A unique halo number.
    uint64_t npstartA;  ///< Where to start counting in the particle output for subsample A
    uint64_t npstartB;  ///< Where to start counting in the particle output for subsample B
    uint32_t npoutA;    ///< Number of taggable particles pos/vel/aux written out in subsample A
    uint32_t npoutB;    ///< Number of taggable particles pos/vel/aux written out in subsample B
    uint32_t ntaggedA;      ///< Number of tagged particle PIDs written out in subsample A. A particle is tagged if it is taggable and is in the largest L2 halo for a given L1 halo.
    uint32_t ntaggedB;
    uint32_t N; ///< The number of particles in this halo
    uint32_t L2_N[N_LARGEST_SUBHALOS];   ///< The number of particles in the largest L2 subhalos
    uint32_t L0_N;    ///< The number of particles in the L0 parent group

    float x_com[3];      ///< Center of mass position
    float v_com[3];      ///< Center of mass velocity
    float sigmav3d_com;  ///< Sum of eigenvalues
    float meanSpeed_com;  ///< Mean speed (the norm of the velocity vector)
    float sigmav3d_r50_com;  ///< Velocity dispersion of the inner 50% of particles
    float meanSpeed_r50_com;  ///< Mean speed of the inner 50% of particles
    float r100_com; ///<Radius of 100% of mass
    float vcirc_max_com; ///< max circular velocity, based on the particles in this L1 halo
    float SO_central_particle[3]; ///< Coordinates of the SO central particle
    float SO_central_density;  ///< Density of the SO central particle.
    float SO_radius;           ///< Radius of SO halo (distance to particle furthest from central particle)

    float x_L2com[3];   ///< Center of mass pos of the largest L2 subhalo
    float v_L2com[3];   ///< Center of mass vel of the largest L2 subhalo
    float sigmav3d_L2com;  ///< Sum of eigenvalues
    float meanSpeed_L2com;  ///< Mean speed
    float sigmav3d_r50_L2com;  ///< Velocity dispersion of the inner 50% of particles
    float meanSpeed_r50_L2com;  ///< Mean speed of the inner 50% of particles
    float r100_L2com; /// Radius of 100% of mass, relative to L2 center.
    float vcirc_max_L2com;   ///< max circular velocity, based on the particles in this L1 halo
    float SO_L2max_central_particle[3]; ///< Coordinates of the SO central particle for the largest L2 subhalo.
    float SO_L2max_central_density;  ///< Density of the SO central particle of the largest L2 subhalo.
    float SO_L2max_radius;           ///< Radius of SO halo (distance to particle furthest from central particle) for the largest L2 subhalo

    int16_t sigmavMin_to_sigmav3d_com; ///< Min(sigmav_eigenvalue) / sigmav3d, compressed
    int16_t sigmavMax_to_sigmav3d_com; ///< Max(sigmav_eigenvalue) / sigmav3d, compressed
    uint16_t sigmav_eigenvecs_com;  ///<Eigenvectors of the velocity dispersion tensor, compressed into 16 bits.
    int16_t sigmavrad_to_sigmav3d_com; ///< sigmav_rad / sigmav3d, compressed
    int16_t sigmavtan_to_sigmav3d_com; ///< sigmav_tan / sigmav3d, compressed

    int16_t r10_com, r25_com, r33_com, r50_com, r67_com, r75_com, r90_com, r95_com, r98_com; ///<Expressed as ratios of r100, and scaled to 32000 to store as int16s.
    int16_t sigmar_com[3]; ///<sqrt( Eigenvalues of the moment of inertia tensor ), sorted largest to smallest
    int16_t sigman_com[3]; ///<sqrt( Eigenvalues of the weighted moment of inertia tensor ), sorted largest to smallest
    uint16_t sigmar_eigenvecs_com;  ///<Eigenvectors of the moment of inertia tensor, compressed into 16 bits. Compression format Euler16.
    uint16_t sigman_eigenvecs_com;  ///<Eigenvectors of the weighted moment of inertia tensor, compressed into 16 bits. Compression format Euler16.
    int16_t rvcirc_max_com; ///< radius of max velocity, stored as int16 ratio of r100 scaled by 32000.

    // The largest (most massive) subhalo center of mass
    int16_t sigmavMin_to_sigmav3d_L2com; ///< Min(sigmav_eigenvalue) / sigmav3d, compressed
    int16_t sigmavMax_to_sigmav3d_L2com; ///< Max(sigmav_eigenvalue) / sigmav3d, compressed
    uint16_t sigmav_eigenvecs_L2com;  ///<Eigenvectors of the velocity dispersion tensor, compressed into 16 bits.
    int16_t sigmavrad_to_sigmav3d_L2com; ///< sigmav_rad / sigmav3d, compressed
    int16_t sigmavtan_to_sigmav3d_L2com; ///< sigmav_tan / sigmav3d, compressed
    int16_t r10_L2com, r25_L2com, r33_L2com, r50_L2com, r67_L2com, r75_L2com, r90_L2com, r95_L2com, r98_L2com;
        ///< Radii of this percentage of mass, relative to L2 center. Expressed as ratios of r100 and compressed to int16.

    int16_t sigmar_L2com[3];
    int16_t sigman_L2com[3];
    uint16_t sigmar_eigenvecs_L2com;   ///< euler16 format
    uint16_t sigman_eigenvecs_L2com;   ///< euler16 format
    int16_t rvcirc_max_L2com;   ///< radius of max circular velocity, stored as ratio to r100, relative to L2 center

};
"""

# Note we never actually create a Numpy array with this dtype
# But it is a useful format for parsing the needed dtypes for the Astropy table columns
# Could automatically generate this from the raw dtype, but perhaps an explicit listing is helpful

clean_dt = np.dtype([('npstartA_merge', np.int64),
                     ('npstartB_merge', np.int64),
                     ('npoutA_merge', np.uint32),
                     ('npoutB_merge', np.uint32),
                     ('N_total', np.uint32),
                     ('N_merge', np.uint32),
                     ('is_merged_to', np.int64),
                     ], align=True)

user_dt = np.dtype([('id', np.uint64),
                    ('npstartA', np.uint64),
                    ('npstartB', np.uint64),

                    ('npoutA', np.uint32),
                    ('npoutB', np.uint32),
                    ('ntaggedA', np.uint32),
                    ('ntaggedB', np.uint32),
                    ('N', np.uint32),
                    ('L2_N', np.uint32, 5),
                    ('L0_N', np.uint32),

                    ('x_com', np.float32, 3),
                    ('v_com', np.float32, 3),
                    ('sigmav3d_com', np.float32),
                    ('meanSpeed_com', np.float32),
                    ('sigmav3d_r50_com', np.float32),
                    ('meanSpeed_r50_com', np.float32),
                    ('r100_com', np.float32),
                    ('vcirc_max_com', np.float32),
                    ('SO_central_particle', np.float32, 3),
                    ('SO_central_density', np.float32),
                    ('SO_radius', np.float32),

                    ('x_L2com', np.float32, 3),
                    ('v_L2com', np.float32, 3),
                    ('sigmav3d_L2com', np.float32),
                    ('meanSpeed_L2com', np.float32),
                    ('sigmav3d_r50_L2com', np.float32),
                    ('meanSpeed_r50_L2com', np.float32),
                    ('r100_L2com', np.float32),
                    ('vcirc_max_L2com', np.float32),
                    ('SO_L2max_central_particle', np.float32, 3),
                    ('SO_L2max_central_density', np.float32),
                    ('SO_L2max_radius', np.float32),

                    ('sigmavMin_com', np.float32),
                    ('sigmavMid_com', np.float32),
                    ('sigmavMaj_com', np.float32),

                    ('r10_com', np.float32),
                    ('r25_com', np.float32),
                    ('r33_com', np.float32),
                    ('r50_com', np.float32),
                    ('r67_com', np.float32),
                    ('r75_com', np.float32),
                    ('r90_com', np.float32),
                    ('r95_com', np.float32),
                    ('r98_com', np.float32),

                    ('sigmar_com', np.float32, 3),
                    ('sigman_com', np.float32, 3),
                    ('sigmar_eigenvecsMin_com', np.float32, 3),
                    ('sigmar_eigenvecsMid_com', np.float32, 3),
                    ('sigmar_eigenvecsMaj_com', np.float32, 3),
                    ('sigmav_eigenvecsMin_com', np.float32, 3),
                    ('sigmav_eigenvecsMid_com', np.float32, 3),
                    ('sigmav_eigenvecsMaj_com', np.float32, 3),
                    ('sigman_eigenvecsMin_com', np.float32, 3),
                    ('sigman_eigenvecsMid_com', np.float32, 3),
                    ('sigman_eigenvecsMaj_com', np.float32, 3),

                    ('sigmavrad_com', np.float32),
                    ('sigmavtan_com', np.float32),
                    ('rvcirc_max_com', np.float32),

                    ('sigmavMin_L2com', np.float32),
                    ('sigmavMid_L2com', np.float32),
                    ('sigmavMaj_L2com', np.float32),

                    ('r10_L2com', np.float32),
                    ('r25_L2com', np.float32),
                    ('r33_L2com', np.float32),
                    ('r50_L2com', np.float32),
                    ('r67_L2com', np.float32),
                    ('r75_L2com', np.float32),
                    ('r90_L2com', np.float32),
                    ('r95_L2com', np.float32),
                    ('r98_L2com', np.float32),

                    ('sigmar_L2com', np.float32, 3),
                    ('sigman_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMin_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMid_L2com', np.float32, 3),
                    ('sigmar_eigenvecsMaj_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMin_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMid_L2com', np.float32, 3),
                    ('sigmav_eigenvecsMaj_L2com', np.float32, 3),
                    ('sigman_eigenvecsMin_L2com', np.float32, 3),
                    ('sigman_eigenvecsMid_L2com', np.float32, 3),
                    ('sigman_eigenvecsMaj_L2com', np.float32, 3),

                    ('sigmavrad_L2com', np.float32),
                    ('sigmavtan_L2com', np.float32),
                    ('rvcirc_max_L2com', np.float32),
], align=True)
