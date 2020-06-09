# abacus_mergertree
Code to develop halo associations between Abacus timeslices.

This code has a few non-standard dependencies, including

* `numba`
* `joblib`
* `tqdm`

Each of these can be installed easily using `pip install package_name`.

One also needs to install the `abacusutils` package from
https://github.com/abacusorg/abacusutils.

Finally, the data format relies on a special ``abacus`` fork of the
``asdf`` package (https://asdf.readthedocs.io/en/2.6.0/). You can
download this version from https://github.com/lgarrison/asdf/.

# Running the code

The main script in this repository is the file called
``create_associations_slabwise.py``.  To find the list of arguments
that a user needs to pass to run the code, execute:

``python create_associations_slabwise.py --help``

# Output format