# abacus_mergertree
Code to develop halo associations between Abacus timeslices.

This code has a few non-standard dependices, including

* `numba`
* `joblib`
* `asdf`
* `tqdm`

Each of these can be installed easily using `pip install package_name`.

One also needs to install the `abacusutils` package from https://github.com/abacusorg/abacusutils.

# Running the code

The main script in this repository is the file called ``create_associations_slabwise.py``.
To find the list of arguments that a user needs to pass to run the code, execute:

``python create_associations_slabwise.py --help``