#!/usr/bin/env python3
'''
Take the halo association ASDF files and:
- reduce the type widths of certain columns
- delete certain columns
- apply blosc compression
'''

import argparse
import os

import numpy as np
import asdf
import asdf.compression

FLOAT32_KEYS = ['HaloMass', 'MainProgenitorFrac', 'MainProgenitorPrecFrac', 'HaloVmax']
INT8_KEYS = ['IsAssociated', 'IsPotentialSplit']
INT32_KEYS = ['NumProgenitors']

asdf_compression = 'blsc'
asdf.compression.validate(asdf_compression)
asdf.compression.set_compression_options(asdf_block_size=12*1024**2, blocksize=3*1024**2, nthreads=4, typesize='auto', shuffle='shuffle')

def minify(fn, inplace=False,
            float32_keys=FLOAT32_KEYS,
            int8_keys=INT8_KEYS,
            int32_keys=INT32_KEYS,
            delete_keys=[]):
    
    newfn = fn + '.minified'

    INT8_MAX = np.iinfo(np.int8).max
    INT32_MAX = np.iinfo(np.int32).max

    with asdf.open(fn, lazy_load=False, memmap=False, mode='r') as af:

        tree = {'data':{},
                'header':af.tree['header']}
        data = tree['data']
        for key in af.tree['data']:
            # deletions
            if key in delete_keys:
                continue

            elif key in float32_keys:
                data[key] = af.tree['data'][key].astype(np.float32)
                assert np.isfinite(data[key]).all()

            elif key in int8_keys:
                assert (af.tree['data'][key] <= INT8_MAX).all()
                assert (af.tree['data'][key] >= 0).all()
                data[key] = af.tree['data'][key].astype(np.int8)

            elif key in int32_keys:
                assert (af.tree['data'][key] <= INT32_MAX).all()
                assert (af.tree['data'][key] >= 0).all()
                data[key] = af.tree['data'][key].astype(np.int32)

            else:
                data[key] = np.array(af.tree['data'][key])

    newaf = asdf.AsdfFile(tree)
    newaf.write_to(newfn, all_array_compression=asdf_compression)

    if inplace:
        os.remove(fn)
        os.rename(newfn, fn)


class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)

    parser.add_argument('asdf-file', help='The file to minify', nargs='+')
    parser.add_argument('--inplace', help='Overwrite the original file', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    for fn in args.pop('asdf-file'):
        minify(fn=fn, **args)
