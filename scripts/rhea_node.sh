#!/usr/bin/env bash

# Same as rhea.sh, but for the node associations script

set -e

SIMLISTFN=$1
SIMNAME=$(awk -v NUM=$SLURM_PROCID '{if(NR==NUM+1) print $0}' ${SIMLISTFN})

exec >> logs/${SIMNAME}.nodes.out 2>&1

#SIMNAME=${1:-AbacusSummit_highbase_c000_ph100}
ABACUSSUMMIT=$PROJWORK/AbacusSummit
OUTDIR=$ABACUSSUMMIT/merger

NCORES=${SLURM_CPUS_PER_TASK:-16}

cd $HOME/abacus_mergertree

echo "Starting node association for ${SIMNAME} on $(hostname) with ${NCORES} cores"

./create_node_associations.py -inputdir "$ABACUSSUMMIT" \
        -simname $SIMNAME \
        -num_chunks=-1 \
        -num_cores $NCORES \
        -outputdir $OUTDIR \
        -num_epochs 3
        #-num_slabs_todo -1

echo "Done node associations."
