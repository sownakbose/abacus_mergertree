#!/usr/bin/env bash

# Parallel scheme:
# - Break 170 simulation up into <=6 sets, each <=64 sims
# - In each set, chain together 4 jobs. Each job must be < 36 hours
# - Each node will do 8 epochs of 1 sim with one invocation of the merger script (16 cores). Presumably num_chunks=34.

set -e
shopt -s extglob

ABACUSSUMMIT=$PROJWORK/AbacusSummit

#NTOT=$(find ${ABACUSSUMMIT} -maxdepth 1 -name 'AbacusSummit_base_c???_ph???' | wc -l)
#SETS=('AbacusSummit_base_c{000..004}_ph???' 'AbacusSummit_base_c{009..020}_ph000' 'AbacusSummit_base_c{100..126}_ph000' 'AbacusSummit_base_c{130..181}_ph000')
#HRS_PER_JOB=36

NTOT=$(find ${ABACUSSUMMIT} -maxdepth 1 -name 'AbacusSummit_hugebase_c???_ph???' | wc -l)
SETS=('AbacusSummit_hugebase_c???_ph{000..012}' 'AbacusSummit_hugebase_c???_ph{013..024}')
HRS_PER_JOB=12

#SETS=('AbacusSummit_huge_c000_ph{201..202}')

mkdir -p logs

NRUNNING=0
for SET in ${SETS[@]}; do
    #echo $ABACUSSUMMIT/$SET
    eval SIMS=($ABACUSSUMMIT/$SET)
    SIMS=($(xargs -n1 basename <<< "${SIMS[@]}"))
    NTHISSET=${#SIMS[@]}
    echo ${NTHISSET}

    # We are explicitly assuming that each sim has 33 epochs (32 association jobs). Check this!
    for SIM in ${SIMS[@]}; do
        NEPOCH=$(find $ABACUSSUMMIT/$SIM/halos -maxdepth 1 -name 'z*' -type d | wc -l)
        [ $NEPOCH -eq 33 ] || { echo "Only $NEPOCH epochs in $SIM"; exit 1; }
    done

    NRUNNING=$(( NRUNNING + ${NTHISSET} ))

    SIMFN=$(mktemp sims.XXXX.txt)
    (IFS=$'\n'; echo "${SIMS[*]}" > ${SIMFN})

    JOBID=$(sbatch --ntasks-per-node=1 -c16 --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --wrap "srun rhea.sh ${SIMFN}")
    JOBID=$(sbatch --ntasks-per-node=1 -c16 --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "srun rhea.sh ${SIMFN}")
    JOBID=$(sbatch --ntasks-per-node=1 -c16 --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "srun rhea.sh ${SIMFN}")
    JOBID=$(sbatch --ntasks-per-node=1 -c16 --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "srun rhea.sh ${SIMFN}")
done

echo $NRUNNING, $NTOT

#echo $(echo "${SIMS}" | wc -l)
