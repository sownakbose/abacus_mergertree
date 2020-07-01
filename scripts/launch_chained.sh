#!/usr/bin/env bash

# Parallel scheme:
# - Break 170 simulation up into <=6 sets, each <=64 sims
# - In each set, chain together 4 jobs. Each job must be < 36 hours
# - Each node will do 8 epochs of 1 sim with one invocation of the merger script (16 cores). Presumably num_chunks=34.

set -e
shopt -s extglob

ABACUSSUMMIT=$PROJWORK/AbacusSummit

NTOT=$(find ${ABACUSSUMMIT} -maxdepth 1 -name 'AbacusSummit_base_c???_ph???' | wc -l)
#SCRIPT=rhea.sh
#SETFNS=(set7.txt set8.txt set9.txt set11.txt set_incomplete.txt )
#SETFNS=(set13.txt)

SCRIPT=rhea_node.sh
SETFNS=(set14.txt)

HRS_PER_JOB=36
CORES=16
PART=batch

#SETFNS=(set6.txt)
#HRS_PER_JOB=48
#CORES=28
#PART=gpu

mkdir -p logs

NRUNNING=0
for SETFN in ${SETFNS[@]}; do
    readarray -t SIMS < $SETFN
    NTHISSET=${#SIMS[@]}
    echo ${NTHISSET}

    # We are explicitly assuming that each sim has 33 epochs (32 association jobs). Check this!
    for SIM in ${SIMS[@]}; do
        NEPOCH=$(find $ABACUSSUMMIT/$SIM/halos -maxdepth 1 -name 'z*' -type d | wc -l)
        [ $NEPOCH -eq 33 ] || { echo "Only $NEPOCH epochs in $SIM"; exit 1; }
    done

    NRUNNING=$(( NRUNNING + ${NTHISSET} ))

    if [[ ${SCRIPT} == *node* ]]; then
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --wrap "echo node ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo first node srun complete")
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "echo node ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo second node srun complete")
    else
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --wrap "echo ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo first srun complete")
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "echo ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo second srun complete")
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "echo ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo third srun complete")
        JOBID=$(sbatch -p ${PART} --ntasks-per-node=1 -c${CORES} --mem=0 -A AST145 -t 0-${HRS_PER_JOB} -N ${NTHISSET} --parsable --depend=afterok:${JOBID} --kill-on-invalid-dep=yes --wrap "echo ${SETFN}; srun -K0 ${SCRIPT} ${SETFN} && echo Last srun complete")
    fi
done

echo $NRUNNING, $NTOT

#echo $(echo "${SIMS}" | wc -l)
