#!/usr/bin/env bash
#SBATCH -A AST145
#SBATCH -M dtn
#SBATCH -t 2-0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p batch
#SBATCH --mem 0  # all memory

set -e 

MERGER=$PROJWORK/AbacusSummit/merger
HPSSDIR=/hpss/prod/ast145/proj-shared/AbacusSummit/merger

hsi mkdir -p $HPSSDIR

cd $MERGER

for SIM in AbacusSummit_*; do
    echo $SIM
    htar -T 16 -cvf $HPSSDIR/${SIM}.tar $SIM
done

echo All done!
