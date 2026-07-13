#!/bin/bash
set -xve

# Activate conda environment
. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate nashmetpy

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

declare -a expnames=( 'AC_Test' )

dayst=36300

# Loop over all the Experiments
for exp in "${expnames[@]}"
do
   echo "$i"
   ./PressureInterpMetPy.py --expname ${exp} --dayst ${dayst}

done




