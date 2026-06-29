#!/bin/bash
set -xve

# Activate conda environment
. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate agcm_environment

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

declare -a arr=( 'HeatingJJA_1979-2023' )

#declare -a arr=( 'heatingJJA_1979-2023_tropics')

toffset=0
ichunk=1

# Loop over all the Experiments
for i in "${arr[@]}"
do
   echo "$i"
   ./RunModel.betaJJA.py --expname ${i} --toffset ${toffset} --ichunk ${ichunk}

done




