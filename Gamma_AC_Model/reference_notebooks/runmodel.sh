#!/bin/bash
set -xve

# Activate conda environment
. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate agcm_environment

# Confirm conda and python location (for testing purposes)
which conda
which python
echo $CONDA_DEFAULT_ENV

expname='Test'
toffset=6540
ichunk=3600
#toffset=2970
#ichunk=120

./RunModel.Gamma.py --expname ${expname} --toffset ${toffset} --ichunk ${ichunk}




