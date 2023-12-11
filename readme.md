# Protect Your Score

This repo contains accompanying code for the AAAI 2023 paper:
'Protect Your Score: Contact Tracing with Differential Privacy Guarantees'.

The library is named DPFN, after the decentralized algorithm that we introduce.

## Typical usage

Starting point for experiments will be the following command:

For ABM:
```
python3 dpfn/experiments/compare_abm.py \
    --inference_method "fn" \
    --config_data intermediate_graph_abm_02 \
    --config_model model_ABM01
```

For COVASIM:
```
python3 dpfn/experiments/compare_abm.py \
    --inference_method "fn" \
    --config_data intermediate_graph_cv_02 \
    --simulator=covasim \
    --config_model model_CV01
```

Experiments take two configs: one for the model and one for the simulator (data).
These configs can be found in dpfn/config/model_*.ini. For example, in that config
one can change the epsilon and delta for Differential Privacy, or change the model
parameters. In dpfn/config/graph_*.ini, one can change the simulator parameters
like fraction_test or num_users.

Inference method 'fn' will run Factorised Neighbors and, depending on the other
configs, this experiment will be DP. We are in the process of open-sourcing a
C++ implementation of DPFN, whereby inference will be about 2x faster. Also,
memory issues for large scale experiments are resolved. In case the compilation
of the CPP util library does not work, one can simply remove any link to the 'fncpp'
inference method and you're able to run everything in python.

The most used experimental setup would be 'prequential'. This indicates the simulation
whose results can be found throughout the paper. Prequential is slang for sequential
prediction. At every 'day' of the simulation, the algorithm predicts covidscores
for all users and decides who to test for the covid virus. For further analysis, one could use
the experimental setup 'single'. This setup uses a static contact graph where one can
analyze properties of the inference algorithm such as likelihoods and evidence.

## Overview of important scripts

Most of the code for running inference with DPFN is in inference.py.
Utility functions for calculating differential privacy terms are in util.py and util_dp.py

The naming of variables in this code generally follows two works:
Herbrich et al. 2020, "Crisp", Romijnders et al. 2023, "No time to waste"

The two main simulations on OpenABM and COVASIM are run with dpfn/experiments/compare_abm.py. This file
contains the main for-loop that runs the simulation and logs metrics. At the start of each experiment,
the most recent git-commit and git-diff is logged for better reproducibility.

Different versions for obtaining a differentially private covidscore are numbered like
dp1, dp2, ..., dp7. These are explained in constants.py. The results for DPFN correspond to
dp_method=5.
An important parameter in the config is `noisy_test` as this controls the False Positive Rate
and False Negative Rate of the tests for covid. We distinguish between levels -1, 0, 1, 2, 3, 4.
Levels 0,1,2,3 correspond to the levels used in the paper. Level 4 has FPR=FNR=1/2 which turns the
covid test into a coinflip and this can be used for sanity checking some experiments (when the
covid test is a coinflip, then the PIR should be very high). Leaving noisy_test a negative number
will not change the FPR or FNR from their specified value or the default value.


## Code convention

Code convention: We care about good code and scientific reproducibility. As of August 2023, the code contains
72 unittests, spanning more than a thousand lines of code (run tests with `$ make test` or `nose2 -v` in the base directory).

Most functions include type hints (type hints can be checked with `$ make hint` or `pytype dpfn` in the base directory).

The code is style checked with the included '.pylintrc' and pycodestyle (style check with `$ make lint` or `pylint dpfn` in the base directory).

## Installation

Installation of OpenABM will require the GSL utility functions and the SWIG library to interop between C++ and Python.

For GSL, follow [these instructions](https://coral.ise.lehigh.edu/jild13/2016/07/11/hello/)

```
# get the installation file
wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz

# Unpack archive
tar -zxvf gsl-latest.tar.gz

# make a directory for the gsl installation
mkdir /var/scratch/${USER}/projects/gsl

# installation
./configure --prefix=/var/scratch/${USER}/projects/gsl
make
make check
make install
```

[SWIG](https://www.swig.org/) install
```
sudo apt-get update
sudo apt-get -y install swig
```

OpenABM install
```
# Get the ABM code
cd ../

mkdir abm
cd abm

git clone https://github.com/aleingrosso/OpenABM-Covid19.git .

cd src
make all

make swig-all
```

Insights from debugging:
  * 'gsl/gsl_rng.h: No such file or directory' -> make sure the includes are set correctly. Like -I/var/scratch/${USER}/projects/gsl/include to the compiler
  * 'cannot find -lgsl' -> Make sure the libraries are set correctly. Like -L/var/scratch/${USER}/projects/gsl/lib to the linker

## Run a sweep with WandB
To run a sweep with WandB, run the following command

`$ wandb sweep sweep/dpfn.yaml`

This command will run a sweepid that can be used on the computer where you'l run the actual experiments. For example,
you can setup the sweep from a local laptop, and run the actual experiments in the cloud or on a compute node.
On the target machine, start up an agent with

```
$ export SWEEP=sweepid
$ wandb agent "${WANDBUSERNAME}/dpfn-dpfn_experiments/$SWEEP"
```

## Attribution

The starting point for this repo was a fork of [our earlier project](https://github.com/QUVA-Lab/nttw).

This work is financially supported by Qualcomm Technologies Inc., the University of Amsterdam and the allowance Top consortia for Knowledge and Innovation (TKIs) from the Netherlands Ministry of Economic Affairs and Climate Policy.

All communication may go to romijndersrob@gmail.com or r.romijnders@uva.nl.
