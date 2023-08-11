# Protect Your Score: Contact Tracing With Differential Privacy Guarantees

This repo contains accompanying code for the anonymous submission

## Typical usage

Starting point for experiments will be the following command:

For ABM:
```
python3 dpfn/experiments/compare_abm.py \
    --inference_method "fn" \
    --experiment_setup "prequential" \
    --config_data intermediate_graph_abm_02 \
    --config_model model_ABM01
```

For COVASIM:
```
python3 dpfn/experiments/compare_covasim.py \
    --inference_method "fn" \
    --config_data intermediate_graph_cv_02 \
    --config_model model_CV01
```

Experiments take two configs: one for the model and one for the simulator (data).
These configs can be found in dpfn/config/model_*.ini. For example, in that config
one can change the epsilon and delta for Differential Privacy, or change the model
parameters.

The most used experimental setup would be 'prequential'. This indicates the simulation
whose results can be found throughout the paper. For further analysis, one could use
the experimental setup 'single'. This setup uses a static contact graph where one can
analyze properties of the inference algorithm such as likelihoods and evidence.

## Overview of important scripts

Most of the code for running inference with DPFN is in inference.py.
Utility functions for calculating DP terms are in util.py.

The two main simulations on OpenABM and COVASIM are run with dpfn/experiments/compare_abm.py
and dpfn/experiments/compare_covasim.py



## Code convention

Code convention: We care about good code and scientific reproducibility. As of August 2023, the code contains
72 unittests, spanning more than one thousands line of code (`$ make test` or `nose2 -v`).

The code includes type hints (type hints can be checked with `$ make hint` or `pytype dpfn`).

Code is styled with included '.pylintrc' and pycodestyle (`$ make lint` or `pylint dpfn`)

## Installation

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

ABM install
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

This readme is anonymous for double-blind review.
