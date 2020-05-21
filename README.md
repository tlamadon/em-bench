# EM benchmark

We benchmark different implementations of the exact same algorithm. Each implementaiton is wrapped into a script that can be called from the command-line.

We implement the EM Algorithm using three different approaches in Python.
  - NumPy
  - Numba
  - SKLearn
  - Tensorflow


## Setting up the environment

Running the code requires a few dependencies. To keep things simple we try to rely entirely on conda and provided an environement file. You can then setup a system locally by running:

```bash
conda env create -f environment.yml
conda activate em-bench
```

You can then call a python estimator by running

```bash
python python/main.py --iter 100 --nobs 1000 --estimator numpy -o result.json
```
