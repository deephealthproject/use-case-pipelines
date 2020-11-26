# 3rd Hackathon python pipeline

__Tested with PyECVL 0.5.1__

This directory contains a Python version of the pipeline.

It requires [PyECVL](https://github.com/deephealthproject/pyecvl), which can
be installed following this
[guide](https://deephealthproject.github.io/pyecvl/installation.html). Note
that PyECVL/ECVL must be installed with
[EDDL](https://github.com/deephealthproject/eddl) support, so you also need to
install [PyEDDL](https://github.com/deephealthproject/pyeddl), as explained in
the instructions. Note that all software packages must be compiled with the
same compiler.

An alternative to manually installing the required packages is to use the
[DeepHealth Docker images](https://github.com/deephealthproject/docker-libs).



## Running the pipeline on GPU

If you want to run the pipeline on GPU, you need to install EDDL/PyEDDL with
GPU support. The [PyEDDL installation
guide](https://deephealthproject.github.io/pyeddl/installation.html) explains
how to do this.


## Development history

The Python version of the pipeline was originally hosted in the PyECVL
repository under
[examples/use_case_pipeline](https://github.com/deephealthproject/pyecvl/tree/eedc6041e548f850ccf3022a6165dbd1386978e9/examples/use_case_pipeline).
