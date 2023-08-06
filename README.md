# PyQuda

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython.

This project aims to benifit from the optimized linear algebra library [CuPy](https://github.com/cupy/cupy) in Python based on CUDA. CuPy and QUDA will allow us to do most operation in lattice QCD research with high performance.

This project is based on the latest QUDA release v1.1.0.

Use `pip install .` to build and install the wrapper library, or use `python3 setup.py build_ext --inplace` to build the library in the repo folder and not install it. You need to build `libquda.so` and move it to the repo folder as a prerequisite.

## Installation

### QUDA

This is an example to build QUDA for single GPU.

```bash
git clone https://github.com/CLQCD/quda.git
pushd quda
git checkout b47950dd
mkdir build
pushd build
cmake .. -DQUDA_DIRAC_DOMAIN_WALL=OFF -DQUDA_CLOVER_DYNAMIC=OFF -DQUDA_CLOVER_RECONSTRUCT=OFF -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF -DQUDA_DIRAC_TWISTED_CLOVER=OFF -DQUDA_DIRAC_TWISTED_MASS=OFF -DQUDA_INTERFACE_MILC=OFF -DQUDA_LAPLACE=ON -DQUDA_MULTIGRID=ON
cmake --build . -j8
cmake --install .
popd
popd
```

### PyQuda

Build, install and run the example.

`chroma` is needed here for generating the reference file.

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
git clone https://github.com/IHEP-LQCD/PyQuda.git
pushd PyQuda
python3 -m pip install -r requirements.txt
cp ../quda/build/lib/libquda.so ./
python3 -m pip install .

chroma -i tests/test.clover.ini.xml
python3 tests/test.clover.py
popd
```

### MyDslash
My dslash is based provide implement of dslash interface, then you can use these functions in PyQuda.
First you should `source env.sh`.
If you have installed PyQuda and QUDA, then it is easy to use my files.
First, go to directory PyQuda/qcu, then you can choose compile GPU/CPU src.
For example, you entered gpuSrc, then you execute `make` and then `make install`, new `libqcu.so` will be stored into PyQuda/lib, then you can execute Python code in `tests`. By the way, donnot forget to copy `libquda.so` from QUDA.
```bash
pushd PyQuda
source env.sh
popd
```
