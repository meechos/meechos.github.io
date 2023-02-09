---
layout: page
title: Accelerated Linear Algebra Libraries (MKL vs OpenBLAS)
permalink: /MKL_vs_OpenBLAS
---

# 

## 1. BLAS background
Accelerated Linear Algebra Libraries, also mostly known as Basic Linear Algebra Subprograms (BLAS), are a set of low-level routines for performing common linear algebra operations such as vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication. The implementations are often optimized for speed for example by taking advantage of special floating point hardware such as vector registers or SIMD instructions. Using them can bring substantial performance benefits. 

BLAS originated as a Fortran library in 1979 and its interface was standardized by the BLAS Technical (BLAST) Forum, whose latest BLAS report can be found on the netlib website. This Fortran library is known as the reference implementation (sometimes confusingly referred to as the BLAS library) and is not optimized for speed but is in the public domain.

Most libraries that offer linear algebra routines conform to the BLAS interface, allowing library users to develop programs that are indifferent to the BLAS library being used. CPU-based examples of BLAS libraries include: OpenBLAS, BLIS (BLAS-like Library Instantiation Software), Arm Performance Libraries, ATLAS, and Intel Math Kernel Library (MKL). 
- ATLAS is a portable library that automatically optimizes itself for an arbitrary architecture. 
- MKL is a freeware and proprietary vendor library optimized for x86 and x86-64 with a performance emphasis on Intel processors.
- OpenBLAS is an open-source library that is hand-optimized for many of the popular architectures. The LINPACK benchmarks rely heavily on the BLAS routine gemm for its performance measurements.

Many numerical software applications use BLAS-compatible libraries to do linear algebra computations, including GNU Octave, MATLAB, NumPy, R and Julia.



## 2 Benchmarking in Numpy
NumPy doesn’t depend on any other Python packages, however, it does depend on an accelerated linear algebra library - typically Intel MKL or OpenBLAS. The used BLAS can affect performance, behavior and size on disk:

- The MKL package is a lot larger than OpenBLAS, it’s about 700 MB on disk while OpenBLAS is about 30 MB.
- MKL is typically a little faster and more robust than OpenBLAS.

To build Numpy against the two different BLAS versions we have to use a `site.cfg` config or build two different enviroments. We will go with the second option.

### 2.1 Numpy with MKL 

Build an enviroment with numpy built against MKL by running the following command in terminal. 

```python
# create MKL environment
conda create -n MKL_env -c anaconda python numpy mkl=2019.* blas=*=*mkl
conda activate MKL_env
```

Confirm the used BLAS Numpy is using:
```
np.__config__.show()

blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/meechos/anaconda3/envs/MKL/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/meechos/anaconda3/envs/MKL/include']
blas_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/meechos/anaconda3/envs/MKL/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/meechos/anaconda3/envs/MKL/include']
lapack_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/meechos/anaconda3/envs/MKL/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/meechos/anaconda3/envs/MKL/include']
lapack_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/home/meechos/anaconda3/envs/MKL/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/home/meechos/anaconda3/envs/MKL/include']
```
Finally run a set of operations to measure run times.


```python
# Seed numpy for reproducibility.
np.random.seed(666)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

# Matrix multiplication
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
del A, B

# Vector multiplication
N = 5000
t = time()
for i in range(N):
    np.dot(C, D)
delta = time() - t
print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
del C, D

# Singular Value Decomposition (SVD)
N = 3
t = time()
for i in range(N):
    np.linalg.svd(E, full_matrices = False)
delta = time() - t
print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
del E

# Cholesky Decomposition
N = 3
t = time()
for i in range(N):
    np.linalg.cholesky(F)
delta = time() - t
print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

# Eigendecomposition
t = time()
for i in range(N):
    np.linalg.eig(G)
delta = time() - t
print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))
```



```
Dotted two 4096x4096 matrices in 0.95 s.
Dotted two vectors of length 524288 in 0.12 ms.
SVD of a 2048x1024 matrix in 0.30 s.
Cholesky decomposition of a 2048x2048 matrix in 0.10 s.
Eigendecomposition of a 2048x2048 matrix in 3.92 s.
```

### 2.2 Numpy with OpenBLAS 

Build an enviroment with the default OpenBLAS:

```python
conda create -n OpenBLAS python numpy
activate OpenBLAS
```


Confirm the used BLAS Numpy is using:
```
np.__config__.show()
openblas64__info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
    runtime_library_dirs = ['/usr/local/lib']
blas_ilp64_opt_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None)]
    runtime_library_dirs = ['/usr/local/lib']
openblas64__lapack_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
    runtime_library_dirs = ['/usr/local/lib']
lapack_ilp64_opt_info:
    libraries = ['openblas64_', 'openblas64_']
    library_dirs = ['/usr/local/lib']
    language = c
    define_macros = [('HAVE_CBLAS', None), ('BLAS_SYMBOL_SUFFIX', '64_'), ('HAVE_BLAS_ILP64', None), ('HAVE_LAPACKE', None)]
    runtime_library_dirs = ['/usr/local/lib']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
```

```python
# Seed numpy for reproducibility.
np.random.seed(666)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

# Matrix multiplication
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
del A, B

# Vector multiplication
N = 5000
t = time()
for i in range(N):
    np.dot(C, D)
delta = time() - t
print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
del C, D

# Singular Value Decomposition (SVD)
N = 3
t = time()
for i in range(N):
    np.linalg.svd(E, full_matrices = False)
delta = time() - t
print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
del E

# Cholesky Decomposition
N = 3
t = time()
for i in range(N):
    np.linalg.cholesky(F)
delta = time() - t
print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

# Eigendecomposition
t = time()
for i in range(N):
    np.linalg.eig(G)
delta = time() - t
print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))
```

```
Dotted two 4096x4096 matrices in 0.90 s.
Dotted two vectors of length 524288 in 0.11 ms.
SVD of a 2048x1024 matrix in 0.74 s.
Cholesky decomposition of a 2048x2048 matrix in 0.15 s.
Eigendecomposition of a 2048x2048 matrix in 9.41 s.
```

## 3. Discussion

In this simple demonstration we can see that MKL is performing significantly better when it comes to matrix decomposition operations. In specific, `eigendecomposition` was completed in half the time using the MKL library, as opposed to OpenBLAS.

Please note, this demo does not constitute a holistic benchmarking procedure but rather an example of how different BLAS libraries can be built against and used in Numpy.

For benchmarking of the libraries across engines with a variety of memory sizes and processors see 
- https://github.com/tmolteno/necpp/issues/18
- https://shaalltime.medium.com/benchmark-numpy-with-openblas-and-mkl-library-on-amd-ryzen-3950x-cpu-96184f91057f


---

### References:

- https://github.com/numpy/numpy/blob/main/site.cfg.example
- https://stackoverflow.com/a/38189809
- https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/
- https://www.intel.com/content/www/us/en/developer/articles/technical/numpyscipy-with-intel-mkl.html
- https://docs.continuum.io/mkl-optimizations/
- https://www.intel.com/content/www/us/en/developer/articles/technical/numpyscipy-with-intel-mkl.html
- https://en.wikipedia.org/wiki/Basic_Linear_Algebra_SubprogramsQ

