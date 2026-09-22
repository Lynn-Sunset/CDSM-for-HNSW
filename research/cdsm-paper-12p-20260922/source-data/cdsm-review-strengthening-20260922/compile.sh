#!/bin/bash
set -euo pipefail
cd /home/testsv/cdsm-ann-20260920/research/cdsm-review-strengthening-20260922
export PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH
export LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p runtime
for item in bench check; do
  g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -ffp-contract=off -I../native-faiss-mechanism-20260913/upstream/faiss-1.15.0 "$item.cpp" ../ann-systematic-comparison-20260919/build/faiss/libfaiss_avx2.a /lib64/libopenblas.so.0 -lpthread -ldl -o "runtime/r2_$item"
done
echo R2_BUILD_COMPLETE
