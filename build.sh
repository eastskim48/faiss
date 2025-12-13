pip3 uninstall faiss -y
rm -rf build
cmake -B build \
    -DFAISS_ENABLE_GPU=ON \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release
make -C build -j faiss
make -C build -j swigfaiss
pip install build/faiss/python
