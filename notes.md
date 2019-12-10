untuk persiapan environment python:
1. Conda:
   $ conda create -n mask34 python=3.4 -c conda-forge
   $ conda activate mask34
2. Dependency library:
   $ pip install tensorflow==1.3.0
   $ pip install tensorflow-gpu==1.3.0
   $ pip install keras==2.0.8
   $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
3. Running setup:
   $ python setup.py install
4. Another dep:
   $ sudo apt install nvidia-cuda-dev
5. Error: ImportError: libcublas.so.8.0
   Solution: https://github.com/tensorflow/tensorflow/issues/16136
   Steps:
   1. Edit: $ sudo vi /etc/profile
   2. Add 2 lines:
   ```
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
   3. Save changes:
   $ source /etc/profile
6. Error: AttributeError: module 'tensorflow' has no attribute 'random_uniform'
   Solution: https://www.tensorflow.org/api_docs/python/tf/keras/backend/random_uniform
7. Error: AttributeError: module 'tensorflow' has no attribute 'Session'
   Solution: 

