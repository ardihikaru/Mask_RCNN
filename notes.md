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
4. 
