from setuptools import setup

setup(name='deeplate',
      version='0.1',
      description='Bacteria microscopy image segmentation using deep learning ',
      url='https://github.com/guiwitz/DeepPlateSegmenter',
      author='Guillaume Witz',
      author_email='',
      license='MIT',
      packages=['deeplate'],
      zip_safe=False,
      install_requires=['numpy','scikit-image','scipy','keras','jupyter','pandas','h5py','jupyterlab'],
      )