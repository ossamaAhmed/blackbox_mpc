from setuptools import setup
from setuptools import find_packages
from os.path import splitext
from glob import glob
from os.path import basename

setup(name='tf_neuralmpc',
      version='1.0.0',
      author="Ossama Ahmed",
      install_requires=['gym',
                        'xmltodict'],
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      include_package_data=True,
)