# -*- coding: utf-8 -*-

"""
Setup file for the uchuutools package.

(setup.py and most package setup details are
adapted from the 1313e/e13Tools package)
"""


# %% IMPORTS
# Built-in imports
from codecs import open
import re
from os.path import join as pjoin

# Package imports
from setuptools import find_packages, setup

# %% SETUP DEFINITION
# Get the long description from the README file
pkgname = "uchuutools"
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Read the __version__.py file
with open(f"{pkgname}/__version__.py", 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

scripts = ['convert_ctrees_to_h5', 'convert_halocat_to_h5']
scripts = [pjoin(f"{pkgname}/scripts", s) for s in scripts]

# Setup function declaration
setup(name=pkgname,
      version=version,
      author="Manodeep Sinha",
      author_email='manodeep@gmail.com',
      description=("A collection of utility functions "
                   "created for the Uchuu Project"),
      long_description=long_description,
      url=f"https://{pkgname}.readthedocs.io",
      license='MIT',
      platforms=['Windows', 'Mac OS-X', 'Linux', 'Unix'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Utilities',
          ],
      keywords=['uchuu', 'mergertrees', 'darkmatter',
                'cosmological simulation', 'Rockstar',
                'Consistent-Trees'],
      python_requires='>=3.6, <4',
      packages=find_packages(),
      package_dir={f"{pkgname}": f"{pkgname}"},
      scripts=scripts,
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False,
      )
