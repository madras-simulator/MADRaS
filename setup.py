"""Setup File for MADRaS sim."""
from setuptools import setup

setup(name='MADRaS',
      version='0.1',
      description='Multi Agent Driving Simulator',
      install_requires=['gym',
                        'pyyaml',
                        'pTable',
                        'matplotlib',
                        'mpi4py',
                        'tensorflow'])
