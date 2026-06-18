from __future__ import absolute_import #print_function
from setuptools import setup,find_packages
import os

setup(name='unicore',
      version=0.1,
      description='unicore packages',
      author='Zhiguang Fan',
      license='MIT',
      packages=find_packages(),

      #package_dir={"": "KGDiffEcConf"},  # 告诉setuptools在src目录下找包

      zip_safe=False,
      include_package_data=True
      )