#!/usr/bin/python3

from setuptools import setup, find_namespace_packages
import os


datadir = os.path.join("data")
datafiles = [
    (d, [os.path.join(d, f) for f in files]) for d, folders, files in os.walk(datadir)
]

setup(
    name="ifmdock",
    version="1.0.0",
    description="Flow matching for rigid-pocket docking",
    license="MIT",
    packages=find_namespace_packages(include=["ifmdock", "ifmdock.*"]),
    zip_safe=False,
    data_files=datafiles,
)
