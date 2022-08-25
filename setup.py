#!/usr/bin/env python
import os
from setuptools import setup, find_packages

setup(
    name="mini_lseg",
    version="0.0.1",
    description="Lightweight inference only language segmentation inference library",
    author="Chris Choy",
    author_email="cchoy@nvidia.com",
    license="MIT",
    packages=['MiniLseg'],
    package_dir={"MiniLseg": "./MiniLseg"},
    include_package_data=True,
    zip_safe=False,
)
