#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    dependencies = [dep for dep in f.read().splitlines() if dep]

setup(
    name="mini_lseg",
    version="0.0.1",
    description="Lightweight inference only language segmentation inference library",
    author="Chris Choy",
    author_email="cchoy@nvidia.com",
    license="MIT",
    packages=["MiniLseg"],
    package_dir={"MiniLseg": "./MiniLseg"},
    install_requires=dependencies,
    include_package_data=True,
    zip_safe=False,
)
