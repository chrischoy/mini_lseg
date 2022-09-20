#!/usr/bin/env python
#
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from setuptools import setup

with open("requirements.txt", "r") as f:
    dependencies = [dep for dep in f.read().splitlines() if dep]

setup(
    name="mini_lseg",
    version="0.0.2",
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
