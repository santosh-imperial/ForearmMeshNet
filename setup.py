from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent
requirements = (this_dir / "requirements.txt").read_text().splitlines()

setup(
    name="forearm-meshnet",
    version="0.1.0",
    description="ForearmMeshNet pipeline for subject-specific forearm meshes.",
    author="ForearmMeshNet Authors",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples", "docs")),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=requirements,
)
"""
Setup configuration for ForearmMeshNet package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forearm_meshnet",
    version="0.1.0",
    author="ForearmMeshNet Team",
    description="A Python package for forearm mesh reconstruction from MRI data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/santosh-imperial/forearm_meshnet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch_geometric>=2.3.0",
        "trimesh>=3.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "SimpleITK>=2.2.0",
        "open3d>=0.16.0",
        "scikit-image>=0.19.0",
        "pymeshfix>=0.16.0",
        "shapely>=2.0.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "viz": ["pyvista>=0.38.0"],
    }
)
