from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.absolute()

with open(here / "README.rst", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hubmap-fastq-utils",
    version="0.2.5",
    description="FASTQ utility functions for HuBMAP computational pipelines",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/hubmapconsortium/antibodies-tsv-util",
    author="Penny Cuda",
    author_email="pcuda@andrew.cmu.edu",
    license="GPLv3",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="antibodies",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)
