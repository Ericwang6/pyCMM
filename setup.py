from setuptools import setup, find_packages

setup(
    name="pyCMM",
    version="0.1.0",
    packages=find_packages(where="cmm"),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "parmed",
        "openmm",
        "pytest",
        "scipy",
        "torch",
        "torch-scatter",
        "iodata-qc"
    ],
    author="Eric Wang, Joseph Heindel, Aalim Abdullah",
    author_email="ericwangyz@berkeley.edu,heindelj@lbl.gov,aalimabdullah@berkeley.edu",
    description="A package for differentiable multipolar polarizable force fields",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ericwang6/pyCMM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)