from setuptools import setup, find_packages

setup(
    name="pyCMM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openmm==8.2.0",
        "pytest",
        "scipy",
        "qc-iodata",
        "tables"
    ],
    extras_require={
        "torch": ["torch>=2.0.0", "torchvision", "torchaudio", "torch-scatter", "tensordict"],
    },
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
    python_requires=">=3.12",
)