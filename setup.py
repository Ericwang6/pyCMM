from setuptools import setup, find_packages

setup(
    name="pyCMM",
    version="0.1.0",
    packages=find_packages(where="cmm"),
    install_requires=[
        "openmm==8.2.0",
        "pytest==8.3.4",
        "scipy==1.15.1",
        "torch==2.6.0",
        "torch-scatter==2.1.2",
        "torchaudio==2.6.0",
        "torchvision==0.21.0"
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
    python_requires=">=3.6",
)