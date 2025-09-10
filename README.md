# PyCMM - Completely Multipolar Model as a General Framework for Many-Body Interactions

> [!NOTE]
> This project is being actively developed and the APIs are subject to change.

A Python implementation of CMM model. 

## Papers
+ Heindel, Joseph P., Selim Sami, and Teresa Head-Gordon. "Completely multipolar model as a general framework for many-body interactions as illustrated for water." *Journal of Chemical Theory and Computation* 20.19 (2024): 8594-8608. [![DOI: 10.1039/D4DD00357H](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.4c00812-blue)](https://doi.org/10.1021/acs.jctc.4c00812)
+ Heindel, Joseph P., et al. "Completely Multipolar Model for Many-Body Water–Ion and Ion–Ion Interactions." *The Journal of Physical Chemistry Letters* 16.4 (2025): 975-984. [![DOI: 10.1021/acs.jpclett.4c02940](https://img.shields.io/badge/DOI-10.1021%2Facs.jpclett.4c02940-blue)](https://doi.org/10.1021/acs.jpclett.4c02940)

## Installation

Please first install the following dependencies via conda or pip

+ numpy
+ pandas
+ matplotlib
+ scipy
+ [openmm](https://openmm.org/)
+ [parmed](https://github.com/ParmEd/ParmEd)
+ [torch](https://pytorch.org/)
+ [torch-scatter](https://github.com/rusty1s/pytorch_scatter)

In order to use the parameterization code, the following external programs are needed and their executables should be added to the `PATH` environment variables.

+ Q-Chem (for quantum chemistry calculations and EDA calculations)
+ GDMA (for deriving atomic multipoles in global coordinate system)
+ XTB (for running short dynamics for sampling monomer geometries)
+ POLEDIT (part of TINKER software, for generating atomic multipoles in local coordinate system)
+ Multiwfn (for computing electrostatic potential surface)

## Usage

### Molecular Dynamics with CMM

PyCMM implements the CMM model and it takes coordinates and box and outputs energies. The forces can be easily obtained via automatic gradient with pytorch. Therefore, in principle the package can be interfaced with any molecular dynamics enegine. But in practice, the developers now only offers interfaces with ASE and the scripts are in: `scripts/water_md.py`

### Parameter Optimization

We are now building some workflow to automate the process of deriving new parameters beyond water. A script for fit nonbonded paramters for methanol against EDA data can be found in `scripts/methanol.ipynb`

There are also some codes (`cmm/develop/workflow.py`) for streamlining the calculation of quantum chemsitry data for monomers (atomic multipoles, molecular polarizabilities, dipole surface data, electrostatic potential surface data) 

## Code structure

The repository contains a lot of legacy codes and efforts are undergoing to clean up. Now several important components are:

+ `cmm/ffxml`: Parse force field parameters in XML format and generate parametrizers to assign parameters to given topology 
+ `cmm/topology.py`: Definition of topolgy data
+ `cmm/system.py`: CMM model implementation to run MD
+ `cmm/batched_system.py`: CMM model implementation to do batch computation, useful in parameter optimization
+ `cmm/develop`: Util functions for parameter optimization 
