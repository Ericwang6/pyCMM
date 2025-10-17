# PyCMM - Completely Multipolar Model as a General Framework for Many-Body Interactions

> [!NOTE]
> This project is being actively developed and the APIs are subject to change.

A Python implementation of CMM model. 

## Papers
+ Heindel, Joseph P., Selim Sami, and Teresa Head-Gordon. "Completely multipolar model as a general framework for many-body interactions as illustrated for water." *Journal of Chemical Theory and Computation* 20.19 (2024): 8594-8608. [![DOI: 10.1039/D4DD00357H](https://img.shields.io/badge/DOI-10.1021%2Facs.jctc.4c00812-blue)](https://doi.org/10.1021/acs.jctc.4c00812)
+ Heindel, Joseph P., et al. "Completely Multipolar Model for Many-Body Water–Ion and Ion–Ion Interactions." *The Journal of Physical Chemistry Letters* 16.4 (2025): 975-984. [![DOI: 10.1021/acs.jpclett.4c02940](https://img.shields.io/badge/DOI-10.1021%2Facs.jpclett.4c02940-blue)](https://doi.org/10.1021/acs.jpclett.4c02940)

## Installation

Please first install the following dependencies via conda/mamba/pip

+ numpy
+ pandas
+ matplotlib
+ scipy
+ [openmm](https://openmm.org/)
+ [parmed](https://github.com/ParmEd/ParmEd)
+ [torch](https://pytorch.org/)
+ [torch-scatter](https://github.com/rusty1s/pytorch_scatter)
+ ase

For example, this code snippet provides how to setup the environment via `conda` and `pip`

```bash
mamba create -n cmm python=3.12
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
# The numpy/scipy/matplotlib will be installed automatically with ASE
pip install ase pandas
mamba install openmm==8.2.0 -c conda-forge
```

If you encounter any errors, try to remove caches and run the code again:
```bash
rm -rf ~/.triton
```


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

### Customized OPs and Code Profiling
Run profiling with environment variable `CMM_PROFILE=1`. For example,
```bash
cd scripts/
CMM_PROFILE=1 python water_md_nve.py
```

To use customized torch operators for CMM, please install [torchff-lib](https://github.com/Ericwang6/torchff-lib) first and set `use_customized_ops=True` during `ff.parameterize`:

```python
ff = ForceFieldXML(ff_path, device='cuda')
pdb = app.PDBFile(pdb_path)
top = Topology.fromOpenmm(pdb.topology, device)
system = ff.parametrize(
    top, use_fd_morse=True, use_polarization=True, polarization_tolerance=1e-5, 
    use_hardness_change=False, use_lr_dispersion=True, cutoff_sr=9.0, use_switch=True, 
    use_customized_ops=True
)
```

## Code structure

The repository contains a lot of legacy codes and efforts are undergoing to clean up. Now several important components are:

+ `cmm/ffxml`: Parse force field parameters in XML format and generate parametrizers to assign parameters to given topology 
+ `cmm/topology.py`: Definition of topolgy data
+ `cmm/system.py`: CMM model implementation to run MD
+ `cmm/batched_system.py`: CMM model implementation to do batch computation, useful in parameter optimization
+ `cmm/develop`: Util functions for parameter optimization 
