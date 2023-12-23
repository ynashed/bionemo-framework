# DIPS Dataset

Database of Interacting Protein Structures ([DIPS](https://github.com/drorlab/DIPS)) {cite:p}`townshend2019end` is constructed to address limitations in existing structural biology tasks, particularly in predicting protein interactions. The dataset is two orders of magnitude larger than previous datasets, aiming to explore whether performance can be enhanced by utilizing large repositories of tangentially related structural data.

DIPS is built by mining the Protein Data Bank (PDB) for pairs of interacting proteins, yielding a dataset of 42,826 binary complexes. To ensure data quality, complexes are selected based on specific criteria, including a buried surface area of ≥ 500 Å2, solved using X-ray crystallography or cryo-electron microscopy at better than 3.5 Å resolution, containing protein chains longer than 50 amino acids, and being the first model in a structure.

Sequence-based pruning is applied to prevent cross-contamination between the DIPS and Docking Benchmark 5 (DB5) datasets. Any complex with individual proteins having over 30% sequence identity with any protein in DB5 is excluded. This pruning process, along with sequence-level exclusion, results in a dataset over two orders of magnitude larger than DB5.