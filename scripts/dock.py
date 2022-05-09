"""
Requires running in a conda environment

conda create --name docking python=3.8
conda install -c omnia openmm
conda install -c conda-forge pdbfixer
conda install -c conda-forge deepchem
conda install -c rdkit rdkit
pip install vina
conda install -c anaconda networkx
"""

import deepchem as dc
import numpy as np

vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)

protein_file = "../data/CBDAS.pdb"

ligand_file = "../data/substrate.sdf"

centroid = np.array([-10.581, -1.236, -6.007])  # CA of Y483 (active site)
box_dims = np.array([20.0, 20.0, 20.0])

poses, scores = vpg.generate_poses(
    (protein_file, ligand_file),
    centroid=centroid,
    box_dims=box_dims,
    exhaustiveness=1,
    num_modes=20,
    out_dir="../data/vina",
    generate_scores=True,
)
