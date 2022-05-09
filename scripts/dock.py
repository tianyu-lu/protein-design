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
from pathlib import Path
from typing import Optional
import deepchem as dc
import numpy as np
import typer


vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)


def dock(protein: Path, ligand: Path, out: Path, centroid: Optional[str] = None, box: Optional[str] = None, exhaustiveness: int = 1, num_modes: int = 20):
    """Dock a ligand to a protein using the deepchem interface to AutoDock Vina

    Parameters
    ----------
    protein
        File path of target protein structure in .pdb format (receptor)
    ligand
        File path of ligand to dock in .sdf format
    out
        Output folder for docked structures and intermediate files
    centroid, optional
        Center of search space for docking, by default None
    box, optional
        Dimensions of a cubic box defining the search space, by default None
    exhaustiveness, optional
        Intensity of the search for docked poses, by default 1
    num_modes, optional
        Number of docked structures, by default 20
    """
    if centroid is not None:
        x, y, z = tuple(centroid.split(","))
        centroid = np.array([float(x), float(y), float(z)])
    else:
        centroid = np.array([-10.581, -1.236, -6.007])  # CA of Y483 (active site)

    if box is not None:
        x, y, z = tuple(box.split(","))
        box = np.array([float(x), float(y), float(z)])
    else:
        box = np.array([20.0, 20.0, 20.0])

    poses, scores = vpg.generate_poses(
        (str(protein), str(ligand)),
        centroid=centroid,
        box_dims=box,
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
        out_dir=str(out),
        generate_scores=True,
    )


if __name__ == '__main__':
    typer.run(dock)
