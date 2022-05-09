from typing import NewType, List, Optional, Union, Tuple
from pathlib import Path
import subprocess
import itertools

import numpy as np
import py3Dmol
import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser, PDBIO, Selection, Structure, Chain, Residue
from Bio.PDB.PDBExceptions import PDBConstructionException


parser = PDBParser()
io = PDBIO()
Resid = NewType("Resid", Tuple[int, str, Tuple[str, int, str]])


def download_pdb(pdb_code: str) -> Structure:
    """Download a pdb file from RCSB and load it as a Biopython Structure

    Parameters
    ----------
    pdb_code
        PDB code

    Returns
    -------
        Bio.PDB.Structure
    """
    pdb_code = pdb_code.upper()
    subprocess.run(
        ["wget", f"https://files.rcsb.org/download/{pdb_code}.pdb"], check=True
    )
    structure = parser.get_structure("s", f"{pdb_code}.pdb")
    return structure


def remove_hetero(structure: Structure, subset: Optional[str] = None) -> None:
    """Removes all water molecules in {structure}

    Parameters
    ----------
    structure
        Bio.PDB.Structure
    subset
        Comma-delimited string of the residues to remove, e.g. "HOH,SO4" to remove all waters and SO4 molecules
    """
    residues_to_remove = []
    for residue in Selection.unfold_entities(structure, "R"):
        resname = residue.get_resname()
        if subset is not None:
            if resname in subset.split(","):
                residues_to_remove.append(residue)
        else:
            if seq1(resname) == "X":
                residues_to_remove.append(residue)
    for r in residues_to_remove:
        r.get_parent().detach_child(r.get_id())


def get_xyz(entity: List) -> np.ndarray:
    """Get xyz coordinates of all atoms in the entity list

    Parameters
    ----------
    entity
        Any one of Bio.PDB.(Structure, Model, Chain, Residue, Atom)

    Returns
    -------
        N x 3 numpy array of xyz coordinates, where N is the number of atoms
    """
    coords = []
    for atom in Selection.unfold_entities(entity, "A"):
        coords.append(atom.coord)
    return np.array(coords)


def get_res_data(residue: Residue) -> Resid:
    """Get the model, chain and residue id of the input {residue}

    Parameters
    ----------
    residue
        Bio.PDB.Residue

    Returns
    -------
    Resid tuple containing three elements:
        model:
            Model number
        chain:
            Single letter chain
        resid:
            Full residue id: (heteroatom flag, numerical id, insertion code)
    """
    _, model, chain, full_resid = residue.get_full_id()
    return (model, chain, full_resid)


def get_interface(c1: Chain, c2: Chain, cutoff: float = 5.0) -> List[Resid]:
    """Given two chains, find the residues where at least one pair of atoms is
    within {cutoff} Angstroms of each other

    Parameters
    ----------
    c1
        Chain 1
    c2
        Chain 2
    cutoff, optional
        Interatomic distance in Angstroms to consider a pair of residues as
        "in contact" and thus belonging to the interface, by default 5.0

    Returns
    -------
        List of residue identifiers used to subset the structure containing chains 1 and 2
    """
    res1_interface = []
    res2_interface = []
    for res1, res2 in itertools.product(c1.get_residues(), c2.get_residues()):
        closest_dist = 1e6
        res1_coords, res2_coords = get_xyz(res1), get_xyz(res2)
        for coord in res1_coords:
            dists = np.linalg.norm(coord.reshape(1, 3) - res2_coords, axis=1)
            curr_dist = np.min(dists)
            if curr_dist < closest_dist:
                closest_dist = curr_dist
        if closest_dist <= cutoff:
            res1_interface.append(get_res_data(res1))
            res2_interface.append(get_res_data(res2))
    return list(set(res1_interface).union(set(res2_interface)))


def subset_structure(structure, subset_list) -> None:
    """Subsets the {structure} to keep only those residues in {subset_list}.
    Note: this function modifies the input {structure} in place.

    Parameters
    ----------
    structure
        Bio.PDB.Structure
    subset_list
        List of residue identifiers to keep, obtained from e.g. get_interface()
    """
    to_remove = []
    for residue in Selection.unfold_entities(structure, "R"):
        model_id = residue.get_parent().get_parent().get_id()
        chain_id = residue.get_parent().get_id()
        residue_id = residue.get_id()
        full_id = (model_id, chain_id, residue_id)
        if full_id not in subset_list:
            to_remove.append(residue)
    for residue in to_remove:
        try:
            residue.get_parent().detach_child(residue.get_id())
        except AttributeError:
            pass


def show_from_pdbid(pdbid: str) -> None:
    """Show an interactive structure of {pdbid} using py3Dmol

    Parameters
    ----------
    pdbid
        PDB code
    """
    view = py3Dmol.view(query=f"pdb:{pdbid}", options={"doAssembly": True})
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.show()


def show_from_file(pdb_file: Union[Path, str], chains: str = "A") -> None:
    """Show an interactive structure from chains {chains} from file {pdb_file}

    Parameters
    ----------
    pdb_file
        Path to pdb file
    chains
        String of chains to show, by default "A"
    """
    with open(pdb_file) as fp:
        interface_pdb = "".join(
            line
            for line in fp.readlines()
            if line.startswith("ATOM") and line[21] in chains
        )

    view = py3Dmol.view()
    view.addModelsAsFrames(interface_pdb)
    for chain in chains:
        view.setStyle({"chain": chain}, {"cartoon": {"color": "spectrum"}, "stick": {}})
    view.show()

    view.setHoverable(
        {},
        True,
        """function(atom,viewer,event,container) {
                    if(!atom.label) {
                        atom.label = viewer.addLabel(atom.resn+":"+atom.atom,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                    }}""",
        """function(atom,viewer) { 
                    if(atom.label) {
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }
                    }""",
    )


def structure_to_sequence(structure: Structure) -> str:
    """Convert a Bio.PDB.Structure into a sequence.

    Parameters
    ----------
    structure
        Bio.PDB.Structure

    Returns
    -------
        Sequence
    """
    seq = []
    for residue in Selection.unfold_entities(structure, "R"):
        aa = seq1(residue.get_resname())
        seq.append(aa)
    return "".join(seq)


if __name__ == "__main__":
    entries = []
    data_dir = Path("data/sabdab/2022-04-03/")
    summary_fp = data_dir / "sabdab_nano_summary_all.tsv"
    df = pd.read_csv(summary_fp, sep="\t")
    df = df.loc[df["pdb"] != "6iyn"]
    df.dropna(subset=["Hchain"], inplace=True)
    pdb_codes = df["pdb"].to_list()
    nanobody_chains = df["Hchain"].to_list()
    structures_dir = data_dir / "imgt"
    for pdb, chain in tqdm(zip(pdb_codes, nanobody_chains)):
        pdb_fp = structures_dir / f"{pdb}.pdb"
        structure = parser.get_structure(pdb, pdb_fp)
        remove_hetero(structure)
        try:
            seq = structure_to_sequence(structure[0][chain])
            entries.append(f">{pdb}_{chain}\n{seq}")
        except KeyError:
            print(pdb, chain)
            continue
        except PDBConstructionException:
            print(pdb, chain)
            continue
    with open(data_dir / "vhh.fasta", "w") as fp:
        fp.write("\n".join(entries))
