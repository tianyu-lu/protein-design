from pathlib import Path
from typing import List, Optional, Tuple
from logging import Logger

from Bio import SeqIO
import numpy as np
from tqdm import tqdm
from abnumber import Chain
from abnumber.exceptions import ChainParseError

from protein_design.constants import AA, AA_IDX, IDX_AA


logger = Logger("protein_design")


def read_fasta(fname: Path, format: str = "fasta") -> List[str]:
    """Read a file of sequences and returns a list of sequences

    Parameters
    ----------
    fname : Path
        Path to sequence file
    format : str, optional
        Format of the sequence file, by default "fasta"

    Returns
    -------
    List[str]
        List of sequences
    """
    return [record.seq for record in SeqIO.parse(fname, format)]


def write_fasta(
    fname: Path, seqs: List[str], headers: Optional[List[str]] = None
) -> None:
    """Write sequences to a fasta file

    Parameters
    ----------
    fname
        Full path of file name to save as
    seqs
        List of sequences
    headers, optional
        List of headers, by default None, in which case sequences will be labeled as
        integers starting from 0
    """
    if headers is None:
        headers = [f">{i}" for i in range(len(seqs))]
    records = []
    for i, seq in enumerate(seqs):
        records.append(headers[i])
        records.append(seq)
    with open(fname, "w") as fp:
        fp.write("\n".join(records))


def seq_to_onehot(seq: str, max_len: Optional[int] = None) -> np.ndarray:
    """Convert a sequence to a 2D one-hot encoded matrix

    Parameters
    ----------
    seq : str
        Single letter amino acid sequence, possibly containing gap '-' characters
    max_len : Optional[int], optional
        If provided, the returned one-hot encoded matrix will have shape (max_len, 21)
        Sequences longer than max_len are truncated at the tail (C-terminus)
        Gap characters are added to sequences shorter than max_len at the tail (C-terminus), by default None

    Returns
    -------
    np.ndarray
        Array of shape (length of seq, 21) or (max_len, 21) if max_len is provided.
    """
    seq = seq.upper()

    if max_len is None:
        max_len = len(seq)
    elif max_len > len(seq):
        seq = seq + "-" * (max_len - len(seq))
    elif max_len < len(seq):
        seq = seq[:max_len]

    onehot = np.zeros((max_len, len(AA)))
    for i in range(max_len):
        onehot[i, AA_IDX[seq[i]]] = 1

    return onehot


def seqs_to_onehot(seqs: List[str], flatten: bool = False) -> np.ndarray:
    """Converts a list of amino acid sequences to a 3D one-hot encoding matrix

    Parameters
    ----------
    seqs : list[str]
        List of amino acid sequences
    flatten : bool
        If True, flattens each one-hot encoded sequence

    Returns
    -------
    np.ndarray
        Array of shape (numer of sequences, maximum length of sequences, 21)
    """
    max_len = max(len(seq) for seq in seqs)

    onehots = []
    for seq in seqs:
        onehots.append(seq_to_onehot(seq, max_len=max_len))

    if flatten:
        result = np.array(onehots)
        N, L, D = result.shape
        return result.reshape(N, L * D)
    else:
        return np.array(onehots)


def integer_to_seqs(mat: np.ndarray) -> List[str]:
    """Convert an integer encoded list of sequences back into letters

    Parameters
    ----------
    mat
        Integer encoded sequences

    Returns
    -------
        List of amino acid strings
    """
    seqs = []
    for row in mat:
        seq = "".join(IDX_AA[i] for i in row)
        seqs.append(seq)
    return seqs


def seqs_to_chain(seqs: List[str], scheme: str = "imgt") -> List[Chain]:
    """Converts a list of antibody sequences to a list of abnumber.Chain objects
    Logs sequences which could not be processed by abnumber

    Parameters
    ----------
    seqs
        List of antibody sequences
    scheme, optional
        Antibody number scheme to use, by default "imgt". One of: imgt, chothia, kabat, aho

    Returns
    -------
        List of abnumber.Chain objects
    """
    chains = []
    for seq in tqdm(seqs):
        try:
            chains.append(Chain(seq, scheme=scheme))
        except (NotImplementedError, ChainParseError) as err:
            logger.warning(err)
    return chains


def align_antibody_seqs(
    seqs: List[str], scheme: str = "imgt"
) -> Tuple[List[str], List[str]]:
    """Aligns a list of antibody sequences

    Parameters
    ----------
    seqs
        List of sequences
    scheme, optional
        Antibody number scheme to use, by default "imgt". One of: imgt, chothia, kabat, aho

    Returns
    -------
        Tuple of positions and aligned sequences
    """
    chains = seqs_to_chain(seqs, scheme=scheme)
    alignment = chains[0].align(*chains[1:])
    positions, aligned_seqs = [], []
    for pos, (aas) in alignment:
        positions.append(pos.format())
        aligned_seqs.append(list(aas))
    aligned_seqs = np.array(aligned_seqs)
    aligned_seqs = ["".join(aligned_seqs[:, i]) for i in range(len(aligned_seqs))]
    return positions, aligned_seqs
