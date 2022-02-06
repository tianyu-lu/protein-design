from pathlib import Path
from typing import List, Optional

from Bio import SeqIO
import numpy as np

from protein_design.constants import AA, AA_IDX, IDX_AA


def get_seqs(fname: Path, format: str = "fasta") -> List[str]:
    """
    Read a file of sequences and returns a list of sequences

    Parameters
    ----------
    fname:
        Path to sequence file
    format:
        Format of the sequence file, e.g. "fasta"

    Returns
    -------
    List of sequences
    """
    return [record.seq for record in SeqIO.parse(fname, format)]


def seq_to_onehot(seq: str, max_len: Optional[int] = None) -> np.ndarray:
    """
    Convert a sequence to a one-hot encoded matrix

    Parameters
    ----------
    seq:
        Single letter amino acid sequence, possibly containing gap '-' characters
    max_len:
        If provided, the returned one-hot encoded matrix will have shape (max_len, 21)
        Sequences longer than max_len are truncated at the tail (C-terminus)
        Gap characters are added to sequences shorter than max_len at the tail (C-terminus)

    Returns
    -------
    np.ndarray of shape (length of seq, 21) or (max_len, 21) if max_len is provided.
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


def seqs_to_onehot(fname: Path, format: str = "fasta") -> np.ndarray:
    """
    Convert a file of sequences into a one-hot encoded matrix

    Returns
    -------
    np.ndarray of shape (numer of sequences, maximum length of sequences, 21)
    """
    seqs = get_seqs(fname, format=format)

    max_len = max(len(seq) for seq in seqs)

    onehots = []
    for seq in seqs:
        onehots.append(seq_to_onehot(seq, max_len=max_len))

    return np.array(onehots)
