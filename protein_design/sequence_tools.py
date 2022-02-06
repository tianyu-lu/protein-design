from pathlib import Path
from typing import List

from Bio import SeqIO
import numpy as np


def get_seqs(fp: Path, format: str = "fasta") -> List[str]:
    """
    Read a file of sequences and returns a list of sequences
    """
    return [record.seq for record in SeqIO.parse(fp, format)]


def seq_to_onehot(fp: Path, format: str = "fasta") -> np.ndarray:
    """
    Convert a file of sequences into a one-hot encoded matrix
    """
    seqs = get_seqs(fp, format=format)
    
    return np.zeros((len(seqs), 200))
