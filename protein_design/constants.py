import torch

AA = list("-ACDEFGHIKLMNPQRSTVWY")

AA_IDX = {AA[i]: i for i in range(len(AA))}

IDX_AA = {i: AA[i].upper() for i in range(len(AA))}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
