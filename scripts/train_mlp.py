import numpy as np
import pandas as pd

from protein_design.splitter import random_split
from protein_design.sequence import seqs_to_onehot
from protein_design.discriminative import MLP
from protein_design.trainer import train

fname = "/home/tianyulu/Documents/csberg/2Q8A_AFinalBindings_Process_1_Of_32.txt"

df = pd.read_csv(fname, sep='\t', skiprows=1)

df_filtered = df.loc[df["Best?"] == True]
df_filtered = df_filtered.drop_duplicates("Slide")

seqs = df_filtered["Slide"].to_list()
X = seqs_to_onehot(seqs, flatten=True)

y = energies1 = df_filtered["Energy"].to_numpy()

y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

X_train, y_train, X_test, y_test = random_split(X, y=y)

model = MLP(data_dim=11*21, hid_dim=20)

batch_size = 16
epochs = 1
print(X_train.shape)
model = train(model, X_train, X_test, "mlp.pt", y_train=y_train, y_test=y_test, 
              steps=int(len(df_filtered) / batch_size * epochs))