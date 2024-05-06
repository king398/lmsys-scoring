import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train.csv")
df['label'] = np.argmax(df[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)
df['fold'] = -1

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for fold, (train_index, test_index) in enumerate(skf.split(df, df['label'])):
    df.loc[test_index, 'fold'] = fold
df.to_csv("/home/mithil/PycharmProjects/lmsys-scoring/data/train.csv", index=False)
