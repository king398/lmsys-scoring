from autofaiss import build_index
import numpy as np
embeddings = np.load('/home/mithil/PycharmProjects/lmsys-scoring/data/open_hermes_embeddings.npy')

index, index_infos = build_index(embeddings, save_on_disk=True)


