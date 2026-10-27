"""Submitted deterministic entropy precision and stride-two map serialization."""
import numpy as np


def deterministic_entropy(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    entropy=-(p*np.log(np.clip(p,1e-8,1.))).sum(1)
    return entropy[:,::2,::2][:,:128,:128].astype(np.float32).astype(np.float64)
