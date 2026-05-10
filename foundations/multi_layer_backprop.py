import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        

        z = np.array(W1) @ np.array(x) + np.array(b1)
        a1 = np.maximum(0.0, z)
        z2 = np.array(W2) @ np.array(a1) + np.array(b2)

        pred = z2

        loss = np.mean((pred - y_true)**2)

        dL_dPred = 2/(len(y_true)) * (pred-y_true)
        
        dL_da1 = np.array(W2).T @ dL_dPred
        dL_dz1 = dL_da1 * (z>0)


        return {
            'loss': round(loss,4),
            'dW2': (np.outer(dL_dPred, a1).round(4)+0.0).tolist(),
            'db2': np.round(dL_dPred,4),
            'db1': np.round(1 * dL_dz1,4),
            'dW1': (np.outer(dL_dz1, x).round(4)+0.0).tolist()
        }


