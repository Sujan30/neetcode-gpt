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


        # the forward pass
        
        
        z1 = np.array(W1) @ np.array(x)  + b1
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ np.array(W2).T + np.array(b2)
        loss = np.round(np.mean((z2 - y_true)**2),4)

        #back propogation, 2nd layer

        dL_dpred = 2/(len(z2)) * (z2 - y_true)
        dw2 = np.outer(dL_dpred,a1)
        db2 = dL_dpred

        # back propogation, 1st layer
        
        dL_da1 = np.array(W2).T @ dL_dpred
        dL_z1 = dL_da1 * (z1>0)
        db1 = dL_z1 
        dw1 = np.outer(dL_z1,x)+0.0




        return{
            'loss': loss,
            'dW2': np.round(dw2+0.0, 4).tolist(),
            'db2': np.round(db2+0.0, 4).tolist(),
            'db1': np.round(db1+0.0, 4).tolist(),
            'dW1': np.round(dw1+0.0, 4).tolist()

        }












        pass
