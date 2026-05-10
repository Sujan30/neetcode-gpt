import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)

        for i in range(len(weights)):
            if i == 0:
                z = np.array(x) @ np.array(weights[i]) + np.array(biases[i])
                a1 = np.maximum(0,z)
                output = a1
            else:
                output = a1 @ np.array(weights[i]) + np.array(biases[i])
                a1 = np.maximum(output, 0)
        
        return output





        pass
