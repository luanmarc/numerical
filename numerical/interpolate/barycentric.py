"""
Authors: Luiz Gustavo Mugnaini Anselmo (nUSP: 11809746)
         Victor Manuel Dias Saliba     (nUSP: 11807702)
         Luan Marc Suquet Camargo      (nUSP: 11809090)

Computacao III (CCM): EP 4 Barycentric Interpolation
"""
import numpy as np

class Barycentric:
    def __init__(self, knots: list[float], end_vals: tuple[float, float]):
        self.knots = knots
        self.end_vals = end_vals
        self.weights = self.find_weights()

    def find_weights(self) -> np.ndarray:
        """Finds the weights of the Barycentric polynomial given the knots"""
        ws = np.ones(len(self.knots), dtype=float)
        for j in range(len(ws)):
            for (k, x) in enumerate(self.knots):
                if j != k:
                    ws[j] *= 1 / (self.knots[j] - x)
        return ws
