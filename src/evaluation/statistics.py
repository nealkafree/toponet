import numpy as np
from libpysal.weights import lat2W
from esda.moran import Moran


def moran_i(x: np.ndarray, spatial_grid_width: int) -> float:
    """
    Calculates Moran's I spatial autocorrelation metric.
    """
    x = x.reshape((spatial_grid_width, spatial_grid_width))

    # Create the matrix of weights
    w = lat2W(spatial_grid_width, spatial_grid_width)

    # Return the metric
    return Moran(x, w).I
