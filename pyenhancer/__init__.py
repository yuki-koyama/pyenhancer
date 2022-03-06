import numpy as np
from typing import List

GAMMA_VALUE = 2.2


def _convert_to_linear(image: np.ndarray) -> np.ndarray:
    return np.power(image, GAMMA_VALUE)


def _convert_from_linear(image: np.ndarray) -> np.ndarray:
    return np.power(image, 1.0 / GAMMA_VALUE)


def _apply_lift_gamma_gain_effect(image: np.ndarray, lift: np.ndarray,
                                  gamma: np.ndarray,
                                  gain: np.ndarray) -> np.ndarray:

    image = np.clip((image - 1.0) * (2.0 - (lift + 0.5)) + 1.0, 0.0, 1.0)
    image = image * (gain + 0.5)
    image = np.clip(np.power(image, 1.0 / (gamma + 0.5)), 0.0, 1.0)

    return image


def get_labels() -> List[str]:
    """Get labels of parameters."""
    return [
        "Lift (R)",
        "Lift (G)",
        "Lift (B)",
        "Gamma (R)",
        "Gamma (G)",
        "Gamma (B)",
        "Gain (R)",
        "Gain (G)",
        "Gain (B)",
    ]


def enhance(image: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Enhance colors of an image using specified parameters.

    Parameters
    ----------
    image : ndarray of shape (height, width, 3)
        Target image.
    params : ndarray of shape (9, )
        Parameters.

    Returns
    -------
    image : ndarray of shape (height, width, 3)
        Enhanced image.
    """
    assert image.dtype == np.float64
    assert params.shape == (9, )

    image = _convert_to_linear(image)
    image = _apply_lift_gamma_gain_effect(image, params[0:3], params[3:6],
                                          params[6:9])
    image = _convert_from_linear(image)

    return image
