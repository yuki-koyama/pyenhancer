import numpy as np
from typing import List
from skimage.color import rgb2hsv, hsv2rgb

GAMMA_VALUE = 2.2


def _convert_to_linear(image: np.ndarray) -> np.ndarray:
    return np.power(image, GAMMA_VALUE)


def _convert_from_linear(image: np.ndarray) -> np.ndarray:
    return np.power(image, 1.0 / GAMMA_VALUE)


def _apply_brightness_effect(image: np.ndarray, brightness: float) -> np.ndarray:
    BRIGHTNESS_SCALE = 1.5

    return np.power(image, 1.0 / (1.0 + BRIGHTNESS_SCALE * brightness))


def _apply_contrast_effect(image: np.ndarray, constrast: float) -> np.ndarray:
    PI_4 = 3.14159265358979 * 0.25

    contrast_coeff = np.tan((constrast + 1.0) * PI_4)

    image = _convert_from_linear(image)
    image = np.maximum(contrast_coeff * (image - 0.5) + 0.5, 0.0)
    image = _convert_to_linear(image)

    return image


def _apply_saturation_effect(image: np.ndarray, saturation: float) -> np.ndarray:
    image = rgb2hsv(image)
    image[:, :, 1] = np.clip(image[:, :, 1] * (saturation + 1.0), 0.0, 1.0)
    return hsv2rgb(image)


def _apply_lift_gamma_gain_effect(
    image: np.ndarray, lift: np.ndarray, gamma: np.ndarray, gain: np.ndarray
) -> np.ndarray:
    LIFT_SCALE = 0.25

    image = np.clip((image - 1.0) * (1.0 - LIFT_SCALE * (lift - 0.5)) + 1.0, 0.0, 1.0)
    image = image * (gain + 0.5)
    image = np.clip(np.power(image, 1.0 / (gamma + 0.5)), 0.0, 1.0)

    return image


def get_labels() -> List[str]:
    """Get labels of parameters."""
    return [
        "Brightness",
        "Contrast",
        "Saturation",
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
    assert params.shape == (12,)

    image = _convert_to_linear(image)
    image = _apply_lift_gamma_gain_effect(image, params[3:6], params[6:9], params[9:12])
    image = _apply_brightness_effect(image, params[0] - 0.5)
    image = _apply_contrast_effect(image, params[1] - 0.5)
    image = _apply_saturation_effect(image, params[2] - 0.5)
    image = _convert_from_linear(np.clip(image, 0.0, 1.0))

    return image
