"""Features « classiques » pour la baseline : couleur (HSV), texture (HOG, LBP) et morphologie.

Intuition métier :
- HSV : rouille (teinte orangée), efflorescence (blanc, faible saturation), armatures (sombres/métalliques) ;
- HOG / LBP : orientation et micro-texture, une fissure est une structure fine et orientée ;
- morphologie : le black top-hat isole les structures sombres et étroites (fissures) du fond béton.
"""

import numpy as np
from skimage import feature, measure, morphology
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu
from skimage.transform import resize

FEATURE_SIZE = 128
HSV_BINS = 16
LBP_P, LBP_R = 8, 1


def _hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    return hist / max(hist.sum(), 1)


def color_features(img: np.ndarray) -> np.ndarray:
    hsv = rgb2hsv(img)
    return np.concatenate([_hist(hsv[..., c], HSV_BINS, (0.0, 1.0)) for c in range(3)])


def texture_features(gray: np.ndarray) -> np.ndarray:
    hog = feature.hog(
        gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True
    )
    lbp = feature.local_binary_pattern((gray * 255).astype(np.uint8), LBP_P, LBP_R, method="uniform")
    lbp_hist = _hist(lbp, LBP_P + 2, (0, LBP_P + 2))
    return np.concatenate([hog, lbp_hist])


def morphology_features(gray: np.ndarray) -> np.ndarray:
    blackhat = morphology.black_tophat(gray, morphology.disk(5))
    try:
        mask = blackhat > threshold_otsu(blackhat)
    except ValueError:  # image uniforme
        mask = np.zeros_like(blackhat, dtype=bool)
    mask = morphology.remove_small_objects(mask, max_size=14)  # bruit : objets de moins de 15 px

    regions = measure.regionprops(measure.label(mask))
    elongations = [r.axis_major_length / max(r.axis_minor_length, 1.0) for r in regions]
    skeleton_ratio = morphology.skeletonize(mask).mean() if mask.any() else 0.0
    edges = feature.canny(gray, sigma=2.0)

    return np.array(
        [
            gray.mean(),
            gray.std(),
            blackhat.mean(),
            blackhat.std(),
            np.percentile(blackhat, 95),
            mask.mean(),
            len(regions),
            max(elongations, default=0.0),
            skeleton_ratio,
            edges.mean(),
        ]
    )


def extract_features(rgb: np.ndarray) -> np.ndarray:
    """rgb : tableau HxWx3 (uint8 ou float). Retourne un vecteur de taille fixe."""
    img = resize(rgb, (FEATURE_SIZE, FEATURE_SIZE), anti_aliasing=True)  # float dans [0, 1]
    gray = rgb2gray(img)
    return np.concatenate([color_features(img), texture_features(gray), morphology_features(gray)]).astype(
        np.float32
    )
