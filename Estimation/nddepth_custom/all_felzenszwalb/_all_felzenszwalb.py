#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

from ._allfelzenszwalb_cy import _felzenszwalb8_all_cython, _felzenszwalb4_all_cython


def all_felzenszwalb(down_cost, right_cost, dright_cost, uright_cost, height, width, scale=1, sigma=0.8, min_size=20):
    """Computes Felsenszwalb's efficient graph based image segmentation.

    Produces an oversegmentation of a multichannel (i.e. RGB) image
    using a fast, minimum spanning tree based clustering on the image grid.
    The parameter ``scale`` sets an observation level. Higher scale means
    less and larger segments. ``sigma`` is the diameter of a Gaussian kernel,
    used for smoothing the image prior to segmentation.

    The number of produced segments as well as their size can only be
    controlled indirectly through ``scale``. Segment size within an image can
    vary greatly depending on local contrast.

    For RGB images, the algorithm uses the euclidean distance between pixels in
    color space.

    Parameters
    ----------
    image : (width, height, 3) or (width, height) ndarray
        Input image.
    scale : float
        Free parameter. Higher means larger clusters.
    sigma : float
        Width (standard deviation) of Gaussian kernel used in preprocessing.
    min_size : int
        Minimum component size. Enforced using postprocessing.
    multichannel : bool, optional (default: True)
        Whether the last axis of the image is to be interpreted as multiple
        channels. A value of False, for a 3D image, is not currently supported.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    References
    ----------
    .. [1] Efficient graph-based image segmentation, Felzenszwalb, P.F. and
           Huttenlocher, D.P.  International Journal of Computer Vision, 2004

    Notes
    -----
        The `k` parameter used in the original paper renamed to `scale` here.

    Examples
    --------
    >>> from skimage.segmentation import felzenszwalb
    >>> from skimage.data import coffee
    >>> img = coffee()
    >>> segments = felzenszwalb(img, scale=3.0, sigma=0.95, min_size=5)
    """
    if dright_cost != 0:
        return _felzenszwalb8_all_cython(down_cost, right_cost, dright_cost, uright_cost, height=height, width=width, scale=scale, sigma=sigma,
                         min_size=min_size)
    else:
        return _felzenszwalb4_all_cython(down_cost, right_cost, height=height, width=width, scale=scale, sigma=sigma,
                         min_size=min_size)
