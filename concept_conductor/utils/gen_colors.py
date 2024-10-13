import random
import numpy as np
from concept_conductor.utils import hasel

 
def get_n_hsl_colors(num, seed=0):
    """Generate a number of random contrasting HSL colors.

    Args:
        num (int): Number of colors needed.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of processed arrays, each of shape (num,)
    """
    np.random.seed(seed)
    h = np.linspace(0, 360, num)
    h_total_bias = np.random.rand(1)[0] * 360
    h += h_total_bias
    h %= 360
    # if num >= 3:
    #     h_bias = (np.random.rand(num - 2) * 2 - 1) * interval / 2
    #     h[1:-1] += h_bias
    s = np.random.beta(a=5, b=2, size=num) * 100
    l = np.random.beta(a=4, b=3, size=num) * 100
    
    
    return (h,s,l)
 
def gen_n_colors(num, seed=0, shuffle=False):
    """Generate a number of random contrasting RGB colors.

    Args:
        num (int): Number of colors needed.
        seed (int, optional): Random seed. Defaults to 0.
        shuffle (bool, optional): Whether or not to shuffle the output colors. Defaults to False.

    Returns:
        List[List[int, int, int] * num]: List of RGB colors.
    """
    rgb_colors = []
    h, s, l = get_n_hsl_colors(num, seed=seed)
    hsl = np.expand_dims(np.stack([h / 360, s / 100, l / 100], axis=0).T, axis=0)
    rgb = hasel.hsl2rgb(hsl)
    rgb_colors = rgb[0].tolist()
    if shuffle:
        random.seed(seed)
        random.shuffle(rgb_colors)
 
    return rgb_colors