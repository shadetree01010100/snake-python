BLACK = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)

def blend_colors(color_a, color_b, alpha):
    """ Returns a new color that is `alpha` percent `color_a` and
    remainder `color_b`.
    """
    channels = zip(color_a, color_b)
    blended = []
    for a, b in channels:
        channel = round(a * alpha + (1 - alpha) * b)
        blended.append(channel)
    return tuple(blended)
