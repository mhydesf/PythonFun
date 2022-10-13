"""
NOT AS FAST AS THE SINGLE
THREADED METHOD...yet ;)
"""

import numpy as np

thread_results = {
    1: None,
    2: None,
    3: None,
    4: None
}

def compute_mandlebrot_iters(num: complex,
                             attempts: int) -> int:
    """
    Computes how long a number takes to diverge in when
    passed into the function which generates the Mandlebrot
    Set. If the magnitude of the complex number exceeds 2
    then the function will diverge.

    Args:
        num (complex): Point in the complex plain to
                        analyze in the Mandlebrot Set

    Returns:
        int: Number of iterations until divergence
    """
    f0 = complex(0, 0)
    for att in range(attempts):
        f0 = f0*f0 + num
        if f0.real + f0.imag > 2:
            # return number of attempts at
            # point of divergence
            return att
    # point does not diverge
    return att

def mandlebrotify(space: np.ndarray,
                  attempts: int,
                  key: int) -> None:
    """
    Calculates a sub-region of the complex
    plane to be joined later as a full set.

    Args:
        space (np.ndarray): Sub-region of complex plane
        attempts (int): Number of attempts to determine
                        divergence
        key (int): Dict key to save results
    """
    region = np.zeros(shape=space.shape, dtype=np.int8)
    for i, row in enumerate(space):
        for j, val in enumerate(row):
            region[i, j] = compute_mandlebrot_iters(val, attempts=attempts)
    thread_results[key] = region

if __name__ == "__main__":

    from threading import Thread
    import matplotlib.pyplot as plt
    import matplotlib.cm as colors
    plt.style.use('dark_background')

    STEP = 0.001
    TRIES = 101

    x = np.arange(-3, 1, STEP)
    y = np.arange(-2, 2, STEP)
    div = int(len(x)/2)

    space = np.array([[complex(xv, yv) for xv in x] for yv in y])
    
    quad1 = space[:div, :div] # TOP LEFT
    quad2 = space[:div, div:] # BOT LEFT
    quad3 = space[div:, :div] # TOP RIGHT
    quad4 = space[div:, div:] # BOT RIGHT

    quadrants = [quad1, quad2, quad3, quad4]
    threads = [Thread(target=mandlebrotify, args=(q, TRIES, i+1))\
                for i, q in enumerate(quadrants)]

    for t in threads:
        t.start()
    
    for t in threads:
        t.join()

    top_half = np.concatenate((thread_results[1], thread_results[3]), axis=0)
    bot_half = np.concatenate((thread_results[2], thread_results[4]), axis=0)
    result = np.concatenate((top_half, bot_half), axis=1)
    plt.imshow(result)
    plt.show()