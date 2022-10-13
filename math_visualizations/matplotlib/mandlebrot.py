import numpy as np

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

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.cm as colors
    plt.style.use('dark_background')

    STEP = 0.0001
    TRIES = 101

    x = np.arange(-3, 1, STEP)
    y = np.arange(-2, 2, STEP)

    mandle = np.zeros(shape=(len(x), len(y)), dtype=np.int8)

    for i, x_val in enumerate(x):
        for j, y_val in enumerate(y):
            num = complex(x_val, y_val)
            result = compute_mandlebrot_iters(num, TRIES)
            mandle[j, i] = result

    fig, ax = plt.subplots(figsize=(25, 14))
    cmap = colors.get_cmap('Greys_r')
    ax.tick_params(axis='both', colors='black')
    ax.imshow(mandle, cmap=cmap)
    plt.show()
