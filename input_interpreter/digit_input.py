"""
This code is sligthly messy, and some of it is borrowed, but it works!
like https://www.reddit.com/r/Bossfight/comments/e3e582/beachy_the_udder_walking_cow/
"""

import matplotlib.pyplot as plt
import numpy as np


def center_lines(lines):
    x_min, x_max, y_min, y_max = 1, 0, 1, 0

    for line in lines:
        x_min = min(min(p[0] for p in line), x_min)
        x_max = max(max(p[0] for p in line), x_max)
        y_min = min(min(p[1] for p in line), y_min)
        y_max = max(max(p[1] for p in line), y_max)

    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2

    scale = 0.3/(y_max-y_mid)

    centered_lines = []
    for line in lines:
        centered_lines += [(scale*(x-x_mid)+0.5, scale*(y-y_mid)+0.5) for x, y in line],

    return centered_lines


def draw_input():
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Write a digit')

    drawing = False
    lines = []

    def on_click(event):
        nonlocal drawing
        drawing = True
        lines.append([(event.xdata, event.ydata)])  # Start a new line

    def on_release(event):
        nonlocal drawing
        drawing = False

    def on_motion(event):
        if drawing:
            lines[-1].append((event.xdata, event.ydata))  # Add point to the current line
            ax.plot(event.xdata, event.ydata, 'bo')
            fig.canvas.draw()

    # Attach the event listeners
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()

    return center_lines(lines)


def draw_line(mat, x0, y0, x1, y1):
    """"
    Copied and slightly modified solution from
    https://stackoverflow.com/questions/50387606/python-draw-line-between-two-coordinates-in-a-matrix/50388949#50388949
    """

    if not (0 <= x0 < mat.shape[0] and 0 <= x1 < mat.shape[0] and
            0 <= y0 < mat.shape[1] and 0 <= y1 < mat.shape[1]):
        raise ValueError('Invalid coordinates.')

    # Swap axes if Y slope is smaller than X slope
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        mat = mat.T
        x0, y0, x1, y1 = y0, x0, y1, x1

    # Swap line direction to go left-to-right if necessary
    if x0 > x1:
        x0, y0, x1, y1 = x1, y1, x0, y0
    # Compute intermediate coordinates using line equation

    x = np.arange(x0, x1)
    try:
        y = np.round(((y1 - y0) / (x1 - x0)) * (x - x0) + y0).astype(x.dtype)
    except ZeroDivisionError:
        return mat if not transpose else mat.T

    # Write intermediate coordinates
    mat[x, y] -= 180
    mat[x-1, y] -= 180
    mat[x+1, y] -= 180
    mat[x, y-1] -= 180
    mat[x, y+1] -= 180
   
    mat = np.maximum(mat, 0)

    return mat if not transpose else mat.T


def convert_to_bitmap(lines):
    bitmap = np.zeros((84, 84))+255
    for line in lines:
        for r0, r1 in zip(line[:-1], line[1:]):
            y0, x0 = r0
            y1, x1 = r1
            x0 = round(x0*84)
            y0 = round(y0*84)
            x1 = round(x1*84)
            y1 = round(y1*84)

            bitmap = draw_line(bitmap, 84-x0, y0, 84-x1, y1)
    return bitmap


def matrix_mapper(X: np.ndarray, n: int, m: int):
    """
    Function that maps one matrix X to another x
    """
    N, M = X.shape
   
    # Get number of rows and columns to remove
    N_div_n, N_mod_n = divmod(N, n)
    M_div_m, M_mod_m = divmod(M, m)

    # Determine how many to the left/right/top/bottom
    N_mod_n_div_2, N_mod_n_mod_2 = divmod(N_mod_n, 2)
    M_mod_m_div_2, M_mod_m_mod_2 = divmod(M_mod_m, 2)

    remove_t = N_mod_n_div_2
    remove_b = N_mod_n_div_2+N_mod_n_mod_2
    remove_l = M_mod_m_div_2
    remove_r = M_mod_m_div_2+M_mod_m_mod_2

    # Return P'XQ
    return (
        np.repeat(np.eye(n), N_div_n, axis=0).T
        .dot(X[remove_t:N-remove_b, remove_l:M-remove_r])
        .dot(np.repeat(np.eye(m), M_div_m, axis=0))
        /(N_div_n*M_div_m)
    )


if __name__ == '__main__':
    lines = draw_input()
    X = convert_to_bitmap(lines)
    x = x = matrix_mapper(X, 28, 28)
    x[x<200] = 0
    plt.imshow(x, cmap='gray')
    plt.show()
    
