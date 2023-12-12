import matplotlib.pyplot as plt
from input_interpreter.digit_input import draw_input, convert_to_bitmap, matrix_mapper

lines = draw_input()
X = convert_to_bitmap(lines)
x = x = matrix_mapper(X, 28, 28)
x[x<200] = 0
plt.imshow(x, cmap='gray')
plt.show()

digit = (255-x).reshape(784)
