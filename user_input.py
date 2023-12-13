import matplotlib.pyplot as plt
from input_interpreter.digit_input import draw_input, convert_to_bitmap, matrix_mapper
import pickle

with open('nn.pickle', 'rb') as f:
    nn = pickle.load(f)

lines = draw_input()
X = convert_to_bitmap(lines)
x = x = matrix_mapper(X, 28, 28)
x[x<200] = 0
plt.imshow(x, cmap='gray')
plt.show()

digit = (255-x).reshape(784)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.5))
ax1.bar(height=nn.predict(digit), x=[f'{i}' for i in range(10)], color='b')
ax2.imshow((255-digit).reshape((28, 28)), cmap='gray')
ax2.set_axis_off()
plt.show()