"""
This is a bit messy, sorry about that. But it works :)
"""

import numpy as np
import matplotlib.pyplot as plt
from input_interpreter.digit_input import draw_input, convert_to_bitmap, matrix_mapper
import pickle

with open('nn.pickle', 'rb') as f:
    nn = pickle.load(f)

while True:
	try:
		lines = draw_input()
		X = convert_to_bitmap(lines)
		break
	except ValueError:
		pass

x = matrix_mapper(X, 28, 28)
x[x<200] = 0
digit = (255-x).reshape(784)

digit_norm = (digit-digit.mean())/digit.std()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.bar(height=nn.predict(digit_norm), x=[f'{i}' for i in range(10)], color='b')
ax2.imshow((255-digit).reshape((28, 28)), cmap='gray')
ax2.set_axis_off()
plt.show()
		
correct = input('What was the digit?: ')

print('Training before: ', nn.training)
try:
	nn.train([[digit_norm, 1*np.array([int(correct) == i for i in range(10)])]], 1, 0.005)
	print('Training after: ', nn.training)
except ValueError:
	print('No new training completed')

with open('nn.pickle', 'wb') as f:
    pickle.dump(nn, f)

