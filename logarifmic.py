# | Created by Ar4ikov
# | Время: 26.04.2020 - 15:46

import pylab
import matplotlib.pyplot as plt
a = [pow(10, i) for i in range(10)]
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)

line, = ax.plot(a, color='blue', lw=1)

ax.set_yscale('log')

pylab.show()