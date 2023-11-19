import matplotlib.pyplot as plt
from ipywidgets import Output
from plt_overfit import overfit_example, output
from IPython.display import display
plt.style.use('../../deeplearning.mplstyle')

plt.close("all")
display(output)
ofit = overfit_example(False)

plt.show()