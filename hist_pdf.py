import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

f_inp_name="nmr_data_test.dat"
with open(f_inp_name) as finp:
    finp_lines = finp.readlines()
    finp_text = []
    # print(finp_lines[0].split())
    for i in range(len(finp_lines)):
        finp_text.append(finp_lines[i].split())

data = []
for i in range(len(finp_lines)):
    data.append(finp_text[i][5])

print(data)
data_array = np.array(data, dtype=float)
print(data_array)

# Histogram calculations in Numpy

hist, bin_edges = np.histogram(data_array)

print(hist, bin_edges)

# Visualizing the histogram



# An "interface" to matplotlib.axes.Axes.hist() method
#n, bins, patches = plt.hist(x=data_array, bins='rice', color='#0504aa',
#                            alpha=0.7, rwidth=0.85)
#plt.grid(axis='y', alpha=0.75)
#plt.xlabel('Value')
#plt.ylabel('Frequency')
#plt.title('My Very Own Histogram')
## plt.text(23, 45, r'$\mu=15, b=3$')
#maxfreq = n.max()
#
## Set a clean upper y-axis limit.
#plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#
#print(n, len(bins), bins)
#
#plt.show()

# Plotting a Kernel Density Estimate (KDE)
# fig, ax = plt.subplots()
dist = pd.DataFrame(data_array)
# print(dist)
#
# dist.plot.kde(ax=ax)
# dist.plot.hist(density=True, ax=ax)
#plt.show()

# Alternative with Seaborn
from scipy import stats
import seaborn as sns
(mu, sigma) = stats.norm.fit(dist)
sns.distplot(dist, bins='rice', fit=stats.norm, fit_kws={"label": "Norm Fit mu={:.2f} sigma={:.2f}".format(mu, sigma), "lw": 3}, kde=True,kde_kws={"label": "KDE"}, rug=True, rug_kws={"color": "blue"},hist_kws={"histtype": "step", "lw": 2}, hist=True)
plt.legend(loc='best')
print("mu={0} , sigma={1}".format(mu, sigma))
plt.show()

# print(mean)