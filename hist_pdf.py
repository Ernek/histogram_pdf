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

#print(data)
data_array = np.array(data, dtype=float)
#print(data_array)

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
f = plt.figure()
plt.xlim(left=440, right=500)
plt.ylim(top=0.12)
sns.distplot(dist, bins='rice', fit=stats.norm, fit_kws={"label": "Norm Fit\nmu={:.2f} sigma={:.2f}".format(mu, sigma), "lw": 3}, kde=True,kde_kws={"label": "KDE"}, rug=True, rug_kws={"color": "blue"},hist_kws={"histtype": "step", "lw": 2}, hist=True)
plt.legend(loc='upper left')
print("mu={0} , sigma={1}".format(mu, sigma))
# plt.show()
plt.tight_layout()
f.savefig("test.png", bbox_inches=None)

# f.savefig("test.pdf", bbox_inches="tight")
# print(mean)


# Code for a multiple column file with data starting on row "R"

#Function to read a file and obtain an array of its column data
def read_file_lines(ffile, Nskip):
    with open(ffile) as f:
        flines = f.readlines()
        Ncol = len(flines[0].split())  #Number of columns based on the first row data or labels
        fwlist = []
        for i in range(len(flines)):
            if i <= Nskip - 1 :
                continue
            else:
                fwlist.append(flines[i].split())

        fwarray = np.array(fwlist)

        final_data = []
        for k in range(Ncol):
            fcarray = []
            for i in range(len(fwarray)):
                if k + 1 > len(fwarray[i]):
                    continue
                fcarray.append(fwarray[i][k])
            final_data.append(fcarray)

    return fwarray, final_data

data_test = read_file_lines("nmr_al_na_all_nofilecount.dat", 1)[1]
print(data_test[0], "\n", data_test[-1])

for i in range(len(data_test)):
    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            #print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)
    #print(dist)
    #print('JAJAJAJAJA')

    (mu, sigma) = stats.norm.fit(dist)
    f = plt.figure()
    plt.xlim(left=440, right=500)
    plt.ylim(top=0.12)
    sns.distplot(dist, bins='rice', fit=stats.norm,
                 fit_kws={"label": "Norm Fit\nmu={:.2f} sigma={:.2f}".format(mu, sigma), "lw": 3}, kde=True,
                 kde_kws={"label": "KDE"}, rug=True, rug_kws={"color": "blue"}, hist_kws={"histtype": "step", "lw": 2},
                 hist=True)
    plt.legend(loc='upper left')
    print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
    plt.tight_layout()
    f.savefig(f"test_{i+1}.png", bbox_inches=None)

f = plt.figure()
plt.xlim(left=440, right=500)
plt.ylim(top=0.12)
#plt.legend(loc='upper left')
labels = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
for i in range(len(data_test)):
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            # print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)
    # print(dist)
    # print('JAJAJAJAJA')
    #d_array = np.array(data_test[i], dtype=float)
    dist = pd.DataFrame(d_array)
    (mu, sigma) = stats.norm.fit(dist)

    sns.distplot(dist, bins='rice', fit=stats.norm,
                 fit_kws={"lw": 0.5}, kde=True,
                 kde_kws={"label": f"{labels[i]}"}, hist_kws={"histtype": "step", "lw": 2},
                 hist=True)

    #print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
plt.tight_layout()
f.savefig(f"test_all.png", bbox_inches=None)


f = plt.figure()
plt.xlim(left=440, right=500)
plt.ylim(top=0.12)
#plt.legend(loc='upper left')
labels = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
for i in range(len(data_test)):
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            # print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)

    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)

    (mu, sigma) = stats.norm.fit(dist)

    sns.distplot(dist, bins='rice', kde=True,
                 kde_kws={"label": f"{labels[i]}"}, hist_kws={"histtype": "step", "lw": 2},
                 hist=True)
    #print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
plt.tight_layout()
f.savefig(f"test_all_nofit.png", bbox_inches=None)



f = plt.figure()
plt.xlim(left=440, right=500)
plt.ylim(top=0.12)
#plt.legend(loc='upper left')
labels = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
for i in range(len(data_test)):
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            # print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)
    # print(dist)
    # print('JAJAJAJAJA')
    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)
    (mu, sigma) = stats.norm.fit(dist)

    sns.distplot(dist, bins='rice', kde=True,
                 kde_kws={"label": f"{labels[i]}"}, hist=False)

    #print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
plt.tight_layout()
f.savefig(f"test_all_nofit_nohist.png", bbox_inches=None)


f = plt.figure()
plt.xlim(left=440, right=500)
plt.ylim(top=0.12)
#plt.legend(loc='upper left')
labels = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
for i in range(len(data_test)):
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            # print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)
    # print(dist)
    # print('JAJAJAJAJA')
    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)
    (mu, sigma) = stats.norm.fit(dist)

    sns.distplot(dist, bins='rice', fit=stats.norm,
                 fit_kws={"label":f"{labels[i]}", "color":f"C{i}",  "lw":1.5}, kde=False, hist=False)

    #print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
plt.legend(loc='upper left')
plt.tight_layout()
f.savefig(f"test_all_onlyfit.png", bbox_inches=None)


f = plt.figure()
#plt.legend(loc='upper left')
labels = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
for i in range(len(data_test)):
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            # print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])

    d_array = np.array(new_list, dtype=float)
    dist = pd.DataFrame(d_array)
    # print(dist)
    # print('JAJAJAJAJA')
    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)

    (mu, sigma) = stats.norm.fit(dist)

    sns.distplot(dist, bins='rice', hist_kws={"histtype": "step", "label":f"{labels[i]}", "color":f"C{i}", "lw": 2},
                 hist=True, kde=False)

    #print("mu={0} , sigma={1}".format(mu, sigma))
    # plt.show()
plt.legend(loc='upper left')
plt.tight_layout()
f.savefig(f"test_all_onlyhist.png", bbox_inches=None)


print("\n\n\n\n")
print(data_test)
print("\n", len(data_test))
for i in range(len(data_test)):
    print(len(data_test[i]))


final_list = []
for i in range(len(data_test)):
    #d_array = np.array(data_test[i], dtype=float)
    #dist = pd.DataFrame(d_array)
    new_list = []
    for k in range(len(data_test[i])):
        #print(data_test[i][k])
        if data_test[i][k] == "null":
            #print('YOU KNOW')
            continue
        else:
            new_list.append(data_test[i][k])
    final_list.append(new_list)

print("\n\n\n\n")
print(final_list)

for i in range(len(final_list)):
    print(len(final_list[i]))

data_dataframe = pd.DataFrame(final_list).T
print(data_dataframe)
    #d_array = np.array(new_list, dtype=float)
    #dist = pd.DataFrame(d_array)
#print(final_list)
#nfinal_list = []
#for i in range(len(final_list)):
#    arr = np.array(final_list[i], dtype=float)
#    nfinal_list.append(arr)
#print("\n\n\n\n")
#print(nfinal_list)

# print("\n\n\n\n")
# final_array = np.array([np.array(xi) for xi in final_list])
# print(final_array)

#data_dataframe = pd.DataFrame(final_array)
#print(data_dataframe)
data_dataframe.columns = ['2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0']
print(data_dataframe)


fig, ax = plt.subplots()
sns.boxplot(data=data_dataframe)
ax.set_ylabel('Al NMR shielding tensor')
ax.set_xlabel('r(Na-Al) in Å')
sns.despine(offset=10, trim=True)


#for i in range(8):
#    for j in range(1):
#        ax[i,j].sns.displot

#plt.show()
fig.savefig('al_na_nmr_N_4.png',  bbox_inches='tight')
#g = sns.FacetGrid(data_dataframe, col="2.5", height=1.7, aspect=4,)

#g.map(sns.distplot,

      # (dist, bins='rice', fit=stats.norm,
      #            fit_kws={"label": "Norm Fit", "lw": 3}, kde=True,
      #            kde_kws={"label": "KDE"}, rug=True, rug_kws={"color": "blue"}, hist_kws={"histtype": "step", "lw": 2},
      #            hist=True))

