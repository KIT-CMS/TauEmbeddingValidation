import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from importer import verify_events, initialize_dir
from genmatching import calculate_dr, get_closest_muon_data, get_filter_list
from plotting import histogram, q_comparison

hdf_path = "./data/converted/converted_nanoaod.h5"
dr_plot_path = "./output/dr_plots"

#Initialize plotting directory

initialize_dir(dr_plot_path)


print("Directory initialized")

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df_matched")

verify_events(data_df, emb_df)

print("Data loaded and verified")

filter_list = get_filter_list()
dr_unfiltered = calculate_dr(emb_df, 5, filter=None)
dr_filtered = calculate_dr(emb_df, 5, filter=filter_list)

ax = q_comparison(dr_filtered[:,0,0], dr_unfiltered[:,0,0], np.linspace(0,6), r"$\Delta r_\text{filtered}$", r"$\Delta r_\text{unfiltered}$", r"$\Delta r$")
ax[0].set_yscale("log")
plt.savefig(os.path.join(dr_plot_path, f"dr_comparison.png"))
plt.close()

#creating plots twice (1. with filter, then without)
for filter in [filter_list, None]:
    
    dr = calculate_dr(emb_df, 5, filter=filter)
    mu_index, mu_dr = get_closest_muon_data(dr)

    if filter:
        name_extension = "_filtered"
    else:
        name_extension = "_unfiltered"

    #plotting dr spectra
    ax = histogram(dr[:,0,0], 25, r"$\delta r(µ_\text{data, 1}, µ_\text{emb, 1})$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(dr_plot_path, f"dr_raw{name_extension}.png"))
    plt.close()





    #plotting closest dr without matching
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(14)
    fig.set_figwidth(14)

    hist2, edges, _ = ax.hist([dr[mu_index==num, 0,0] for num in np.unique(mu_index)], bins=25, label=[f"mindr bei emb $µ_{num+1} $)" for num in np.unique(mu_index)], stacked=True)
    hist, _, _ = ax.hist(dr[:,0,0], bins=edges, label=f"dr(data $µ_1$, emb $µ_{1}$)", histtype="step", color="black", linewidth=1.6)

    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"$dr(\text{data, emb})$")
    ax.set_yscale("log")

    plt.legend()
    plt.savefig(os.path.join(dr_plot_path, f"dr_uncorrected{name_extension}.png"))
    plt.close()



    #plotting closest dr with matching

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(14)
    fig.set_figwidth(14)

    hist2, edges, _ = ax.hist([mu_dr[mu_index==num] for num in np.unique(mu_index)], bins=25, label=[f"mindr bei emb $µ_{num+1} $)" for num in np.unique(mu_index)], stacked=True)
    hist, _, _ = ax.hist(dr[:,0,0], bins=edges, label=f"dr(data $µ_1$, emb $µ_{1}$)", histtype="step", color="black", linewidth=1.6)

    hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"$dr(\text{data, emb})$")
    ax.set_yscale("log")

    plt.legend()
    plt.savefig(os.path.join(dr_plot_path, f"dr_corrected{name_extension}.png"))
    plt.close()

print("Plotting finished")