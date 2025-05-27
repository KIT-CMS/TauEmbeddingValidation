import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from importer import verify_events, initialize_dir
from genmatching import subtract_columns
from plotting import histogram, q_comparison

hdf_path = "./data/converted/converted_nanoaod.h5"
dr_plot_path = "./output/dr_plots"

#Initialize plotting directory

initialize_dir(dr_plot_path)


print("Directory initialized")

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df_matched = pd.read_hdf(hdf_path, "emb_df_matched")


print("Data loaded and verified")


bins = np.linspace(0,5,40)

dphi_1 = subtract_columns(emb_df_matched["phi_1"], data_df["phi_1"], "phi_1")
deta_1 = subtract_columns(emb_df_matched["eta_1"], data_df["eta_1"], "eta_1")
dr_1 = np.sqrt(np.square(dphi_1) + np.square(deta_1))
ax = histogram(dr_1, bins, r"$\delta r_\text{µ1}$")
plt.savefig(os.path.join(dr_plot_path, f"dr_unmatched.png"))
plt.close()

dphi_2 = subtract_columns(emb_df_matched["LM_phi"], data_df["LM_phi"], "LM_phi")
deta_2 = subtract_columns(emb_df_matched["LM_eta"], data_df["LM_eta"], "LM_eta")
dr_2 = np.sqrt(np.square(dphi_2) + np.square(deta_2))
ax = histogram(dr_2, bins, r"$\delta r_\text{µ1}$")
plt.savefig(os.path.join(dr_plot_path, f"dr_matched.png"))
plt.close()





# dr = calculate_dr(data_df, 5, filter=None)
# mu_index, mu_dr = get_closest_muon_data(dr)
# ax = histogram(mu_dr, np.linspace(0,0.5,25), "$\delta r$")

# ax.set_yscale("log")
# ax.set_xlim(0,0.5)
# plt.savefig(os.path.join(dr_plot_path, f"dr_spectra.png"))
# plt.close()

#creating plots twice (1. with filter, then without)
# for filter in [filter_list, None]:
    
#     dr = calculate_dr(emb_df, 5, filter=filter)
#     mu_index, mu_dr = get_closest_muon_data(dr)

#     if filter:
#         name_extension = "_filtered"
#     else:
#         name_extension = "_unfiltered"

    # #plotting dr spectra
    # ax = histogram(dr[:,0,0], 25, r"$\delta r(µ_\text{data, 1}, µ_\text{emb, 1})$")
    # ax.set_yscale("log")
    # plt.savefig(os.path.join(dr_plot_path, f"dr_raw{name_extension}.png"))
    # plt.close()





    # #plotting closest dr without matching
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.set_figheight(14)
    # fig.set_figwidth(14)

    # hist2, edges, _ = ax.hist([dr[mu_index==num, 0,0] for num in np.unique(mu_index)], bins=25, label=[f"mindr bei emb $µ_{num+1} $)" for num in np.unique(mu_index)], stacked=True)
    # hist, _, _ = ax.hist(dr[:,0,0], bins=edges, label=f"dr(data $µ_1$, emb $µ_{1}$)", histtype="step", color="black", linewidth=1.6)

    # hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    # ax.set_ylabel("Counts")
    # ax.set_xlabel(r"$dr(\text{data, emb})$")
    # ax.set_yscale("log")

    # plt.legend()
    # plt.savefig(os.path.join(dr_plot_path, f"dr_uncorrected{name_extension}.png"))
    # plt.close()



    #plotting closest dr with matching

    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # fig.set_figheight(14)
    # fig.set_figwidth(14)

    # hist2, edges, _ = ax.hist([mu_dr[mu_index==num] for num in np.unique(mu_index)], bins=25, label=[f"mindr bei emb $µ_{num+1} $)" for num in np.unique(mu_index)], stacked=True)
    # hist, _, _ = ax.hist(dr[:,0,0], bins=edges, label=f"dr(data $µ_1$, emb $µ_{1}$)", histtype="step", color="black", linewidth=1.6)

    # hep.cms.label("Private work (data/simulation)", data=True, loc=0, year="2022G", com=13.6)#, lumi=59.8
    # ax.set_ylabel("Counts")
    # ax.set_xlabel(r"$dr(\text{data, emb})$")
    # ax.set_yscale("log")
    # # ax.set_xlim(-0.01,0.01)

    # plt.legend()
    # plt.savefig(os.path.join(dr_plot_path, f"dr_corrected{name_extension}.png"))
    # plt.close()

print("Plotting finished")