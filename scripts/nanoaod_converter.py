import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

from importer import nanoaod_to_dataframe, get_z_m_pt, initialize_dir
from genmatching import calculate_dr, apply_genmatching, get_filter_list
from helper import verify_events, create_concordant_subsets, copy_columns_from_to, get_matching_df, subtract_columns
from plotting import match_plot, dr_plot

########################################################################################################################################################################
# paths for input and output 
########################################################################################################################################################################
data_path = "./data/2022G-nanoaod_gen/"
emb_path = "./data/2022G-nanoaod_gen/"

data_filenames = "2022G-data_*.root"
emb_filenames = "2022G-emb_gen_*.root"

output_path = "./data/converted"

match_plot_path = "./output/match_plots"

########################################################################################################################################################################
# columns to be read and their new names in the resulting df
########################################################################################################################################################################


data_quantities = [
    {"key":"PuppiMET_pt",       "target":"PuppiMET_pt",     "expand":False},
    {"key":"PuppiMET_phi",      "target":"PuppiMET_phi",    "expand":False},
    {"key":"Muon_phi",          "target":"phi",             "expand":True},
    {"key":"Muon_pt",           "target":"pt",              "expand":True},
    {"key":"Muon_eta",          "target":"eta",             "expand":True},
    {"key":"Muon_mass",         "target":"m",               "expand":True},
    {"key":"Jet_phi",           "target":"Jet_phi",         "expand":False},
    {"key":"Jet_pt",            "target":"Jet_pt",          "expand":False},
    {"key":"Jet_eta",           "target":"Jet_eta",         "expand":False},
    {"key":"Jet_mass",          "target":"Jet_mass",        "expand":False},
    {"key":"run",               "target":"run",             "expand":False},
    {"key":"luminosityBlock",   "target":"lumi",            "expand":False},
    {"key":"event",             "target":"event",           "expand":False}
]

selection_q = [
    {"key":"TauEmbedding_chargeLeadingMuon",		"target":"LM_charge",	"expand":False },
    {"key":"TauEmbedding_chargeTrailingMuon",		"target":"TM_charge",	"expand":False },
    {"key":"TauEmbedding_phiLeadingMuon",		    "target":"LM_phi",		"expand":False },
    {"key":"TauEmbedding_phiTrailingMuon",		    "target":"TM_phi",		"expand":False },
    {"key":"TauEmbedding_ptLeadingMuon",		    "target":"LM_pt",		"expand":False },
    {"key":"TauEmbedding_ptTrailingMuon",		    "target":"TM_pt",		"expand":False },
    {"key":"TauEmbedding_etaLeadingMuon",		    "target":"LM_eta",		"expand":False },
    {"key":"TauEmbedding_etaTrailingMuon",		    "target":"TM_eta",		"expand":False },
    {"key":"TauEmbedding_massLeadingMuon",		    "target":"LM_m",		"expand":False },
    {"key":"TauEmbedding_massTrailingMuon",		    "target":"TM_m",		"expand":False },
]

emb_quantities = data_quantities.copy()
emb_quantities += selection_q

########################################################################################################################################################################
# Reading data
########################################################################################################################################################################

print("Loading data")

data_files = list(pathlib.Path(data_path).glob(data_filenames))
emb_files = list(pathlib.Path(emb_path).glob(emb_filenames))

data_df = nanoaod_to_dataframe(files=data_files, quantities=data_quantities)
emb_df = nanoaod_to_dataframe(files=emb_files, quantities=emb_quantities)

print("Data loaded")

########################################################################################################################################################################
# Keeping only those events that are both in data and embedding 
########################################################################################################################################################################
data_df, emb_df = create_concordant_subsets(data_df, emb_df)

verify_events(data_df, emb_df)

print("Data ok")

########################################################################################################################################################################
# copying the columns with info about muons used for embedding to original dataset so that they can be treated equally
######################################################################################################################################################################## 
selection_q_converted = [element["target"] for element in selection_q]
data_df, emb_df = copy_columns_from_to(emb_df, data_df, selection_q_converted)

print("Copied to data:", selection_q_converted)

########################################################################################################################################################################
# Applying matching
########################################################################################################################################################################

emb_df_for_matching = get_matching_df(emb_df, ["LM_pt", "TM_pt", "LM_eta", "TM_eta", "LM_phi", "TM_phi", "LM_m", "TM_m"])

dr = calculate_dr(emb_df, 5, filter=None)
emb_df_matched, muon_id_matched, dr_matched = apply_genmatching(dr.copy(), emb_df_for_matching.copy(deep=True))

filter_list = get_filter_list()
dr = calculate_dr(emb_df, 5, filter=filter_list)
emb_df_matched_filtered, muon_id_matched_filtered, dr_matched_filtered = apply_genmatching(dr.copy(), emb_df_for_matching.copy(deep=True))

print("Genmatching applied")


########################################################################################################################################################################
# Creating plots indicating performance of matching
########################################################################################################################################################################

initialize_dir(match_plot_path)

dphi_1 = subtract_columns(emb_df["phi_1"], data_df["phi_1"], "phi_1")
deta_1 = subtract_columns(emb_df["eta_1"], data_df["eta_1"], "eta_1")
dr_1 = np.sqrt(np.square(dphi_1) + np.square(deta_1))
dphi_2 = subtract_columns(emb_df["phi_2"], data_df["phi_2"], "phi_2")
deta_2 = subtract_columns(emb_df["eta_2"], data_df["eta_2"], "eta_2")
dr_2 = np.sqrt(np.square(dphi_2) + np.square(deta_2))

#dr between muon1|2 data and muon1|2 embedding
ax = dr_plot(np.column_stack([dr_1, dr_2]), r"$\delta r_\text{unmatched}$")
ax.set_yscale("log")
plt.savefig(os.path.join(match_plot_path, f"dr_unmatched.png"))
plt.close()

#dr between l|m muon data and l|m muon embedding
ax = dr_plot(dr_matched, r"$\delta r_\text{matched}$")
ax.set_yscale("log")
plt.savefig(os.path.join(match_plot_path, f"dr_matched.png"))
plt.close()

#dr between l|m muon data and l|m muon embedding with filters
ax = dr_plot(dr_matched_filtered, r"$\delta r_\text{matched+filtered}$")
ax.set_yscale("log")
plt.savefig(os.path.join(match_plot_path, f"dr_matched+filtered.png"))
plt.close()

#frequency of muon id to be used as l|m muon
ax = match_plot(muon_id_matched)
ax.set_yscale("log")
plt.savefig(os.path.join(match_plot_path, f"id_matched.png"))
plt.close()

#frequency of muon id to be used as l|m muon
ax = match_plot(muon_id_matched_filtered)
ax.set_yscale("log")
plt.savefig(os.path.join(match_plot_path, f"id_matched+filtered.png"))
plt.close()

print("Created plots")

########################################################################################################################################################################
# Adding mvis and ptvis
########################################################################################################################################################################

data_df["m_vis"], data_df["pt_vis"] = get_z_m_pt(data_df)
emb_df_matched["m_vis"], emb_df_matched["pt_vis"] = get_z_m_pt(emb_df_matched)
emb_df_matched_filtered["m_vis"], emb_df_matched_filtered["pt_vis"] = get_z_m_pt(emb_df_matched_filtered)

print("Added m_vis and pt_vis")

########################################################################################################################################################################
# Storing datarames in hdf store
########################################################################################################################################################################

initialize_dir(output_path)

store = pd.HDFStore(os.path.join(output_path, "converted_nanoaod.h5"), 'w')  
store.put("data_df", data_df, index=False)
store.put("emb_df_matched", emb_df_matched, index=False)
store.put("emb_df_matched_filtered", emb_df_matched_filtered, index=False)
store.close()

print("Data stored in hdf store")

#thats how you import the data
# hdf_path = "./data/converted/converted_nanoaod.h5"
# data_df = pd.read_hdf(hdf_path, "data_df")
# emb_df = pd.read_hdf(hdf_path, "emb_df")

