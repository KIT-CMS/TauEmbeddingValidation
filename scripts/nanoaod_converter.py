import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib

from source.importer import nanoaod_to_dataframe, get_z_m_pt, initialize_dir, require_min_n
from source.genmatching import calculate_dr, apply_genmatching, remove_muon_jets
from source.helper import verify_events, create_concordant_subsets, copy_columns_from_to, get_matching_df, subtract_columns, prepare_jet_matching,get_n_occurence, set_working_dir, count_n_objects
from source.plotting import match_plot, control_plot, nq_comparison

from source.importer import quality_cut, assert_object_validity, only_global_muons, compactify_objects

########################################################################################################################################################################
# paths for input and output 
########################################################################################################################################################################
data_path = "./data/2022G-nanoaod_gen/"
emb_path = "./data/2022G-nanoaod_gen/"

data_filenames = "2022G-data_*.root"
emb_filenames = "2022G-emb_gen_*.root"

output_path = "./data/converted"

match_plot_path = "./output/match_plots"

create_match_plots = True

set_working_dir()

initialize_dir(match_plot_path)

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
    {"key":"Jet_phi",           "target":"Jet_phi",         "expand":True},
    {"key":"Jet_pt",            "target":"Jet_pt",          "expand":True},
    {"key":"Jet_eta",           "target":"Jet_eta",         "expand":True},
    {"key":"Jet_mass",          "target":"Jet_m",           "expand":True},
    {"key":"run",               "target":"run",             "expand":False},
    {"key":"luminosityBlock",   "target":"lumi",            "expand":False},
    {"key":"event",             "target":"event",           "expand":False},
    {"key":"Muon_isGlobal",     "target":"MuonIsGlobal",    "expand":True}
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

data_df, emb_df = create_concordant_subsets(data_df, emb_df)

print(f"Length dataset:\t {len(emb_df)} events")

print("Data loaded")

# Creating plots comparing jet / muon object
if create_match_plots:
    njet_emb = count_n_objects(emb_df, "Jet_eta_")
    njet_data = count_n_objects(data_df, "Jet_eta_")
    max_njet = max([get_n_occurence(data_df, "Jet_eta_"), get_n_occurence(emb_df, "Jet_eta_")])

    nmu_emb = count_n_objects(emb_df, "eta_")
    nmu_data = count_n_objects(data_df, "eta_")
    max_nmu = max([get_n_occurence(data_df, "eta_"), get_n_occurence(emb_df, "eta_")])

    njet_delta = njet_data - njet_emb
    nmu_delta = nmu_data - nmu_emb
    max_delta = max([np.amax(njet_delta), np.amax(nmu_delta)])
    min_delta = min([np.amin(njet_delta), np.amin(nmu_delta)])


    ax = control_plot(njet_data, njet_emb, np.arange(-0.5, max_njet+0.5, 1), r"$n_\text{jets}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_jet_00_raw.png"))
    plt.close()

    ax = control_plot(nmu_data, nmu_emb, np.arange(-0.5, max_nmu+0.5, 1), r"$n_\text{µ}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_mu_00_raw.png"))
    plt.close()

    ax = nq_comparison({"Mu":nmu_delta, "Jet":njet_delta}, np.arange(min_delta-0.5, max_delta+0.5, 1), r"$n_\text{x, data}-n_\text{x, emb}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_delta_00_raw.png"))
    plt.close()


########################################################################################################################################################################
# Applying quality cuts on muons and jets
########################################################################################################################################################################

data_df = only_global_muons(data_df)
emb_df = only_global_muons(emb_df)

filter_dict = [
    {"col":"pt",  "min":10,  "max":None},
    {"col":"Jet_pt",  "min":25,  "max":None}
]

data_df = quality_cut(data_df, data_quantities, filter_dict)
emb_df = quality_cut(emb_df, data_quantities, filter_dict)

data_df = assert_object_validity(data_df)
emb_df = assert_object_validity(emb_df)

data_df = compactify_objects(data_df)
emb_df = compactify_objects(emb_df)


data_df, emb_df = create_concordant_subsets(data_df, emb_df)
verify_events(data_df, emb_df)

print(f"Length dataset:\t {len(emb_df)} events")

print("Jet & muon cuts applied")


# Creating plots comparing jet / muon object
if create_match_plots:
    njet_emb = count_n_objects(emb_df, "Jet_eta_")
    njet_data = count_n_objects(data_df, "Jet_eta_")
    max_njet = max([get_n_occurence(data_df, "Jet_eta_"), get_n_occurence(emb_df, "Jet_eta_")])

    nmu_emb = count_n_objects(emb_df, "eta_")
    nmu_data = count_n_objects(data_df, "eta_")
    max_nmu = max([get_n_occurence(data_df, "eta_"), get_n_occurence(emb_df, "eta_")])

    njet_delta = njet_data - njet_emb
    nmu_delta = nmu_data - nmu_emb
    max_delta = max([np.amax(njet_delta), np.amax(nmu_delta)])
    min_delta = min([np.amin(njet_delta), np.amin(nmu_delta)])

    
    ax = control_plot(njet_data, njet_emb, np.arange(-0.5, max_njet+0.5, 1), r"$n_\text{jets}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_jet_01_post_filter.png"))
    plt.close()

    
    ax = control_plot(nmu_data, nmu_emb, np.arange(-0.5, max_nmu+0.5, 1), r"$n_\text{µ}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_mu_01_post_filter.png"))
    plt.close()

    ax = nq_comparison({"Mu":nmu_delta, "Jet":njet_delta}, np.arange(min_delta-0.5, max_delta+0.5, 1), r"$n_\text{x, data}-n_\text{x, emb}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_delta_01.png"))
    plt.close()


########################################################################################################################################################################
# Applying muon matching
########################################################################################################################################################################

data_df = require_min_n(data_df, "eta_", 2)
emb_df = require_min_n(emb_df, "eta_", 2)

data_df, emb_df = create_concordant_subsets(data_df, emb_df)
verify_events(data_df, emb_df)

print(f"Length dataset:\t {len(emb_df)} events")

print("Removed events with less than 2 muons")

selection_q_converted = [element["target"] for element in selection_q]
data_df, emb_df = copy_columns_from_to(emb_df, data_df, selection_q_converted)
emb_df_for_matching = get_matching_df(emb_df, ["LM_pt", "TM_pt", "LM_eta", "TM_eta", "LM_phi", "TM_phi", "LM_m", "TM_m"])

dr = calculate_dr(emb_df, "muon", filter=None)
emb_df_matched, muon_id_matched, dr_matched = apply_genmatching(dr.copy(), emb_df_for_matching.copy(deep=True), "muon")


print("Genmatching applied")


# Creating plots indicating performance of matching
if create_match_plots:
    dphi_1 = subtract_columns(emb_df["phi_1"], data_df["phi_1"], "phi_1")
    deta_1 = subtract_columns(emb_df["eta_1"], data_df["eta_1"], "eta_1")
    dr_1 = np.sqrt(np.square(dphi_1) + np.square(deta_1))
    dphi_2 = subtract_columns(emb_df["phi_2"], data_df["phi_2"], "phi_2")
    deta_2 = subtract_columns(emb_df["eta_2"], data_df["eta_2"], "eta_2")
    dr_2 = np.sqrt(np.square(dphi_2) + np.square(deta_2))

    #dr between muon1|2 data and muon1|2 embedding

    ax = nq_comparison({"Leading µ":dr_1, "Trailing µ":dr_2}, 30, r"$\delta r_\text{µ, unmatched}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"muon_dr_unmatched.png"))
    plt.close()

    #dr between l|m muon data and l|m muon embedding
    ax = nq_comparison({"Leading µ":dr_matched[:,0], "Trailing µ":dr_matched[:,1]}, 30, r"$\delta r_\text{µ, matched}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"muon_dr_matched.png"))
    plt.close()

    #frequency of muon id to be used as l|m muon
    ax = match_plot(muon_id_matched, "ID of closest µ")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"muon_id_matched.png"))
    plt.close()


########################################################################################################################################################################
# Adding mvis and ptvis
########################################################################################################################################################################

data_df["m_vis"], data_df["pt_vis"] = get_z_m_pt(data_df)
emb_df_matched["m_vis"], emb_df_matched["pt_vis"] = get_z_m_pt(emb_df_matched)

print("Added m_vis and pt_vis")



########################################################################################################################################################################
# Removing muon jets
########################################################################################################################################################################

dr_cut = 0.35

dr1 = calculate_dr(data_df, "filter", filter=None)
data_df = remove_muon_jets(data_df, dr1, dr_cut)
dr2 = calculate_dr(emb_df, "filter", filter=None)
emb_df = remove_muon_jets(emb_df, dr2, dr_cut)


data_df = assert_object_validity(data_df)
emb_df = assert_object_validity(emb_df)

data_df = compactify_objects(data_df)
emb_df = compactify_objects(emb_df)

data_df, emb_df = create_concordant_subsets(data_df, emb_df)

if create_match_plots:
    # Creating plots indicating performance of muon removal
    dr1 = dr1.flatten()
    dr2 = dr2.flatten()

    ax = nq_comparison({"Data":dr1, "Emb":dr2}, np.linspace(0,10*dr_cut, 30), r"$\delta r_\text{µ jet, uncleaned}$")
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.vlines(dr_cut, 0, ymax, colors="black", linestyles="dashed")
    plt.savefig(os.path.join(match_plot_path, f"mujet_dr.png"))
    plt.close()

 

    # Creating plots indicating performance of muon removal

    njet_emb = count_n_objects(emb_df, "Jet_eta_")
    njet_data = count_n_objects(data_df, "Jet_eta_")
    max_njet = max([get_n_occurence(data_df, "Jet_eta_"), get_n_occurence(emb_df, "Jet_eta_")])

    nmu_emb = count_n_objects(emb_df, "eta_")
    nmu_data = count_n_objects(data_df, "eta_")
    max_nmu = max([get_n_occurence(data_df, "eta_"), get_n_occurence(emb_df, "eta_")])

    njet_delta = njet_data - njet_emb
    nmu_delta = nmu_data - nmu_emb
    max_delta = max([np.amax(njet_delta), np.amax(nmu_delta)])
    min_delta = min([np.amin(njet_delta), np.amin(nmu_delta)])

    
    ax = control_plot(njet_data, njet_emb, np.arange(-0.5, max_njet+0.5, 1), r"$n_\text{jets}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_jet_02_post_filter.png"))
    plt.close()

    
    ax = control_plot(nmu_data, nmu_emb, np.arange(-0.5, max_nmu+0.5, 1), r"$n_\text{µ}$ in embedding and data", None)
    ax[0].set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_mu_02_post_filter.png"))
    plt.close()

    ax = nq_comparison({"Mu":nmu_delta, "Jet":njet_delta}, np.arange(min_delta-0.5, max_delta+0.5, 1), r"$n_\text{x, data}-n_\text{x, emb}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"n_delta_02.png"))
    plt.close()



########################################################################################################################################################################
# Matching jets
########################################################################################################################################################################

data_df, emb_df_matched = prepare_jet_matching(data_df, emb_df_matched)
dr = calculate_dr(emb_df_matched, "jet", filter=None)

emb_df_for_matching = get_matching_df(emb_df_matched, ["LJ_pt", "TJ_pt", "LJ_eta", "TJ_eta", "LJ_phi", "TJ_phi", "LJ_m", "TJ_m"])
emb_df_matched, jet_id_matched, jet_dr_matched = apply_genmatching(dr.copy(), emb_df_for_matching, "jet")

print("Jets matched")


# Creating plots indicating performance of jet matching
if create_match_plots:
    dphi_1 = subtract_columns(emb_df_matched["Jet_phi_1"], data_df["LJ_phi"], "phi_1")
    deta_1 = subtract_columns(emb_df_matched["Jet_eta_1"], data_df["LJ_eta"], "eta_1")
    dr_1 = np.sqrt(np.square(dphi_1) + np.square(deta_1))
    dphi_2 = subtract_columns(emb_df_matched["Jet_phi_2"], data_df["TJ_phi"], "phi_2")
    deta_2 = subtract_columns(emb_df_matched["Jet_eta_2"], data_df["TJ_eta"], "eta_2")
    dr_2 = np.sqrt(np.square(dphi_2) + np.square(deta_2))

    #dr between muon1|2 data and muon1|2 embedding
    ax = nq_comparison({"Leadin jet":dr_1, "Trailing jet":dr_2}, 30, r"$\delta r_\text{Jet, unmatched}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"jet_dr_unmatched.png"))
    plt.close()

    #dr between l|m muon data and l|m muon embedding
    ax = nq_comparison({"Leadin jet":jet_dr_matched[:,0], "Trailing jet":jet_dr_matched[:,1]}, 30, r"$\delta r_\text{Jet, matched}$")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"jet_dr_matched.png"))
    plt.close()


    #frequency of muon id to be used as l|m muon
    ax = match_plot(jet_id_matched, "ID of closest jet")
    ax.set_yscale("log")
    plt.savefig(os.path.join(match_plot_path, f"jet_id_matched.png"))
    plt.close()



########################################################################################################################################################################
# Storing datarames in hdf store
########################################################################################################################################################################
initialize_dir(output_path)

store = pd.HDFStore(os.path.join(output_path, "converted_nanoaod.h5"), 'w')  
store.put("data_df", data_df, index=False)
store.put("emb_df_matched", emb_df_matched, index=False)
# store.put("emb_df_matched_filtered", emb_df_matched_filtered, index=False)
store.close()

print("Data stored in hdf store")

#thats how you import the data
# hdf_path = "./data/converted/converted_nanoaod.h5"
# data_df = pd.read_hdf(hdf_path, "data_df")
# emb_df = pd.read_hdf(hdf_path, "emb_df")

