import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from importer import nanoaod_to_dataframe, compare_cells, get_z_m_pt, verify_events, initialize_dir, create_concordant_subsets, copy_columns_from_to
from genmatching import calculate_dr, apply_genmatching, detect_changes, get_filter_list

# data_path = "./data/2022G-nanoaod/2022G-data.root"
# emb_path = "./data/2022G-nanoaod/2022G-emb.root"
data_path = "./data/2022G-nanoaod_gen/2022G-data.root"
emb_path = "./data/2022G-nanoaod_gen/2022G-emb_gen.root"

output_path = "./data/converted"

initialize_dir(output_path)

print("Directory initialized")

data_quantities = [
    {"key":"PuppiMET_pt",       "target":"PuppiMET_pt",     "expand":False},
    {"key":"PuppiMET_phi",      "target":"PuppiMET_phi",    "expand":False},
    {"key":"PuppiMET_sumEt",    "target":"PuppiMET_sumEt",  "expand":False},
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



print("Loading data")


data_df = nanoaod_to_dataframe(data_path=data_path, quantities=data_quantities)
emb_df = nanoaod_to_dataframe(data_path=emb_path, quantities=emb_quantities)

print("Data loaded")

data_df, emb_df = create_concordant_subsets(data_df, emb_df)

verify_events(data_df, emb_df)

print("Data ok")

selection_q_converted = [element["target"] for element in selection_q]
data_df, emb_df = copy_columns_from_to(emb_df, data_df, selection_q_converted)

print("Copied:", selection_q_converted)



dr = calculate_dr(data_df, 5, filter=None)
data_df_matched = apply_genmatching(dr.copy(), data_df.copy(deep=True))
dr = calculate_dr(emb_df, 5, filter=None)
emb_df_matched = apply_genmatching(dr.copy(), emb_df.copy(deep=True))

# filter_list = get_filter_list()

# dr = calculate_dr(data_df, 5, filter=filter_list)
# data_df_matched_filtered = apply_genmatching(dr.copy(), data_df.copy(deep=True))
# dr = calculate_dr(emb_df, 5, filter=filter_list)
# emb_df_matched_filtered = apply_genmatching(dr.copy(), emb_df.copy(deep=True))

print("Genmatching applied")

data_df["m_vis"], data_df["pt_vis"] = get_z_m_pt(data_df)
data_df_matched["m_vis"], data_df_matched["pt_vis"] = get_z_m_pt(data_df_matched)
# data_df_matched_filtered["m_vis"], data_df_matched_filtered["pt_vis"] = get_z_m_pt(data_df_matched_filtered, data=True)
emb_df_matched["m_vis"], emb_df_matched["pt_vis"] = get_z_m_pt(emb_df_matched)
# emb_df_matched_filtered["m_vis"], emb_df_matched_filtered["pt_vis"] = get_z_m_pt(emb_df_matched_filtered, data=False)

print("Added m_vis and pt_vis")

store = pd.HDFStore(os.path.join(output_path, "converted_nanoaod.h5"), 'w')  
# store.put("data_df", data_df, index=False)
# store.put("emb_df", emb_df, index=False)
store.put("data_df", data_df, index=False)
store.put("data_df_matched", data_df, index=False)
store.put("emb_df_matched", emb_df_matched, index=False)
# store.put("data_df_matched_filtered", data_df_matched_filtered, index=False)
# store.put("emb_df_matched_filtered", emb_df_matched_filtered, index=False)
store.close()

print("Data stored in hdf store")


# hdf_path = "./data/converted/converted_nanoaod.h5"
# data_df = pd.read_hdf(hdf_path, "data_df")
# emb_df = pd.read_hdf(hdf_path, "emb_df")

