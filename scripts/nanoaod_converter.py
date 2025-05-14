import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from importer import nanoaod_to_dataframe, compare_cells, get_z_m_pt, verify_events, initialize_dir
from genmatching import calculate_dr, apply_genmatching, detect_changes, get_filter_list

data_path = "./data/2022G-nanoaod/2022G-data.root"
emb_path = "./data/2022G-nanoaod/2022G-emb.root"

output_path = "./data/converted"

initialize_dir(output_path)

print("Directory initialized")

quantities = [
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

print("Loading data")


data_df = nanoaod_to_dataframe(data_path=data_path, quantities=quantities)
emb_df = nanoaod_to_dataframe(data_path=emb_path, quantities=quantities)

print("Data loaded")


data_df = data_df.sort_values(by=["run", "lumi", "event"], ignore_index=True)
emb_df = emb_df.sort_values(by=["run", "lumi", "event"], ignore_index=True)


verify_events(data_df, emb_df)

print("Data ok")


data_df["m_vis"], data_df["pt_vis"] = get_z_m_pt(data_df)
emb_df["m_vis"], emb_df["pt_vis"] = get_z_m_pt(emb_df)
# data_df["pt_vis"] = data_df["pt_1"] + data_df["pt_2"]
# emb_df["pt_vis"] = emb_df["pt_1"] + emb_df["pt_2"]

print("Added m_vis and pt_vis")


dr = calculate_dr(data_df, emb_df, 2, 5, filter=None)
emb_df_matched = apply_genmatching(dr.copy(), emb_df.copy(deep=True), ["phi", "pt", "eta", "m"])


filter_list = get_filter_list()

dr_filtered = calculate_dr(data_df, emb_df, 2, 5, filter=filter_list)
emb_df_matched_filtered = apply_genmatching(dr_filtered.copy(), emb_df.copy(deep=True), ["phi", "pt", "eta", "m"])

print("Genmatching applied")

detect_changes(emb_df, emb_df_matched, ["phi_1", "pt_1", "eta_1"])

detect_changes(emb_df, emb_df_matched_filtered, ["phi_1", "pt_1", "eta_1"])


store = pd.HDFStore(os.path.join(output_path, "converted_nanoaod.h5"), 'w')  
store.put("data_df", data_df, index=False)
store.put("emb_df", emb_df, index=False)
store.put("emb_df_matched", emb_df_matched, index=False)
store.put("emb_df_matched_filtered", emb_df_matched_filtered, index=False)
store.close()

print("Data stored in hdf store")


# hdf_path = "./data/converted/converted_nanoaod.h5"
# data_df = pd.read_hdf(hdf_path, "data_df")
# emb_df = pd.read_hdf(hdf_path, "emb_df")

