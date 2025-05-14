import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plotting import control_plot, nq_comparison
from importer import verify_events, initialize_dir
from genmatching import detect_changes

hdf_path = "./data/converted/converted_nanoaod.h5"
comparison_output_path = "./output/control_plots/comparison"


initialize_dir(comparison_output_path, ["default", "custom"])

print("Initialized directories")


data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df")
emb_df_matched = pd.read_hdf(hdf_path, "emb_df_matched")
emb_df_matched_filtered = pd.read_hdf(hdf_path, "emb_df_matched_filtered")

verify_events(data_df, emb_df, emb_df_matched, emb_df_matched_filtered)

print("Data loaded and verified")

detect_changes(emb_df, emb_df_matched, ["phi_1", "pt_1", "eta_1"])
detect_changes(emb_df, emb_df_matched_filtered, ["phi_1", "pt_1", "eta_1"])



plotting_instructions = [
    {"col":"eta_1",             
        "bins":np.linspace(-2.5, 2.5, 25),      
        "title":r"$\eta_\text{µ1}$",             
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_eta",           
        "bins":np.linspace(-5, 5, 25),          
        "title":r"LJet eta",                 
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "bins":np.linspace(0, 100, 25),         
        "title":r"LJet mass/ GeV",           
        "ylog":True,    "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-4, 4, 25),      
        "title":r"LJet $\phi$",              
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(0, 200, 25),         
        "title":r"$m_\text{vis}$/ GeV",             
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "bins":np.linspace(-4, 4, 25),          
        "title":r"$\phi_\text{µ1}$",             
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "bins":np.linspace(0, 300, 25),         
        "title":r"$p_\text{T, µ1}$/ GeV",  
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(0, 200, 25),        
        "title":r"$p_\text{T vis}$/ GeV",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-4, 4, 25),      
        "title":r"Missing $p_{T, \phi}$",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 150, 25),         
        "title":r"Missing $p_{T}$ / GeV",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_sumEt",    
        "bins":np.linspace(0, 700, 25),         
        "title":r"Missing $E_\text{T}$ / GeV",      
        "ylog":True,    
        "xlog":False},
]

# #data vs embedding 
# for quantity in plotting_instructions:
#     col = quantity["col"]
#     bins = quantity["bins"]
#     title = quantity["title"]

#     ax = control_plot(data_df[col], emb_df_matched[col], bins, title)

#     if quantity["xlog"]:
#         ax[0].set_xscale("log")
#     if quantity["ylog"]:
#         ax[0].set_yscale("log")
    
#     plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
#     plt.close()

# print("Created matched control plots with custom binning")


#comparison

# for quantity in plotting_instructions:
#     col = quantity["col"]
#     bins = quantity["bins"]
#     title = quantity["title"]

#     col1 = emb_df_matched[col]
#     col2 = emb_df[col]
#     ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
#     if quantity["xlog"]:
#         ax[0].set_xscale("log")
#     if quantity["ylog"]:
#         ax[0].set_yscale("log")
    
#     plt.savefig(os.path.join(matched_comparison_output_path, "custom", f"{col}.png"))
#     plt.close()

# print("Created matched comparison plots with custom binning")

for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        if mode == "default":
            bins = 25
        elif mode == "custom":
            bins = quantity["bins"]

        col = quantity["col"]
        title = quantity["title"]

        col0 = data_df[col]
        col1 = emb_df[col]
        col2 = emb_df_matched[col]
        col3 = emb_df_matched_filtered[col]

        q_dict = {
            "Emb (raw)": col1,
            "Emb (matched)": col2,
            "Emb (matched, filtered)": col3
        }
        ax = nq_comparison(q_dict, bins=bins, title=title, data=col0)
        
        if quantity["xlog"]:
            ax.set_xscale("log")
        if quantity["ylog"]:
            ax.set_yscale("log")
        
        plt.savefig(os.path.join(comparison_output_path, mode, f"{col}.png"))
        plt.close()

print("Created triple comparison plots")

print("Plotting finished")