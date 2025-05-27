import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plotting import control_plot, nq_comparison, q_comparison
from importer import verify_events, initialize_dir
from genmatching import detect_changes

hdf_path = "./data/converted/converted_nanoaod.h5"
comparison_output_path = "./output/control_plots/comparison"
control_output_path = "./output/control_plots/normal"


initialize_dir(comparison_output_path, ["default", "custom"])
initialize_dir(control_output_path, ["default", "custom"])

print("Initialized directories")


data_df = pd.read_hdf(hdf_path, "data_df")
emb_df_matched = pd.read_hdf(hdf_path, "emb_df_matched")
emb_df_matched_filtered = pd.read_hdf(hdf_path, "emb_df_matched_filtered")

verify_events(data_df, emb_df_matched)

print("Data loaded and verified")

nbins = 35

plotting_instructions = [
    {"col":"Jet_eta",           
        "bins":np.linspace(-5, 5, nbins),          
        "title":r"LJet eta",                 
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "bins":np.linspace(0, 100, nbins),         
        "title":r"LJet mass/ GeV",           
        "ylog":True,    "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"LJet $\phi$",              
        "ylog":True,    
        "xlog":False}, 
    {"col":"LM_eta",             
        "bins":np.linspace(-2.5, 2.5, nbins),      
        "title":r"$\eta_\text{µ1}$",             
        "ylog":True,    
        "xlog":False},   
    {"col":"LM_phi",              
        "bins":np.linspace(-3.5, 3.5, nbins),         
        "title":r"$\phi_\text{µ1}$",  
        "ylog":True,    
        "xlog":False},
    {"col":"LM_pt",              
        "bins":np.linspace(0, 300, nbins),         
        "title":r"$p_\text{T, µ1}$/ GeV",  
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(0, 200, nbins),         
        "title":r"$m_\text{µµ}$/ GeV",             
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(0, 200, nbins),        
        "title":r"$p_\text{T, µµ}$/ GeV",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"Missing $p_{T, \phi}$",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 150, nbins),         
        "title":r"Missing $p_{T}$ / GeV",           
        "ylog":True,    
        "xlog":False},
    # {"col":"PuppiMET_sumEt",    
    #     "bins":np.linspace(0, 700, nbins),         
    #     "title":r"Missing $E_\text{T}$ / GeV",      
    #     "ylog":True,    
    #     "xlog":False},
    {"col":"TM_eta",             
        "bins":np.linspace(-2.5, 2.5, nbins),      
        "title":r"$\eta_\text{µ2}$",             
        "ylog":True,    
        "xlog":False},   
    {"col":"TM_phi",              
        "bins":np.linspace(-3.5, 3.5, nbins),         
        "title":r"$\phi_\text{µ2}$",  
        "ylog":True,    
        "xlog":False},
    {"col":"TM_pt",              
        "bins":np.linspace(0, 300, nbins),         
        "title":r"$p_\text{T, µ2}$/ GeV",  
        "ylog":True,    
        "xlog":False},
]

# for quantity in plotting_instructions:
#     col = quantity["col"]
#     total_l = len(emb_df_matched)
#     col_l = len(emb_df_matched[emb_df_matched[col].notna()])
#     print(col, "\t", col_l- total_l)

# exit()
#data vs embedding 

for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        if mode == "default":
            bins = nbins
        elif mode == "custom":
            bins = quantity["bins"]

        col = quantity["col"]
        title = quantity["title"]

        ax = control_plot(data_df[col], emb_df_matched[col], bins, title)

        if quantity["xlog"]:
            ax[0].set_xscale("log")
        if quantity["ylog"]:
            ax[0].set_yscale("log")
        
        plt.savefig(os.path.join(control_output_path, mode, f"{col}.png"))
        plt.close()

print("Created control plots ")


# comparison

# for quantity in plotting_instructions:
#     col = quantity["col"]
#     bins = quantity["bins"]
#     title = quantity["title"]

#     col1 = data_df[col]
#     col2 = emb_df[col]
#     ax = q_comparison(col1, col2, bins, "Data", "Emb", title)
#     if quantity["xlog"]:
#         ax[0].set_xscale("log")
#     if quantity["ylog"]:
#         ax[0].set_yscale("log")
    
#     plt.savefig(os.path.join(comparison_output_path, "custom", f"{col}.png"))
#     plt.close()


for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        if mode == "default":
            bins = nbins
        elif mode == "custom":
            bins = quantity["bins"]

        col = quantity["col"]
        title = quantity["title"]

        col0 = data_df[col]
        col1 = emb_df_matched_filtered[col]
        col2 = emb_df_matched[col]

        q_dict = {
            "Emb (matched + filtered)": col1,
            "Emb (matched)": col2,
        }
        ax = nq_comparison(q_dict, bins=bins, title=title, data=col0)
        
        if quantity["xlog"]:
            ax.set_xscale("log")
        if quantity["ylog"]:
            ax.set_yscale("log")
        
        plt.savefig(os.path.join(comparison_output_path, mode, f"{col}.png"))
        plt.close()

print("Created comparison plots")

print("Plotting finished")