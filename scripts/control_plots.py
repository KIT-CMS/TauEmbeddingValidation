import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from source.plotting import control_plot, nq_comparison
from source.importer import initialize_dir
from source.helper import verify_events, set_working_dir


########################################################################################################################################################################
# Paths for input and output
########################################################################################################################################################################

hdf_path = "./data/converted/converted_nanoaod.h5"
# comparison_output_path = "./output/control_plots/comparison"
control_output_path = "./output/control_plots/"

initialize_dir(control_output_path, ["default", "custom"])

set_working_dir()

print("Initialized directories")

########################################################################################################################################################################
# Instructions for plots
########################################################################################################################################################################

nbins = 35

plotting_instructions = [
    {"col":"LJ_eta",           
        "bins":np.linspace(-5, 5, nbins),          
        "title":r"LJet $\eta$",                 
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_m",          
        "bins":np.linspace(0, 45, nbins),         
        "title":r"LJet mass/ GeV",           
        "dy":1,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_phi",           
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"LJet $\phi$",              
        "dy":0.5,
        "ylog":True,    
        "xlog":False}, 
    {"col":"LJ_pt",           
        "bins":np.linspace(0, 250, nbins),      
        "title":r"LJet $p_\text{T}$",              
        "dy":0.75,
        "ylog":True,    
        "xlog":False},  
    {"col":"LM_phi",              
        "bins":np.linspace(-3.5, 3.5, nbins),         
        "title":r"$\phi_\text{µ1}$",  
        "dy":0.3,
        "ylog":True,    
        "xlog":False},
    {"col":"LM_pt",              
        "bins":np.linspace(0, 200, nbins),         
        "title":r"$p_\text{T, µ1}$/ GeV",  
        "dy":0.75,
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(0, 200, nbins),         
        "title":r"$m_\text{µµ}$/ GeV",             
        "dy":0.3,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(0, 180, nbins),        
        "title":r"$p_\text{T, µµ}$/ GeV",           
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"Missing $p_{T, \phi}$",           
        "dy":0.75,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 150, nbins),         
        "title":r"Missing $p_{T}$ / GeV",           
        "dy":0.75,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_eta",           
        "bins":np.linspace(-5, 5, nbins),          
        "title":r"TJet $\eta$",                 
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_m",          
        "bins":np.linspace(0, 30, nbins),         
        "title":r"TJet mass/ GeV",           
        "dy":1,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_phi",           
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"TJet $\phi$",              
        "dy":0.5,
        "ylog":True,    
        "xlog":False}, 
    {"col":"TJ_pt",           
        "bins":np.linspace(0, 200, nbins),      
        "title":r"TJet $p_\text{T}$",              
        "dy":0.75,
        "ylog":True,    
        "xlog":False},  
    {"col":"TM_eta",             
        "bins":np.linspace(-2.5, 2.5, nbins),      
        "title":r"$\eta_\text{µ2}$",             
        "dy":0.3,
        "ylog":True,    
        "xlog":False},   
    {"col":"TM_phi",              
        "bins":np.linspace(-3.5, 3.5, nbins),         
        "title":r"$\phi_\text{µ2}$",  
        "dy":0.25,
        "ylog":True,    
        "xlog":False},
    {"col":"TM_pt",              
        "bins":np.linspace(0, 150, nbins),         
        "title":r"$p_\text{T, µ2}$/ GeV",  
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
]

########################################################################################################################################################################
# Reading data
########################################################################################################################################################################

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df_matched = pd.read_hdf(hdf_path, "emb_df_matched")
# emb_df_matched_filtered = pd.read_hdf(hdf_path, "emb_df_matched_filtered")

verify_events(data_df, emb_df_matched)

print("Data loaded and verified")



########################################################################################################################################################################
# Basic control plots comparing data and matched embedding
########################################################################################################################################################################

for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        if mode == "default":
            bins = nbins
            dy = None
        elif mode == "custom":
            bins = quantity["bins"]
            dy = quantity["dy"]

        col = quantity["col"]
        title = quantity["title"]

        ax = control_plot(data_df[col], emb_df_matched[col], bins, title, dy)

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


########################################################################################################################################################################
# Step histograms comparing more than two columns
########################################################################################################################################################################


# for quantity in plotting_instructions:
#     for mode in ["custom", "default"]:
#         if mode == "default":
#             bins = nbins
#         elif mode == "custom":
#             bins = quantity["bins"]

#         col = quantity["col"]
#         title = quantity["title"]

#         col0 = data_df[col]
#         col1 = emb_df_matched_filtered[col]
#         col2 = emb_df_matched[col]

#         q_dict = {
#             "Emb (matched + filtered)": col1,
#             "Emb (matched)": col2,
#         }
#         ax = nq_comparison(q_dict, bins=bins, title=title, data=col0)
        
#         if quantity["xlog"]:
#             ax.set_xscale("log")
#         if quantity["ylog"]:
#             ax.set_yscale("log")
        
#         plt.savefig(os.path.join(comparison_output_path, mode, f"{col}.png"))
#         plt.close()

# print("Created comparison plots")

print("Plotting finished")