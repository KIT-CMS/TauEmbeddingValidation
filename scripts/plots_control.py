import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from source.plotting import control_plot, nq_comparison
from source.importer import initialize_dir
from source.helper import verify_events, set_working_dir, set_plt_fonts


########################################################################################################################################################################
# Paths for input and output
########################################################################################################################################################################

hdf_path = "./output_nofsr_nounmobj/data/converted_nanoaod.h5"
control_output_path = "./output_nofsr_nounmobj/control_plots/"
# hdf_path = "./output_nounmobj/data/converted_nanoaod.h5"
# control_output_path = "./output_nounmobj/control_plots/"

initialize_dir(control_output_path, ["default", "custom"])

set_working_dir()
set_plt_fonts()

print("Initialized directories")

########################################################################################################################################################################
# Instructions for plots
########################################################################################################################################################################

nbins = 35

plotting_instructions = [
    {"col":"LJ_eta",           
        "bins":np.linspace(-4, 4, nbins),          
        "title":r"Leading jet $\eta$",                 
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_m",          
        "bins":np.linspace(0, 45, nbins),         
        "title":r"Leading jet mass/ GeV",           
        "dy":1,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_phi",           
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"Leading jet $\phi$",              
        "dy":0.5,
        "ylog":True,    
        "xlog":False}, 
    {"col":"LJ_pt",           
        "bins":np.linspace(0, 250, nbins),      
        "title":r"Leading jet $p_\text{T}$/ GeV",              
        "dy":0.75,
        "ylog":True,    
        "xlog":False}, 
    {"col":"LM_eta",             
        "bins":np.linspace(-2.5, 2.5, nbins),      
        "title":r"$\eta_\text{µ1}$",             
        "dy":0.3,
        "ylog":True,    
        "xlog":False},   
    {"col":"LM_phi",              
        "bins":np.linspace(-3.5, 3.5, nbins),         
        "title":r"$\phi_\text{µ1}$",  
        "dy":0.3,
        "ylog":True,    
        "xlog":False},
    {"col":"LM_pt",              
        "bins":np.linspace(0, 250, nbins),         
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
        "title":r"Angle $\phi$ of missing momentum",           
        "dy":0.75,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 150, nbins),         
        "title":r"Missing $p_{T}$/ GeV",           
        "dy":0.75,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_eta",           
        "bins":np.linspace(-5, 5, nbins),          
        "title":r"Subleading jet $\eta$",                 
        "dy":0.5,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_m",          
        "bins":np.linspace(0, 30, nbins),         
        "title":r"Subleading jet mass/ GeV",           
        "dy":1,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_phi",           
        "bins":np.linspace(-3.5, 3.5, nbins),      
        "title":r"Subleading jet $\phi$",              
        "dy":0.5,
        "ylog":True,    
        "xlog":False}, 
    {"col":"TJ_pt",           
        "bins":np.linspace(0, 250, nbins),      
        "title":r"Subleading jet $p_\text{T}$/ GeV",              
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
    # {"col":"LE_eta",           
    #     "bins":np.linspace(-5, 5, nbins),          
    #     "title":r"Leading electron $\eta$",                 
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False},
    # {"col":"LE_phi",           
    #     "bins":np.linspace(-3.5, 3.5, nbins),      
    #     "title":r"Leading electron $\phi$",              
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False}, 
    # {"col":"LE_pt",           
    #     "bins":np.linspace(0, 125, nbins),      
    #     "title":r"Leading electron $p_\text{T}$",              
    #     "dy":0.75,
    #     "ylog":True,    
    #     "xlog":False},  
    # {"col":"TE_eta",           
    #     "bins":np.linspace(-5, 5, nbins),          
    #     "title":r"Subleading electron $\eta$",                 
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False},
    # {"col":"TE_phi",           
    #     "bins":np.linspace(-3.5, 3.5, nbins),      
    #     "title":r"Subleading electron $\phi$",              
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False}, 
    # {"col":"TE_pt",           
    #     "bins":np.linspace(0, 30, nbins),      
    #     "title":r"Subleading electron $p_\text{T}$",              
    #     "dy":0.75,
    #     "ylog":True,    
    #     "xlog":False},  
    # {"col":"LP_eta",           
    #     "bins":np.linspace(-5, 5, nbins),          
    #     "title":r"Leading photon $\eta$",                 
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False},
    # {"col":"LP_phi",           
    #     "bins":np.linspace(-3.5, 3.5, nbins),      
    #     "title":r"Leading photon $\phi$",              
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False}, 
    # {"col":"LP_pt",           
    #     "bins":np.linspace(0, 200, nbins),      
    #     "title":r"Leading photon $p_\text{T}$",              
    #     "dy":0.75,
    #     "ylog":True,    
    #     "xlog":False},  
    # {"col":"TP_eta",           
    #     "bins":np.linspace(-5, 5, nbins),          
    #     "title":r"Subleading photon $\eta$",                 
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False},
    # {"col":"TP_phi",           
    #     "bins":np.linspace(-3.5, 3.5, nbins),      
    #     "title":r"Subleading photon $\phi$",              
    #     "dy":0.5,
    #     "ylog":True,    
    #     "xlog":False}, 
    # {"col":"TP_pt",           
    #     "bins":np.linspace(0, 80, nbins),      
    #     "title":r"Subleading photon $p_\text{T}$",              
    #     "dy":0.75,
    #     "ylog":True,    
    #     "xlog":False},  
]

########################################################################################################################################################################
# Reading data
########################################################################################################################################################################

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df")

verify_events(data_df, emb_df)

print("Data loaded and verified")



########################################################################################################################################################################
# Basic control plots comparing data and matched embedding
########################################################################################################################################################################
print(len(data_df))
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

        print("\n", title)

        ax = control_plot(data_df[col], emb_df[col], bins, title, dy)

        if quantity["xlog"]:
            ax[0].set_xscale("log")
        if quantity["ylog"]:
            ax[0].set_yscale("log")
        
        plt.savefig(os.path.join(control_output_path, mode, f"control_{col}.png"))
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
#         col1 = emb_df[col]
#         col2 = emb_df[col]

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