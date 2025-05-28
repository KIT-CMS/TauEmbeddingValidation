import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plotting import histogram
from importer import initialize_dir
from genmatching import subtract_columns
from helper import divide_columns, verify_events

hdf_path = "./data/converted/converted_nanoaod.h5"

# comparison_output_path = "./output/diff_plots/comparison"
abs_output_path = "./output/diff_plots/abs_diff"
rel_output_path = "./output/diff_plots/rel_diff"

# initialize_dir(comparison_output_path, ["default", "custom"])
initialize_dir(abs_output_path, ["default", "custom"])
initialize_dir(rel_output_path, ["default", "custom"])


print("Initialized directories")

data_df = pd.read_hdf(hdf_path, "data_df")
# data_df_matched = pd.read_hdf(hdf_path, "data_df_matched")
emb_df_matched = pd.read_hdf(hdf_path, "emb_df_matched")

verify_events(data_df, emb_df_matched)

print("Data loaded and verified")


nbins = 35

plotting_instructions = [

    {"col":"LJ_eta",           
        "bins":np.linspace(0, 5, nbins),          
        "rel_bins":np.linspace(0, 5, nbins),          
        "title":r"$\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$",                 
        "rel_title":r"$(\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$) / $\eta_\text{LJet, data}$",    
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_m",          
        "bins":np.linspace(0, 50, nbins),         
        "rel_bins":np.linspace(0, 75, nbins),         
        "title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data}) / GeV$",           
        "rel_title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data})$ / $m_\text{LJet, data})$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"LJ_phi",           
        "bins":np.linspace(0, 3.5, nbins),      
        "rel_bins":np.linspace(0, 0.1, nbins),      
        "title":r"$\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data}$",              
        "rel_title":r"$(\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data})$ / $\phi_\text{LJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_eta",           
        "bins":np.linspace(0, 5, nbins),          
        "rel_bins":np.linspace(0, 5, nbins),          
        "title":r"$\eta_\text{TJet, emb}$ - $\eta_\text{LTJetJet, data}$",                 
        "rel_title":r"$(\eta_\text{TJet, emb}$ - $\eta_\text{TJet, data}$) / $\eta_\text{TJet, data}$",    
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_m",          
        "bins":np.linspace(0, 50, nbins),         
        "rel_bins":np.linspace(0, 75, nbins),         
        "title":r"($m_\text{TJet, emb}$ - $m_\text{TJet, data}) / GeV$",           
        "rel_title":r"($m_\text{TJet, emb}$ - $m_\text{TJet, data})$ / $m_\text{TJet, data})$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TJ_phi",           
        "bins":np.linspace(0, 3.5, nbins),      
        "rel_bins":np.linspace(0, 0.1, nbins),      
        "title":r"$\phi_\text{TJet, emb}$ - $\phi_\text{TJet, data}$",              
        "rel_title":r"$(\phi_\text{TJet, emb}$ - $\phi_\text{TJet, data})$ / $\phi_\text{TJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"LM_eta",             
        "bins":np.linspace(0, 2.5, nbins),      
        "rel_bins":np.linspace(0, 0.1, nbins),      
        "title":r"$\eta_\text{µ1, emb}$ - $\eta_\text{µ1, data}$",             
        "rel_title":r"$(\eta_\text{µ1, emb}$ - $\eta_\text{µ1, data}$) / $\eta_\text{µ1, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"LM_phi",             
        "bins":np.linspace(0, 3.5, nbins),          
        "rel_bins":np.linspace(0, 0.1, nbins),          
        "title":r"$\phi_\text{µ1, emb}$ - $\phi_\text{µ1, data}$",             
        "rel_title":r"($\phi_\text{µ1, emb}$ - $\phi_\text{µ1, data})$ / $\phi_\text{µ1, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"LM_pt",              
        "bins":np.linspace(0, 150, nbins),         
        "rel_bins":np.linspace(0, 0.5, nbins),         
        "title":r"$p_\text{T, µ1, emb}$ - $p_\text{T, µ1, data}$",  
        "rel_title":r"$(p_\text{T, µ1, emb}$ - $p_\text{T, µ1, data})$ / $p_\text{T, µ1, data}$",  
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(0, 100, nbins),         
        "rel_bins":np.linspace(0, 1.5, nbins),         
        "title":r"$m_\text{µµ, emb}$ - $m_\text{µµ, data})$",             
        "rel_title":r"$(m_\text{µµ, emb}$ - $m_\text{µµ, data})$ / $m_\text{µµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(0, 500, nbins),        
        "rel_bins":np.linspace(0, 7, nbins),        
        "title":r"$p_\text{T µµ, emb}$ - $p_\text{T µµ, data}$",           
        "rel_title":r"$(p_\text{T µµ, emb}$ - $p_\text{T µµ, data})$ / $p_\text{T µµ, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(0, 3.5, nbins),      
        "rel_bins":np.linspace(0, 3.5, nbins),      
        "title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / GeV",           
        "rel_title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / $E_\text{\phi miss, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 110, nbins),        
        "rel_bins":np.linspace(0, 10, nbins),        
        "title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})/ GeV$",           
        "rel_title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})$ / $p_\text{T miss, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TM_eta",             
        "bins":np.linspace(0, 2.5, nbins),    
        "rel_bins":np.linspace(0, 0.1, nbins),    
        "title":r"$\eta_\text{µ2, emb}$ - $\eta_\text{µ2, data}$",             
        "rel_title":r"$(\eta_\text{µ2, emb}$ - $\eta_\text{µ2, data}$) / $\eta_\text{µ2, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TM_phi",             
        "bins":np.linspace(0, 3.5, nbins),    
        "rel_bins":np.linspace(0, 0.1, nbins),       
        "title":r"$\phi_\text{µ2, emb}$ - $\phi_\text{µ2, data}$",             
        "rel_title":r"($\phi_\text{µ2, emb}$ - $\phi_\text{µ2, data})$ / $\phi_\text{µ2, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"TM_pt",              
        "bins":np.linspace(0, 150, nbins),  
        "rel_bins":np.linspace(0, 0.5, nbins),        
        "title":r"$p_\text{T, µ2, emb}$ - $p_\text{T, µ2, data}$",  
        "rel_title":r"$(p_\text{T, µ2, emb}$ - $p_\text{T, µ2, data})$ / $p_\text{T, µ2, data}$",  
        "relative":False,
        "ylog":True,    
        "xlog":False},
]


#comparison plots

# for quantity in plotting_instructions:
#     for mode in ["custom", "default"]:
#         col = quantity["col"]
#         relative = False

#         if mode == "default":
#             bins = nbins
#         elif mode == "custom" and relative:
#             bins = quantity["rel_bins"]
#         elif mode == "custom" and not relative:
#             bins = quantity["bins"]


#         col1 = subtract_columns(data_df[col], data_df_matched[col], col)
#         col2 = subtract_columns(data_df[col], emb_df_matched[col], col)
#         col3 = subtract_columns(data_df_matched[col], emb_df_matched[col], col)

#         if relative:
#             col1 = divide_columns(col1, np.abs(data_df[col]))
#             col2 = divide_columns(col2, np.abs(data_df[col]))
#             col3 = divide_columns(col3, np.abs(data_df_matched[col]))
#             title = quantity["rel_title"]
#         else:
#             title = quantity["title"]

#         q_dict = {
#             "Data - Data (matched)": col1,
#             "Data - Emb (matched)": col2,
#             "Data (matched) - Emb (matched)": col3
#         }
#         ax = nq_comparison(q_dict, bins=bins, title=title)
        
#         if quantity["xlog"]:
#             ax.set_xscale("log")
#         if quantity["ylog"]:
#             ax.set_yscale("log")
        
#         plt.savefig(os.path.join(comparison_output_path, mode, f"{col}.png"))
#         plt.close()

# print("Created triple comparison plots")





for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        for relative in [True, False]:
            if mode == "default":
                bins = nbins
            elif mode == "custom" and relative:
                bins = quantity["rel_bins"]
            elif mode == "custom" and not relative:
                bins = quantity["bins"]

            col = quantity["col"]

            q_diff = subtract_columns(data_df[col], emb_df_matched[col], col)
            
            if relative:
                title = quantity["rel_title"]
                q_diff = divide_columns(q_diff, np.abs(data_df[col]))
                path = rel_output_path
            else:
                title = quantity["title"]
                path = abs_output_path

            ax = histogram(q_diff, bins, title)

            if quantity["xlog"]:
                ax.set_xscale("log")
            if quantity["ylog"]:
                ax.set_yscale("log")
            
            plt.savefig(os.path.join(path, mode, f"{col}.png"))
            plt.close()

print("Created diff plots")


print("Plotting finished")

