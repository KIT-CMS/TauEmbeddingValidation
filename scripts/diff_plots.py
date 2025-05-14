import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plotting import histogram, nq_comparison
from importer import verify_events, initialize_dir
from genmatching import detect_changes, subtract_columns, divide_columns

hdf_path = "./data/converted/converted_nanoaod.h5"

comparison_output_path = "./output/diff_plots/comparison"
comparison_output_path = "./output/diff_plots/comparison"

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
        "title":r"$\eta_\text{µ1, emb}$ - $\eta_\text{µ1, data}$",             
        "rel_title":r"$(\eta_\text{µ1, emb}$ - $\eta_\text{µ1, data}$) / $\eta_\text{µ1, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_eta",           
        "bins":np.linspace(-5, 5, 25),          
        "title":r"$\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$",                 
        "rel_title":r"$(\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$) / $\eta_\text{LJet, data}$",    
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "bins":np.linspace(0, 50, 25),         
        "title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data}) / GeV$",           
        "rel_title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data})$ / $m_\text{LJet, data})$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-7, 7, 25),      
        "title":r"$\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data}$",              
        "rel_title":r"$(\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data})$ / $\phi_\text{LJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(-200, 400, 25),         
        "title":r"$m_\text{µµ, emb}$ - $m_\text{µµ, data})$",             
        "rel_title":r"$(m_\text{µµ, emb}$ - $m_\text{µµ, data})$ / $m_\text{µµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "bins":np.linspace(-7, 7, 25),          
        "title":r"$\phi_\text{µ1, emb}$ - $\phi_\text{µ1, data}$",             
        "rel_title":r"($\phi_\text{µ1, emb}$ - $\phi_\text{µ1, data})$ / $\phi_\text{µ1, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "bins":np.linspace(0, 150, 25),         
        "title":r"$p_\text{T, µ1, emb}$ - $p_\text{T, µ1, data}$",  
        "rel_title":r"$(p_\text{T, µ1, emb}$ - $p_\text{T, µ1, data})$ / $p_\text{T, µ1, data}$",  
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(-1000, 2500, 25),        
        "title":r"$p_\text{T µµ, emb}$ - $p_\text{T µµ, data}$",           
        "rel_title":r"$(p_\text{T µµ, emb}$ - $p_\text{T µµ, data})$ / $p_\text{T µµ, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-3.5, 3.5, 25),      
        "title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / GeV",           
        "rel_title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / $E_\text{\phi miss, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 110, 25),        
        "title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})/ GeV$",           
        "rel_title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})$ / $p_\text{T miss, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_sumEt",    
        "bins":np.linspace(0, 700, 25),         
        "title":r"$(E_\text{miss, emb}$ - $E_\text{miss, data}$) / GeV",      
        "rel_title":r"$(E_\text{miss, emb}$ - $E_\text{miss, data})$ / $E_\text{miss, data}$",      
        "relative":False,
        "ylog":True,    
        "xlog":False},
]


#comparison plots

for quantity in plotting_instructions:
    for mode in ["custom", "default"]:
        if mode == "default":
            bins = 25
        elif mode == "custom":
            bins = quantity["bins"]

        col = quantity["col"]
        relative = quantity["relative"]

        col1 = subtract_columns(data_df[col], emb_df[col], col)
        col2 = subtract_columns(data_df[col], emb_df_matched[col], col)
        col3 = subtract_columns(data_df[col], emb_df_matched_filtered[col], col)

        if relative:
            col1 = divide_columns(col1, data_df[col])
            col2 = divide_columns(col2, data_df[col])
            col3 = divide_columns(col2, data_df[col])
            title = quantity["rel_title"]
        else:
            title = quantity["title"]

        q_dict = {
            "Emb (raw) - data": col1,
            "Emb (matched) - data": col2,
            "Emb (matched, filtered) - data": col3
        }
        ax = nq_comparison(q_dict, bins=bins, title=title)
        
        if quantity["xlog"]:
            ax.set_xscale("log")
        if quantity["ylog"]:
            ax.set_yscale("log")
        
        plt.savefig(os.path.join(comparison_output_path, mode, f"{col}.png"))
        plt.close()

print("Created triple comparison plots")




# for quantity in plotting_instructions:
#     col = quantity["col"]
#     bins = quantity["bins"]
#     relative = quantity["relative"]


#     q_diff = subtract_columns(data_df[col], emb_df_matched[col], col)
    
#     if relative:
#         title = quantity["rel_title"]
#         q_diff = divide_columns(q_diff, data_df[col])
#     else:
#         title = quantity["title"]


#     ax = histogram(q_diff, bins, title)

#     if quantity["xlog"]:
#         ax.set_xscale("log")
#     if quantity["ylog"]:
#         ax.set_yscale("log")
    
#     plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
#     plt.close()

# print("Created matched diff plots with custom binning")



# #comparison: embedding (raw) -data, embedding (matched) - data
# for quantity in plotting_instructions:
#     col = quantity["col"]
#     bins = 25
#     relative = quantity["relative"]

#     col1 = subtract_columns(data_df[col], emb_df_matched[col], col)
#     col2 = subtract_columns(data_df[col], emb_df[col], col)

#     if relative:
#         col1 = divide_columns(col1, data_df[col])
#         col2 = divide_columns(col2, data_df[col])
#         title = quantity["rel_title"]
#         ax = q_comparison(col1, col2, bins, "(Matched emb - data)/ data", "(Unmatched emb - data)/ data", title)
#     else:
#         title = quantity["title"]
#         ax = q_comparison(col1, col2, bins, "Matched emb - data", "Unmatched emb - data", title)
    
#     if quantity["xlog"]:
#         ax[0].set_xscale("log")
#     if quantity["ylog"]:
#         ax[0].set_yscale("log")
    
#     plt.savefig(os.path.join(matched_comparison_output_path, "default", f"{col}.png"))
#     plt.close()

# print("Created matched comparison plots with default binning")



print("Plotting finished")

