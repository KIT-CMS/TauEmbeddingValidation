import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plotting import histogram, q_comparison, initialize_dir
from importer import verify_events, detect_changes

hdf_path = "./data/converted/converted_nanoaod.h5"
default_output_path = "./output/diff_plots/data-emb_raw"
matched_output_path = "./output/diff_plots/data-emb_matched"
matched_filtered_output_path = "./output/diff_plots/data-emb_matched_filtered_emb"
matched_comparison_output_path = "./output/diff_plots/comparison-emb_raw-emb_matched"
matched_filtered_comparison_output_path = "./output/diff_plots/comparison-emb_raw-emb_matched_filtered"

initialize_dir(default_output_path, ["default", "custom"])
initialize_dir(matched_output_path, ["default", "custom"])
initialize_dir(matched_filtered_output_path, ["default", "custom"])
initialize_dir(matched_comparison_output_path, ["default", "custom"])
initialize_dir(matched_filtered_comparison_output_path, ["default", "custom"])

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
        "title":r"$\eta_\text{Lµ, emb}$ - $\eta_\text{Lµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_eta",           
        "bins":np.linspace(-5, 5, 25),          
        "title":r"$\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$",                 
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "bins":np.linspace(0, 50, 25),         
        "title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data}) / GeV$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-7, 7, 25),      
        "title":r"$\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(-200, 400, 25),         
        "title":r"$m_\text{µµ, emb}$ - $m_\text{µµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "bins":np.linspace(-7, 7, 25),          
        "title":r"$\phi_\text{Lµ, emb}$ - $\phi_\text{Lµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "bins":np.linspace(0, 150, 25),         
        "title":r"$p_\text{T, Lµ, emb}$ - $p_\text{T, Lµ, data}$",  
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(-1000, 2500, 25),        
        "title":r"$p_\text{T µµ, emb}$ - $p_\text{T µµ, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-3.5, 3.5, 25),      
        "title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / GeV",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "bins":np.linspace(0, 110, 25),        
        "title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})/ GeV$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_sumEt",    
        "bins":np.linspace(0, 700, 25),         
        "title":r"$(E_\text{miss, emb}$ - $E_\text{miss, data}$) / GeV",      
        "relative":False,
        "ylog":True,    
        "xlog":False},
]


#data - embedding_raw
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df[col]) 
    else:
        q_diff = data_df[col] - emb_df[col]

    ax = histogram(q_diff, bins, title)
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(default_output_path, "default", f"{col}.png"))
    plt.close()

print("Created unmatched diff plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df[col]) 
    else:
        q_diff = data_df[col] - emb_df[col]

    ax = histogram(q_diff, bins, title)

    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(default_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created unmatched diff plots with custom binning")

#data - embedding (matched)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df_matched[col]) 
    else:
        q_diff = data_df[col] - emb_df_matched[col]

    ax = histogram(q_diff, bins, title)
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched diff plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df_matched[col]) 
    else:
        q_diff = data_df[col] - emb_df_matched[col]

    ax = histogram(q_diff, bins, title)

    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched diff plots with custom binning")



#data - embedding (matched + filtered)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df_matched_filtered[col]) 
    else:
        q_diff = data_df[col] - emb_df_matched_filtered[col]

    ax = histogram(q_diff, bins, title)
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched + filtered diff plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df_matched[col]) 
    else:
        q_diff = data_df[col] - emb_df_matched[col]

    ax = histogram(q_diff, bins, title)

    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched + filtered diff plots with custom binning")


#comparison: embedding (raw) -data, embedding (matched) - data
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    col1 = emb_df_matched[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_comparison_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    col1 = emb_df_matched[col] - data_df[col]
    col2 = emb_df[col] - data_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with custom binning")


#comparison: embedding (raw) -data, embedding (matched, filtered) - data
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    col1 = emb_df_matched_filtered[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_comparison_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched + filtered comparison plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    col1 = emb_df_matched_filtered[col] - data_df[col]
    col2 = emb_df[col] - data_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched + filtered comparison plots with custom binning")




print("Plotting finished")

