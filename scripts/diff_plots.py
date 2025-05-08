import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import initialize_dir
from plotting import histogram, q_comparison
from importer import verify_events

hdf_path = "./data/converted/converted_nanoaod.h5"
default_output_path = "./output/diff_plots_unmatched_emb"
matched_output_path = "./output/diff_plots_matched_emb"
comparison_output_path = "./output/diff_plots_comparison"

initialize_dir(default_output_path, ["default", "custom"])
initialize_dir(matched_output_path, ["default", "custom"])
initialize_dir(comparison_output_path, ["default", "custom"])

print("Initialized directories")

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df")
matched_emb_df = pd.read_hdf(hdf_path, "emb_df_matched")

verify_events(data_df, emb_df, matched_emb_df)

print("Data loaded and verified")


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
        "bins":np.linspace(0, 20, 25),         
        "title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data}) / GeV$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-3.5, 3.5, 25),      
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
        "bins":np.linspace(12, 140, 25),         
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
    
    plt.savefig(os.path.join(matched_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched diff plots with default binning")


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
    
    plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched diff plots with custom binning")

for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    col1 = matched_emb_df[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(comparison_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    col1 = matched_emb_df[col] - data_df[col]
    col2 = emb_df[col] - data_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with custom binning")

print("Plotting finished")

