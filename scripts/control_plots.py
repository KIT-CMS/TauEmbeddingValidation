import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import initialize_dir
from plotting import control_plot, q_comparison
from importer import verify_events, detect_changes


hdf_path = "./data/converted/converted_nanoaod.h5"
default_output_path = "./output/control_plots/data-emb_raw"
matched_output_path = "./output/control_plots/data-emb_matched"
matched_filtered_output_path = "./output/control_plots/data-emb_matched_filtered_emb"
matched_comparison_output_path = "./output/control_plots/comparison-emb_raw-emb_matched"
matched_filtered_comparison_output_path = "./output/control_plots/comparison-emb_raw-emb_matched_filtered"

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
        "title":r"Leading Muon $\eta$",             
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_eta",           
        "bins":np.linspace(-5, 5, 25),          
        "title":r"Leading Jet eta",                 
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "bins":np.linspace(0, 100, 25),         
        "title":r"Leading Jet mass/ GeV",           
        "ylog":True,    "xlog":False},
    {"col":"Jet_phi",           
        "bins":np.linspace(-4, 4, 25),      
        "title":r"Leading Jet $\phi$",              
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(0, 200, 25),         
        "title":r"$m_\text{vis}$/ GeV",             
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "bins":np.linspace(-4, 4, 25),          
        "title":r"Leading Muon $\phi$",             
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "bins":np.linspace(0, 300, 25),         
        "title":r"Leading Muon $p_\text{T}$/ GeV",  
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

#data vs embedding (raw)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]
    ax = control_plot(data_df[col], emb_df[col], bins, title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(default_output_path, "default", f"{col}.png"))
    plt.close()

print("Plotted control plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]
    ax = control_plot(data_df[col], emb_df[col], bins, title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(default_output_path, "custom", f"{col}.png"))
    plt.close()

print("Plotted control plots with custom binning")

#data vs embedding (matched)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    ax = control_plot(data_df[col], emb_df_matched[col], bins, title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched control plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    ax = control_plot(data_df[col], emb_df_matched[col], bins, title)

    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched control plots with custom binning")



#data vs embedding (matched+filtered)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    ax = control_plot(data_df[col], emb_df_matched_filtered[col], bins, title)
    
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_output_path, "default", f"{col}.png"))
    plt.close()

print("Created matched + filtered control plots with default binning")


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = quantity["bins"]
    title = quantity["title"]

    ax = control_plot(data_df[col], emb_df_matched_filtered[col], bins, title)

    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched + filtered control plots with default binning")


#comparison: embedding (matched) vs embedding (raw)
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

    col1 = emb_df_matched[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with custom binning")


#comparison: embedding (matched+filtered) vs embedding (raw)
for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    col1 = emb_df_matched_filtered[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched, filtered emb", "Unmatched emb", title)
    
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

    col1 = emb_df_matched_filtered[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched, filtered emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_filtered_comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched + filtered comparison plots with custom binning")
print("Plotting finished")