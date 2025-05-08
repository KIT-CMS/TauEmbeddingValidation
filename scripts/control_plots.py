import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from helper import initialize_dir
from plotting import control_plot, q_comparison


hdf_path = "./data/converted/converted_nanoaod.h5"
default_output_path = "./output/control_plots_unmatched_emb"
matched_output_path = "./output/control_plots_matched_emb"
comparison_output_path = "./output/control_plots_comparison"

initialize_dir(default_output_path, ["default", "custom"])
initialize_dir(matched_output_path, ["default", "custom"])
initialize_dir(comparison_output_path, ["default", "custom"])

print("Initialized directories")


data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df")
matched_emb_df = pd.read_hdf(hdf_path, "emb_df_matched")

print("Data loaded")


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
        "bins":np.linspace(-3.5, 3.5, 25),      
        "title":r"Leading Jet $\phi$",              
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "bins":np.linspace(1, 175, 25),         
        "title":r"$m_\text{vis}$/ GeV",             
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "bins":np.linspace(-7, 7, 25),          
        "title":r"Leading Muon $\phi$",             
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "bins":np.linspace(1, 175, 25),         
        "title":r"Leading Muon $p_\text{T}$/ GeV",  
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "bins":np.linspace(12, 200, 25),        
        "title":r"$p_\text{T vis}$/ GeV",           
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "bins":np.linspace(-3.5, 3.5, 25),      
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


for quantity in plotting_instructions:
    col = quantity["col"]
    bins = 25
    title = quantity["title"]

    ax = control_plot(data_df[col], matched_emb_df[col], bins, title)
    
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

    ax = control_plot(data_df[col], matched_emb_df[col], bins, title)

    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched control plots with custom binning")


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

    col1 = matched_emb_df[col]
    col2 = emb_df[col]
    ax = q_comparison(col1, col2, bins, "Matched emb", "Unmatched emb", title)
    if quantity["xlog"]:
        ax[0].set_xscale("log")
    if quantity["ylog"]:
        ax[0].set_yscale("log")
    
    plt.savefig(os.path.join(comparison_output_path, "custom", f"{col}.png"))
    plt.close()

print("Created matched comparison plots with custom binning")

print("Plotting finished")