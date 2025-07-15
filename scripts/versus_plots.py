import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from source.plotting import x_vs_y
from source.importer import initialize_dir
from source.helper import verify_events, set_working_dir


########################################################################################################################################################################
# Paths for input and output
########################################################################################################################################################################

hdf_path = "./output/data/converted_nanoaod.h5"
# comparison_output_path = "./output/control_plots/comparison"
versus_output_path = "./output/versus_plots/"

initialize_dir(versus_output_path)

set_working_dir()

print("Initialized directories")

########################################################################################################################################################################
# Instructions for plots
########################################################################################################################################################################


plotting_instructions = [
    {"col":"LJ_eta",  
        "min":-5,
        "max":5,       
        "title":r"LJet $\eta$",                 
        "ylog":False,    
        "xlog":False},
    {"col":"LJ_m",     
        "min":0,
        "max":60,            
        "title":r"LJet mass/ GeV",           
        "ylog":False,    
        "xlog":False},
    {"col":"LJ_phi",    
        "min":-3.5,
        "max":3.5,                  
        "title":r"LJet $\phi$",              
        "ylog":False,    
        "xlog":False}, 
    {"col":"LJ_pt",      
        "min":0,
        "max":250,        
        "title":r"LJet $p_\text{T}$",              
        "ylog":False,    
        "xlog":False}, 
    {"col":"LM_phi",   
        "min":-3.5,
        "max":3.5,              
        "title":r"$\phi_\text{µ1}$",  
        "ylog":False,    
        "xlog":False},
    {"col":"LM_pt",     
        "min":0,
        "max":300,                
        "title":r"$p_\text{T, µ1}$/ GeV",  
        "ylog":False,    
        "xlog":False},
    {"col":"m_vis",     
        "min":0,
        "max":200,            
        "title":r"$m_\text{µµ}$/ GeV",             
        "ylog":False,    
        "xlog":False},
    {"col":"pt_vis",      
        "min":0,
        "max":200,          
        "title":r"$p_\text{T, µµ}$/ GeV",           
        "ylog":False,    
        "xlog":False},
    {"col":"PuppiMET_phi",    
        "min":-3.5,
        "max":3.5,     
        "title":r"Missing $p_{T, \phi}$",           
        "ylog":False,    
        "xlog":False},
    {"col":"PuppiMET_pt",    
        "min":0,
        "max":150,       
        "title":r"Missing $p_{T}$ / GeV",           
        "ylog":False,    
        "xlog":False},
    {"col":"TJ_eta",    
        "min":-5,
        "max":5,      
        "title":r"TJet $\eta$",                 
        "ylog":False,    
        "xlog":False},
    {"col":"TJ_m",         
        "min":0,
        "max":30,        
        "title":r"TJet mass/ GeV",           
        "dy":None,
        "ylog":False,    
        "xlog":False},
    {"col":"TJ_phi",    
        "min":-3.5,
        "max":3.5,            
        "title":r"TJet $\phi$",              
        "ylog":False,    
        "xlog":False}, 
    {"col":"TJ_pt",   
        "min":0,
        "max":200,          
        "title":r"TJet $p_\text{T}$",              
        "ylog":False,    
        "xlog":False}, 
    {"col":"TM_eta",   
        "min":-2.5,
        "max":2.5,                
        "title":r"$\eta_\text{µ2}$",             
        "ylog":False,    
        "xlog":False},   
    {"col":"TM_phi",     
        "min":-3.5,
        "max":3.5,             
        "title":r"$\phi_\text{µ2}$",  
        "ylog":False,    
        "xlog":False},
    {"col":"TM_pt",     
        "min":0,
        "max":150,            
        "title":r"$p_\text{T, µ2}$/ GeV",  
        "ylog":False,    
        "xlog":False},
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

for quantity in plotting_instructions:
    col = quantity["col"]
    x = data_df[col]
    y = emb_df[col]
    xlabel = quantity["title"] + " (data)"
    ylabel = quantity["title"] + " (emb)"
    min_lim = quantity["min"]
    max_lim = quantity["max"]

    ax = x_vs_y(x, y, xlabel, ylabel)

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.plot([min_lim, max_lim], [min_lim, max_lim], ls="dashed", c="black")

    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(versus_output_path, f"{col}.png"))
    plt.close()

print("Created versus plots ")


print("Plotting finished")