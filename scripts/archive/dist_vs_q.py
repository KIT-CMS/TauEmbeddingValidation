import uproot
import os
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vector

from plotting import x_vs_y


hdf_path = "./data/converted/converted_nanoaod.h5"
default_output_path = "./output/diff_vs_dr"
matched_output_path = "./output/diff_matched_vs_dr"
print("Loading data")

data_df = pd.read_hdf(hdf_path, "data_df")
emb_df = pd.read_hdf(hdf_path, "emb_df")
matched_emb_df = pd.read_hdf(hdf_path, "emb_df_matched")

print("Data loaded")


plotting_instructions = [
    {"col":"eta_1",             
        "xrange":(-2.5, 2.5),      
        "title":r"$\eta_\text{Lµ, emb}$ - $\eta_\text{Lµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_eta",           
        "xrange":(-5, 5),          
        "title":r"$\eta_\text{LJet, emb}$ - $\eta_\text{LJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"Jet_mass",          
        "xrange":(0, 20),         
        "title":r"($m_\text{LJet, emb}$ - $m_\text{LJet, data}) / GeV$",           
        "relative":False,
        "ylog":True,    "xlog":False},
    {"col":"Jet_phi",           
        "xrange":(-3.5, 3.5),      
        "title":r"$\phi_\text{LJet, emb}$ - $\phi_\text{LJet, data}$",              
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"m_vis",             
        "xrange":(12, 140),         
        "title":r"$m_\text{µµ, emb}$ - $m_\text{µµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"phi_1",             
        "xrange":(-2.5, 2.5),          
        "title":r"$\phi_\text{Lµ, emb}$ - $\phi_\text{Lµ, data}$",             
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_1",              
        "xrange":(12, 140),         
        "title":r"$p_\text{T, Lµ, emb}$ - $p_\text{T, Lµ, data}$",  
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"pt_vis",            
        "xrange":(12, 140),        
        "title":r"$p_\text{T µµ, emb}$ - $p_\text{T µµ, data}$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_phi",      
        "xrange":(-3.5, 3.5),      
        "title":r"$(E_\text{\phi miss, emb}$ - $E_\text{\phi miss, data}$) / GeV",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_pt",       
        "xrange":(0, 110),         
        "title":r"$(p_\text{T miss, emb}$ - $p_\text{T miss, data})/ GeV$",           
        "relative":False,
        "ylog":True,    
        "xlog":False},
    {"col":"PuppiMET_sumEt",    
        "xrange":(0, 700),         
        "title":r"$(E_\text{miss, emb}$ - $E_\text{miss, data}$) / GeV",      
        "relative":False,
        "ylog":True,    
        "xlog":False},
]

data_p4 = vector.MomentumObject4D(pt=data_df["pt_1"], phi=data_df["phi_1"], eta=data_df["eta_1"], mass=data_df["m_1"])
emb_p4 = vector.MomentumObject4D(pt=emb_df["pt_1"], phi=emb_df["phi_1"], eta=emb_df["eta_1"], mass=emb_df["m_1"])
matched_emb_p4 = vector.MomentumObject4D(pt=matched_emb_df["pt_1"], phi=matched_emb_df["phi_1"], eta=matched_emb_df["eta_1"], mass=matched_emb_df["m_1"])

#dr without matching
p4_diff = data_p4 - emb_p4
dist = np.sqrt(np.square(p4_diff.eta) + np.square(p4_diff.phi))

#dr with matching
matched_p4_diff = data_p4 - matched_emb_p4
matched_dist = np.sqrt(np.square(matched_p4_diff.eta) + np.square(matched_p4_diff.phi))


for quantity in plotting_instructions:
    col = quantity["col"]
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df[col]) 
    else:
        q_diff = data_df[col] - emb_df[col]

    ax = x_vs_y(q_diff, dist, title, r"$dr_\text{data-emb}$")
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(default_output_path, "default_range", f"{col}.png"))
    plt.close()

print("Created unmatched xy plots with default range")

for quantity in plotting_instructions:
    col = quantity["col"]
    x_range = quantity["xrange"]
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - emb_df[col]) 
    else:
        q_diff = data_df[col] - emb_df[col]

    ax = x_vs_y(q_diff, dist, title, r"$dr_\text{data-emb}$")
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    ax.set_xlim(x_range)
    
    plt.savefig(os.path.join(default_output_path, "custom_range", f"{col}.png"))
    plt.close()

print("Created unmatched xy plots with custom range")

for quantity in plotting_instructions:
    col = quantity["col"]
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - matched_emb_df[col]) 
    else:
        q_diff = data_df[col] - matched_emb_df[col]

    ax = x_vs_y(q_diff, matched_dist, title, r"$dr_\text{data-emb}$")
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    
    plt.savefig(os.path.join(matched_output_path, "default_range", f"{col}.png"))
    plt.close()

print("Created matched xy plots with default range")

for quantity in plotting_instructions:
    col = quantity["col"]
    x_range = quantity["xrange"]
    title = quantity["title"]
    relative = quantity["relative"]

    if relative:
        title = title.replace(" - ", " / ")/ data_df[col]
        q_diff = (data_df[col] - matched_emb_df[col]) 
    else:
        q_diff = data_df[col] - matched_emb_df[col]

    ax = x_vs_y(q_diff, matched_dist, title, r"$dr_\text{data-emb}$")
    
    if quantity["xlog"]:
        ax.set_xscale("log")
    if quantity["ylog"]:
        ax.set_yscale("log")
    ax.set_xlim(x_range)
    
    plt.savefig(os.path.join(matched_output_path, "custom_range", f"{col}.png"))
    plt.close()

print("Created matched xy plots with custom range")

print("Plotting finished")