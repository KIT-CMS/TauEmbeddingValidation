import uproot
import os
import pandas as pd
import numpy as np
from source.helper import subtract_columns, get_n_occurence

filter_list = [
    {"col":"dr", "min":0, "max":0.01},
    {"col":"pt", "min":8, "max":np.inf},
    {"col":"eta", "min":-2.5, "max":2.5},
    # {"col":"pt_ratio", "min":0.75, "max":1.25}
]

def get_filter_list():
    return filter_list



#following code is for genmatching
def calculate_dr(df, mode, filter=None):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset

    if mode == "muon":
        n_comp = get_n_occurence(df, "eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "jet":
        n_comp = get_n_occurence(df, "Jet_eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "filter":
        n_comp = get_n_occurence(df, "eta")
        n_target = get_n_occurence(df, "Jet_pt")
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    else:
        raise ValueError("Invalid mode selected")

    #looping over all data particle and embedding particle combinations
    for n in range(1, n_target+1):
        #if dr should be calculated between muon, the columns to be used are different from the columns in the jet case. the following clauses assign the names of the columns based on the mode. 
        if mode=="muon":
            comp_phi = "phi"#this is the name of the columns that the 2 relevant muons are being compared to (simply all muon columns)
            comp_eta = "eta"
            comp_pt = "pt"
            if n == 1:
                master_eta = "LM_eta"#first comparing to leading muon
                master_phi = "LM_phi"
                master_pt = "LM_pt"
            elif n == 2:
                master_eta = "TM_eta"#then comparing to trailing muon
                master_phi = "TM_phi"
                master_pt = "TM_pt"
        elif mode=="jet":
            comp_phi = "Jet_phi"
            comp_eta = "Jet_eta"
            comp_pt = "Jet_pt"
            if n == 1:
                master_eta = "LJ_eta"#comparing the first jet
                master_phi = "LJ_phi"
                master_pt = "LJ_pt"
            elif n == 2:
                master_eta = "TJ_eta"#then comparing the second jet
                master_phi = "TJ_phi"
                master_pt = "TJ_pt"
        elif mode=="filter":
            comp_phi = "phi"
            comp_eta = "eta"
            comp_pt = "pt"
            master_eta = "Jet_eta"#comparing the first jet
            master_phi = "Jet_phi"
            master_pt = "Jet_pt"
        
        for n_m in range(1, n_comp+1):
            if mode == "filter":
                eta_diff = subtract_columns(df[f"{master_eta}_{n}"], df[f"{comp_eta}_{n_m}"], "eta_")
                phi_diff = subtract_columns(df[f"{master_phi}_{n}"], df[f"{comp_phi}_{n_m}"], "phi_")
            else:
                eta_diff = subtract_columns(df[master_eta], df[f"{comp_eta}_{n_m}"], "eta_")
                phi_diff = subtract_columns(df[master_phi], df[f"{comp_phi}_{n_m}"], "phi_")
            #calculating the dr value between them for all events
            dr_temp = np.sqrt(np.square(eta_diff) + np.square(phi_diff))

            if type(filter) != type(None):
                #looping over given filters and applying them on the calculated dr array. thereby values associated with invalid muons are removed
                for f in filter:
                    basename = f["col"]
                    min_val = f["min"]
                    max_val = f["max"]
                    #filter on dr and pt ratio have to be treated separately because they are no nanoaod columns
                    if basename != "dr" and basename != "pt_ratio":#in this case the filter is applied on existing columns from df
                        mask1 = df[f"{basename}_{n_m}"] < min_val 
                        mask2 = df[f"{basename}_{n_m}"] > max_val 
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
                    elif basename == "dr":
                        mask1 = dr_temp < min_val
                        mask2 = dr_temp > max_val
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
                    elif basename == "pt_ratio":
                        pt_ratio = df[master_pt]/ df[f"{comp_pt}_{n_m}"]
                        mask1 = pt_ratio < min_val
                        mask2 = pt_ratio > max_val
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
            dr_arr[:, n-1, n_m-1] = dr_temp

    return dr_arr


def find_closest_muon(dr_slice):
    #returns the minimum value index of a 1d array if existent otherwise nan. thereby errors occuring if all nan slices are encountered are bypassed
    nan_mask = ~np.isnan(dr_slice)

    if nan_mask.sum() > 0:
        index = np.nanargmin(dr_slice)
        return index
    return np.nan
        

def remove_emb_mu_from_dist(dist, id):
    # sets a value of an 1d array to nan, thereby avoiding double selection 
    if ~np.isnan(id):
        dist[:, id] = np.nan
    return dist

def apply_genmatching(dr_arr, df, mode):
    #switches data for those entries where an emb muon closer to the original one is present

    if mode == "muon":
        pt_source = "pt"
        eta_source = "eta"
        phi_source = "phi"
        m_source = "m"
        pt_target_1 = "LM_pt"
        pt_target_2 = "TM_pt"
        eta_target_1 = "LM_eta"
        eta_target_2 = "TM_eta"
        phi_target_1 = "LM_phi"
        phi_target_2 = "TM_phi"
        m_target_1 = "LM_m"
        m_target_2 = "TM_m"
    elif mode == "jet":
        pt_source = "Jet_pt"
        eta_source = "Jet_eta"
        phi_source = "Jet_phi"
        m_source = "Jet_m"
        pt_target_1 = "LJ_pt"
        pt_target_2 = "TJ_pt"
        eta_target_1 = "LJ_eta"
        eta_target_2 = "TJ_eta"
        phi_target_1 = "LJ_phi"
        phi_target_2 = "TJ_phi"
        m_target_1 = "LJ_m"
        m_target_2 = "TJ_m"
    else:
        raise ValueError("invalid mode selected")
    
    target_length = len(df)
    lm_pt = np.full(target_length, fill_value=np.nan)
    tm_pt = np.full(target_length, fill_value=np.nan)
    lm_phi = np.full(target_length, fill_value=np.nan)
    tm_phi = np.full(target_length, fill_value=np.nan)
    lm_eta = np.full(target_length, fill_value=np.nan)
    tm_eta = np.full(target_length, fill_value=np.nan)
    lm_m = np.full(target_length, fill_value=np.nan)
    tm_m = np.full(target_length, fill_value=np.nan)

    muon_best_fit = np.full((target_length,2), fill_value=np.nan)
    dr_min = np.full((target_length,2), fill_value=np.nan)

    for n_event in range(len(df)):
        distances = dr_arr[n_event, :, :]
        distances2 = np.copy(dr_arr[n_event, :, :])
        
        muon1_id = find_closest_muon(distances[0, :])
        muon2_id = find_closest_muon(distances[1, :])

        #checking whether candidates both fit best to the same muon (ignoring nans)
        if muon1_id == muon2_id and ~np.isnan(muon1_id):
            #in this case the muon can only be matched once.
            muon_id = muon1_id
            #thus removing the id for avoiding reselection

            dr1 = distances[0,muon_id]
            dr2 = distances[1,muon_id]
            
            distances = remove_emb_mu_from_dist(distances, muon_id)
            #if muon fits best to first candidate
            if dr1 <= dr2:
                #the second one is recalculated
                muon2_id = find_closest_muon(distances[1, :])
            #otherwise the other way around
            else:
                muon1_id = find_closest_muon(distances[0, :])
        #else: #does not matter
        event = df.iloc[n_event]

        #setting the new value if a valid one could be found - otherwise nan is set
        if ~np.isnan(muon1_id):
            lm_pt[n_event] = event[f"{pt_source}_{muon1_id+1}"]
            lm_eta[n_event] = event[f"{eta_source}_{muon1_id+1}"]
            lm_phi[n_event] = event[f"{phi_source}_{muon1_id+1}"]
            lm_m[n_event] = event[f"{m_source}_{muon1_id+1}"]
            muon_best_fit[n_event, 0] = muon1_id
            dr_min[n_event, 0] = distances2[0, muon1_id]
        else:
            lm_pt[n_event] = np.nan
            lm_eta[n_event] = np.nan
            lm_phi[n_event] = np.nan
            lm_m[n_event] = np.nan
            muon_best_fit[n_event, 0] = np.nan
            dr_min[n_event, 0] = np.nan

        #setting the new value if a valid one could be found - otherwise nan is set
        if ~np.isnan(muon2_id):
            tm_pt[n_event] = event[f"{pt_source}_{muon2_id+1}"]
            tm_eta[n_event] = event[f"{eta_source}_{muon2_id+1}"]
            tm_phi[n_event] = event[f"{phi_source}_{muon2_id+1}"]
            tm_m[n_event] = event[f"{m_source}_{muon2_id+1}"]
            muon_best_fit[n_event, 1] = muon2_id
            dr_min[n_event, 1] = distances2[1, muon2_id]
        else:
            tm_pt[n_event] = np.nan
            tm_eta[n_event] = np.nan
            tm_phi[n_event] = np.nan
            tm_m[n_event] = np.nan

    matched_df = pd.DataFrame({
        f"{pt_target_1}": pd.Series(lm_pt),
        f"{pt_target_2}": pd.Series(tm_pt),
        f"{eta_target_1}": pd.Series(lm_eta),
        f"{eta_target_2}": pd.Series(tm_eta),
        f"{phi_target_1}": pd.Series(lm_phi),
        f"{phi_target_2}": pd.Series(tm_phi),
        f"{m_target_1}": pd.Series(lm_m),
        f"{m_target_2}": pd.Series(tm_m)
    })
    df = pd.concat([df, matched_df], axis=1)
    # df[[pt_target_1, pt_target_2, eta_target_1, eta_target_2, phi_target_1, phi_target_2, m_target_1, m_target_2]] = matched_df
    return df, muon_best_fit, dr_min



def get_closest_muon_data(dr_arr):
    #returns the index and dr of the emb muon closest to the first data muon
    length = dr_arr.shape[0]

    index = -np.full(length, 99, int)
    mu_dr = -np.full(length, np.nan, float)

    for n_event in range(length):
        distances = dr_arr[n_event, :, :]
        
        muon1_id = find_closest_muon(distances[0, :])
        if not np.isnan(muon1_id):
            index[n_event] = muon1_id
            mu_dr[n_event] = dr_arr[n_event, 0, muon1_id]
        else:
            index[n_event] = 0
            mu_dr[n_event] = dr_arr[n_event, 0, 0]
    
    return index, mu_dr


def remove_muon_jets(df, dr_arr):
    #removes those jets that are closer than "value" to a muon
    for n_j in range(dr_arr.shape[1]):
        for n_m in range(dr_arr.shape[2]):
            subset = dr_arr[:,n_j,n_m]
            mask = subset<0.1
            df.loc[mask, f"Jet_eta_{n_j+1}"] = np.nan
            df.loc[mask, f"Jet_m_{n_j+1}"] = np.nan
            df.loc[mask, f"Jet_phi_{n_j+1}"] = np.nan
            df.loc[mask, f"Jet_pt_{n_j+1}"] = np.nan
    return df