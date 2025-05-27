import uproot
import os
import pandas as pd
import numpy as np
import vector

filter_list = [
    {"col":"dr", "min":0, "max":0.01},
    {"col":"pt", "min":27, "max":np.inf},
    {"col":"eta", "min":-2.5, "max":2.5},
    # {"col":"pt_ratio", "min":0.75, "max":1.25}
]

def get_filter_list():
    return filter_list



#following code is for genmatching
def calculate_dr(df, n_muon, filter=None):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset

    dr_arr = np.full(shape=(len(df), 2, n_muon), dtype=float, fill_value=np.nan)

    #looping over all data particle and embedding particle combinations
    for n in range(1, 3):
        if n == 1:
            col_name = "LM"#first comparing to leading muon
        elif n == 2:
            col_name = "TM"#then comparing to trailing muon

        for n_m in range(1, n_muon+1):
            eta_diff = subtract_columns(df[f"{col_name}_eta"], df[f"eta_{n_m}"], "eta_")
            phi_diff = subtract_columns(df[f"{col_name}_phi"], df[f"phi_{n_m}"], "phi_")

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
                        pt_ratio = df[f"{col_name}_pt"]/ df[f"pt_{n_m}"]
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

def apply_genmatching(dr_arr, df):
    #switches data for those entries where an emb muon closer to the original one is present
    target_length = len(df)
    lm_pt = np.full(target_length, fill_value=np.nan)
    tm_pt = np.full(target_length, fill_value=np.nan)
    lm_phi = np.full(target_length, fill_value=np.nan)
    tm_phi = np.full(target_length, fill_value=np.nan)
    lm_eta = np.full(target_length, fill_value=np.nan)
    tm_eta = np.full(target_length, fill_value=np.nan)
    lm_m = np.full(target_length, fill_value=np.nan)
    tm_m = np.full(target_length, fill_value=np.nan)

    for n_event in range(len(df)):
        distances = dr_arr[n_event, :, :]
        
        muon1_id = find_closest_muon(distances[0, :])
        muon2_id = find_closest_muon(distances[1, :])
        
        #checking whether candidates both fit best to the same muon
        if muon1_id == muon2_id:
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

        #ignoring nans
        if ~np.isnan(muon1_id):
            lm_pt[n_event] = event[f"pt_{muon1_id+1}"]
            lm_eta[n_event] = event[f"eta_{muon1_id+1}"]
            lm_phi[n_event] = event[f"phi_{muon1_id+1}"]
            lm_m[n_event] = event[f"m_{muon1_id+1}"]
        else:
            lm_pt[n_event] = np.nan
            lm_eta[n_event] = np.nan
            lm_phi[n_event] = np.nan
            lm_m[n_event] = np.nan

        if ~np.isnan(muon2_id):
            tm_pt[n_event] = event[f"pt_{muon2_id+1}"]
            tm_eta[n_event] = event[f"eta_{muon2_id+1}"]
            tm_phi[n_event] = event[f"phi_{muon2_id+1}"]
            tm_m[n_event] = event[f"m_{muon2_id+1}"]
        else:
            tm_pt[n_event] = np.nan
            tm_eta[n_event] = np.nan
            tm_phi[n_event] = np.nan
            tm_m[n_event] = np.nan


    matched_df = pd.DataFrame({
        "LM_pt": pd.Series(lm_pt),
        "TM_pt": pd.Series(tm_pt),
        "LM_eta": pd.Series(lm_eta),
        "TM_eta": pd.Series(tm_eta),
        "LM_phi": pd.Series(lm_phi),
        "TM_phi": pd.Series(tm_phi),
        "LM_m": pd.Series(lm_m),
        "TM_m": pd.Series(tm_m)
    })
    df[["LM_pt", "TM_pt", "LM_eta", "TM_eta", "LM_phi", "TM_phi", "LM_m", "TM_m"]] = matched_df
    return df



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



def detect_changes(df1, df2, columns:list):
    res = ""
    for column in columns:
        temp = df1[column] - df2[column]
        mask1 = ~np.isnan(temp)
        mask2 = temp != 0
        mask = np.logical_and(mask1, mask2)
        count = mask.sum()
        res += f"{column}: {count}; "
    print(res + "rows different")



def subtract_columns(col1, col2, col_name:str):
    if not "phi" in col_name:
        diff = np.abs(col1 - col2)
    #phi needs to be handled differently because the value must be lower than pi
    else:
        diff = np.abs(col1 - col2)

        mask = diff > np.pi
        
        diff[mask] = 2*np.pi - diff[mask]

    return diff


def divide_columns(numerator, divisor):
    mask1 = divisor != 0
    mask2 = ~np.isnan(divisor)
    mask3 = ~np.isnan(numerator)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)

    q = np.full_like(numerator, np.nan)
    q[mask] = numerator[mask]/ divisor[mask]

    return q