import uproot
import os
import pandas as pd
import numpy as np
import vector

filter_list = [
    {"col":"dr", "min":0, "max":0.01},
    # {"col":"pt", "min":27, "max":np.inf, "emb":True, "data":True},
    # {"col":"eta", "min":-2.5, "max":2.5, "emb":True, "data":True},
    # {"col":"pt_ratio", "min":0.75, "max":1.25}
]

def get_filter_list():
    return filter_list



#following code is for genmatching
def calculate_dr(data_df, emb_df, n_data, n_emb, filter=None):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset

    dr_arr = np.full(shape=(len(data_df), n_data, n_emb), dtype=float, fill_value=np.nan)

    #looping over all data particle and embedding particle combinations
    for n_d in range(1, n_data+1):
        for n_e in range(1, n_emb+1):
            eta_diff = subtract_columns(data_df[f"eta_{n_d}"], emb_df[f"eta_{n_e}"], "eta_")
            phi_diff = subtract_columns(data_df[f"phi_{n_d}"], emb_df[f"phi_{n_e}"], "phi_")

            #calculating the dr value between them for all events
            dr_temp = np.sqrt(np.square(eta_diff) + np.square(phi_diff))

            if type(filter) != type(None):
                #looping over given filters and applying them on the calculated dr array. thereby values associated with invalid muons are removed
                for f in filter:
                    basename = f["col"]
                    min_val = f["min"]
                    max_val = f["max"]
                    #filter on dr and pt ratio have to be treated separately because they are no nanoaod columns
                    if basename != "dr" and basename != "pt_ratio":
                        if f["emb"]:
                            mask1 = emb_df[f"{basename}_{n_e}"] < min_val 
                            mask2 = emb_df[f"{basename}_{n_e}"] > max_val 
                            mask = np.logical_or(mask1, mask2)
                            dr_temp[mask] = np.nan
                        if f["data"]:
                            mask1 = data_df[f"{basename}_{n_d}"] < min_val 
                            mask2 = data_df[f"{basename}_{n_d}"] > max_val 
                            mask = np.logical_or(mask1, mask2)
                            dr_temp[mask] = np.nan
                    elif basename == "dr":
                        mask1 = dr_temp < min_val
                        mask2 = dr_temp > max_val
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
                    elif basename == "pt_ratio":
                        pt_ratio = data_df[f"pt_{n_d}"]/ emb_df[f"pt_{n_e}"]
                        mask1 = pt_ratio < min_val
                        mask2 = pt_ratio > max_val
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
            dr_arr[:, n_d-1, n_e-1] = dr_temp

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


def apply_genmatching(dr_arr, df, switch_quantities):
    #switches data for those entries where an emb muon closer to the original one is present
    for n_event in range(len(df)):
        distances = dr_arr[n_event, :, :]
        
        muon1_id = find_closest_muon(distances[0, :])
        distances = remove_emb_mu_from_dist(distances, muon1_id)

        muon2_id = find_closest_muon(distances[1, :])
        
        #ignoring nans
        if np.isnan([muon1_id, muon2_id]).sum() == 0:
            #nothing to change if best fit is trivial
            if muon1_id != 0 and muon2_id != 1:
                for q_name in switch_quantities:
                    #check whether quantity should be updated
                    temp1 = df.loc[n_event, f"{q_name}_{muon1_id+1}"]
                    temp2 = df.loc[n_event, f"{q_name}_{muon2_id+1}"]
                    #updating quantity
                    df.loc[n_event, f"{q_name}_1"] = temp1
                    df.loc[n_event, f"{q_name}_2"] = temp2
    
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
    if not col_name.startswith("phi"):
        diff = col1 - col2
    #phi needs to be handled differently because the value must be lower than pi
    else:
        diff = col1 - col2
        mask1 = np.abs(diff) > np.pi
        mask2 = col1 >= col2

        mask = np.logical_and(mask1, mask2)
        diff[mask] = 2 * np.pi - diff[mask]

        mask = np.logical_and(mask1, ~mask2)
        diff[mask] = 2 * np.pi + diff[mask]

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