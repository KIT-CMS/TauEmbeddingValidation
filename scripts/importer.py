import uproot
import os
import pandas as pd
import numpy as np
import vector


def nanoaod_to_dataframe(data_path, quantities):
    # initialization
    nanoaod = uproot.open(data_path)

    nanoaod = nanoaod["Events;1"]

    events = pd.DataFrame()

    for quantity in quantities:
        #this is a column of a nanoaod
        column = quantity["key"]#
        target_name = quantity["target"]
        expand = quantity["expand"]

        temp = nanoaod[column].array(library="np")
                
        #can only expand nested arrays and not numbers - all entries of the column should be again arrays
        if expand:
            assert column_is_nested(temp), "Can only expand nested columns"
            # n_expand = quantity["n_expand"]
            # assert n_expand >= 1, "Need room to unpack"
            
            #which are then stored as flat columns
            length_array = get_subarray_length(temp)
            # max_length = min([np.amax(length_array), n_expand])
            max_length = np.amax(length_array)
        
            for num in range(max_length):
                events[f'{target_name}_{num+1}'] = get_nth_element(temp, num)

            # expanded_array = np.array([np.pad(sub_array, (0, max_length - len(sub_array)), constant_values=np.nan) for sub_array in temp])

            # events[[f'{target_name}_{i+1}' for i in range(max_length)]] = expanded_array
            # events[f"{target_name}_length"] = length_array

        else:
            #returning only the first column if expansion of nested array is not wanted
            if column_is_nested(temp):
                events[target_name] = get_nth_element(temp, 0)
            #returning the plain column if it is not nested
            else:
                events[target_name] = pd.Series(temp)

    return events
        
def get_nth_element(array, n):
    #returns the nth sub column if possible 
    if len(array) >= n+1:
        return array[n]
    #nan otherwise
    return np.nan

def column_is_nested(array):
    try:
        sub_length = len(array[0])
        return True
    except TypeError:
        return False

def get_subarray_length(array):
    return len(array)

def compare_cells(column1, column2):
    test = column1 - column2
    assert len(test[test!=0]) == 0, "Mismatch"

def generate_m_vis(pt_1, eta_1, phi_1, m_1, pt_2, eta_2, phi_2, m_2):
    p4_1 = vector.MomentumObject4D(pt=pt_1, phi=phi_1, eta=eta_1, mass=m_1)
    p4_2 = vector.MomentumObject4D(pt=pt_2, phi=phi_2, eta=eta_2, mass=m_2)
    p4_vis = p4_1 + p4_2
    m_vis = p4_vis.m

    return m_vis

def generate_pt_vis(pt_1, eta_1, phi_1, m_1, pt_2, eta_2, phi_2, m_2):
    p4_1 = vector.MomentumObject4D(pt=pt_1, phi=phi_1, eta=eta_1, mass=m_1)
    p4_2 = vector.MomentumObject4D(pt=pt_2, phi=phi_2, eta=eta_2, mass=m_2)
    p4_vis = p4_1 + p4_2
    pt_vis = p4_vis.pt

    return pt_vis

get_nth_element = np.vectorize(get_nth_element, otypes=[np.float32]) #function that extracts the 1st element of a subarray for flattening the data
get_subarray_length = np.vectorize(get_subarray_length, otypes=[np.int32]) #function that extracts the 1st element of a subarray for flattening the data
# generate_pt_vis = np.vectorize(generate_pt_vis, otypes=[np.float32])
# generate_m_vis = np.vectorize(generate_m_vis, otypes=[np.float32])



#following code is for genmatching
def calculate_dr(data_df, emb_df, n_data, n_emb, filter):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset

    dr_arr = np.full(shape=(len(data_df), n_data, n_emb), dtype=float, fill_value=np.nan)

    #looping over all data particle and embedding particle combinations
    for n_d in range(1, n_data+1):
        for n_e in range(1, n_emb+1):
            eta_diff = data_df[f"eta_{n_d}"] - emb_df[f"eta_{n_e}"]
            phi_diff = data_df[f"phi_{n_d}"] - emb_df[f"phi_{n_e}"]

            #calculating the dr value between them for all events
            dr_temp = np.sqrt(np.square(eta_diff) + np.square(phi_diff))

            if filter:
                #only respects muons with pt >27 gev and abs(eta)<2.5
                pt_mask_data = data_df[f"pt_{n_d}"] > 27
                pt_mask_emb = emb_df[f"pt_{n_d}"] > 27
                eta_mask_data = np.logical_and(data_df[f"eta_{n_d}"] < 2.5, data_df[f"eta_{n_d}"] > -2.5)
                eta_mask_emb = np.logical_and(emb_df[f"eta_{n_d}"] < 2.5, emb_df[f"eta_{n_d}"] > -2.5)

                #combining masks
                data_mask = np.logical_and(pt_mask_data, eta_mask_data)
                emb_mask = np.logical_and(pt_mask_emb, eta_mask_emb)
                mask = np.logical_and(data_mask, emb_mask)

                #replacement muon needs to have similar pt as the original one
                pt_ratio = data_df[f"pt_{n_d}"]/ emb_df[f"pt_{n_e}"]
                ratio_mask = np.logical_and(pt_ratio < 1.25, pt_ratio > 0.75)
                mask = np.logical_and(mask, ratio_mask)

                #the dr value of those muons that fail the filter are set to nan.
                dr_temp[mask] = np.nan

                #only respecting distances smaller this threshold because everything else must be a mismatch
                dr_temp[dr_temp>0.1] = np.nan


            dr_arr[:, n_d-1, n_e-1] = dr_temp

    return dr_arr



def find_closest_muon(dr_slice):
    nan_mask = ~np.isnan(dr_slice)

    if nan_mask.sum() > 0:
        index = np.nanargmin(dr_slice)
        return index
    return np.nan
        

def remove_emb_mu_from_dist(dist, id):
    # print(id)
    if ~np.isnan(id):
        dist[:, id] = np.nan
    return dist

def apply_genmatching(dr_arr, df, switch_quantities):

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


