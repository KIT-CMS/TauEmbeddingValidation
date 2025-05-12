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


get_nth_element = np.vectorize(get_nth_element, otypes=[np.float32]) #function that extracts the 1st element of a subarray for flattening the data
get_subarray_length = np.vectorize(get_subarray_length, otypes=[np.int32]) #function that extracts the 1st element of a subarray for flattening the data


def get_z_m_pt(df):
    #finds for each event the muon pair that fits best to the z boson mass. returns arrays with the mass and pt of the best fitting pair
    m_z = 91.1880
    n_muon = get_nmuon(df, "eta_")
    n_events = len(df)
    m_vis = np.full((n_events,n_muon,n_muon), np.nan)
    pt_vis = np.full((n_events,n_muon,n_muon), np.nan)

    #calculates mvis and pt for each muon pair
    for n1 in range(1, n_muon+1):
        for n2 in range(1, n_muon+1):
            m_vis[:,n1-1, n2-1] = generate_m_vis(df[f"pt_{n1}"], df[f"eta_{n1}"], df[f"phi_{n1}"], df[f"m_{n1}"], df[f"pt_{n2}"], df[f"eta_{n2}"], df[f"phi_{n2}"], df[f"m_{n2}"])
            pt_vis[:,n1-1, n2-1] = generate_pt_vis(df[f"pt_{n1}"], df[f"eta_{n1}"], df[f"phi_{n1}"], df[f"m_{n1}"], df[f"pt_{n2}"], df[f"eta_{n2}"], df[f"phi_{n2}"], df[f"m_{n2}"])
    
    #calculates difference from z boson mass
    z_difference = np.absolute(np.copy(m_vis-m_z))

    #finds index of best fitting muon pair for each event
    min_indices = np.nanargmin(z_difference.reshape(n_events, -1), axis=1)

    #converts previously reshaped array into to 2d array
    row_col_indices = np.array([np.unravel_index(idx, (n_muon, n_muon)) for idx in min_indices])

    #extracts values of ideal muon pairs
    m_vis = m_vis[np.arange(n_events), row_col_indices[:, 0], row_col_indices[:, 1]]
    pt_vis = pt_vis[np.arange(n_events), row_col_indices[:, 0], row_col_indices[:, 1]]
    
    return m_vis, pt_vis


def get_nmuon(df, column):
    #returns the number of muons that is available in the dataset (the highest index that can be found behind the column base name)
    max_n = 0
    for col in df.columns:
        if col.startswith(column):
            n = col.rsplit("_", 1)[1]
            n = int(n)
            if n > max_n:
                max_n = n
    return max_n

def generate_m_vis(pt_1, eta_1, phi_1, m_1, pt_2, eta_2, phi_2, m_2):
    #calculates m_vis of a muon pair
    p4_1 = vector.MomentumObject4D(pt=pt_1, phi=phi_1, eta=eta_1, mass=m_1)
    p4_2 = vector.MomentumObject4D(pt=pt_2, phi=phi_2, eta=eta_2, mass=m_2)
    p4_vis = p4_1 + p4_2
    m_vis = p4_vis.m

    return m_vis

def generate_pt_vis(pt_1, eta_1, phi_1, m_1, pt_2, eta_2, phi_2, m_2):
    #calculates pt_vis of a muon pair
    p4_1 = vector.MomentumObject4D(pt=pt_1, phi=phi_1, eta=eta_1, mass=m_1)
    p4_2 = vector.MomentumObject4D(pt=pt_2, phi=phi_2, eta=eta_2, mass=m_2)
    p4_vis = p4_1 + p4_2
    pt_vis = p4_vis.pt

    return pt_vis




#following code is for genmatching
def calculate_dr(data_df, emb_df, n_data, n_emb, filter=None):
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


def verify_events(*data):
    #checks whether all dataframes have the same order
    master_df = data[0]

    for df in data:
        compare_cells(master_df["event"].values, df["event"].values)
        compare_cells(master_df["lumi"].values, df["lumi"].values)
        compare_cells(master_df["run"].values, df["run"].values)


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