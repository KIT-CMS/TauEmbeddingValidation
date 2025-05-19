import uproot
import os
import pandas as pd
import numpy as np
import vector
from pathlib import Path
import shutil



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


def verify_events(*data):
    #checks whether all dataframes have the same order
    master_df = data[0]

    for df in data:
        compare_cells(master_df["event"].values, df["event"].values)
        compare_cells(master_df["lumi"].values, df["lumi"].values)
        compare_cells(master_df["run"].values, df["run"].values)


def initialize_dir(base_path: str, subfolders: list[str]=None):

    base_dir = Path(base_path)

    # Delete the folder if it exists
    if base_dir.exists():
        shutil.rmtree(base_dir)

    # Recreate the base directory
    base_dir.mkdir(parents=True, exist_ok=True)

    if type(subfolders)!=type(None):
        # Create subfolders 
        for sub in subfolders:
            (base_dir / sub).mkdir(parents=True, exist_ok=True)


def create_concordant_subsets(df1, df2):
    l1 = len(df1)
    l2 = len(df1)

    keys = ["run", "lumi", "event"]
    df1 = df1.sort_values(by=keys, ignore_index=True)
    df2 = df2.sort_values(by=keys, ignore_index=True)

    mask = df1[keys].merge(df2[keys], how="inner")

    l3 = len(mask)
    print(f"Previous lengths: {l1}, {l2} - New length: {l3}")

    df1 = df1.merge(mask, how="inner")
    df2 = df2.merge(mask, how="inner")

    return df1, df2

def copy_columns_from_to(from_df, to_df, columns):
    cols_to_copy = from_df[columns].copy(deep=True)
    to_df[columns] = cols_to_copy

    return from_df, to_df