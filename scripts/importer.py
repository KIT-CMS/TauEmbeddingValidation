import uproot
import os
import pandas as pd
import numpy as np
import vector
from pathlib import Path
import shutil



def nanoaod_to_dataframe(files, quantities):
    #imports all files in files and concatenates them
    master_df = pd.DataFrame()

    # initialization
    for path in files:
        print(f"Reading: {path}")
        nanoaod = uproot.open(path)

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
        master_df = pd.concat([master_df, events], ignore_index=True)

    master_df = master_df.drop_duplicates(["event", "lumi", "run"])

    return master_df
        
def get_nth_element(array, n):
    #returns the nth sub column if possible 
    if len(array) >= n+1:
        return array[n]
    #nan otherwise
    return np.nan

def column_is_nested(array):
    #returns whether elements in cells are arrays or scalars
    try:
        sub_length = len(array[0])
        return True
    except TypeError:
        return False

def get_subarray_length(array):
    return len(array)


get_nth_element = np.vectorize(get_nth_element, otypes=[np.float32]) #function that extracts the 1st element of a subarray for flattening the data
get_subarray_length = np.vectorize(get_subarray_length, otypes=[np.int32]) #function that extracts the 1st element of a subarray for flattening the data


def get_z_m_pt(df):
    m_vis = generate_m_vis(df[f"LM_pt"], df[f"LM_eta"], df[f"LM_phi"], df[f"LM_m"], df[f"TM_pt"], df[f"TM_eta"], df[f"TM_phi"], df[f"TM_m"])
    pt_vis = generate_pt_vis(df[f"LM_pt"], df[f"LM_eta"], df[f"LM_phi"], df[f"LM_m"], df[f"TM_pt"], df[f"TM_eta"], df[f"TM_phi"], df[f"TM_m"])

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

