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

        events = {}

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
        events_df = pd.DataFrame(events)
        master_df = pd.concat([master_df, events_df], ignore_index=True)

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



def col_is_expanded(quantities, col):
    # checks in importing instructions whether a column is being expanded or not
    for q in quantities:
        if q["target"] == col:#q["target"] contains the column name 
            return q["expand"]#and this is the variable for setting expansion rules
        
    raise ValueError("Column not found")



def quality_cut(df, quantities, filter_dict):
    # this function applies a list of filters to the dataframe and removes cells (not rows!!) where the requirements are not being met
    # thereby the muon number and jet number of each event can be reduced 
    # note that only the given column is being set to nan and in another step the rest of the corresponding quantities are being removed

    for f in filter_dict:
        basename = f["col"]#column name the filter is being applied on 
        min_val = f["min"]
        max_val = f["max"]
        col_expanded = col_is_expanded(quantities, basename)

        # if column is expanded, there are multiple columns to correct
        if col_expanded:
            for col in df.columns:
                if col.startswith(basename):
                    df.loc[df[col] < min_val, col] = np.nan#sets those cells to nan where boundaries are missed
                    df.loc[df[col] > max_val, col] = np.nan
        else:
            df.loc[df[basename] < min_val, col] = np.nan#same as above with exact column names
            df.loc[df[basename] > max_val, col] = np.nan
    
    return df

    

def assert_object_validity(df, n_min):
    # this function ensures that muon and jet quantities do not contain any nans. this means that if there is a nan in e.g. eta of muon3, the whole muon is 
    # removed. same for jets

    for prefix in ["", "Jet_"]:#applying function on jet and muon quantities
        n_muon_col = get_n_occurence(df, f"{prefix}pt")#number of objects
        
        n_muon = np.zeros(len(df))

        for num in range(1, n_muon_col+1):
            #finding nans in objects
            iso_mask = df[f"{prefix}phi_{num}"].isna()
            eta_mask = df[f"{prefix}eta_{num}"].isna()
            pt_mask = df[f"{prefix}pt_{num}"].isna()
            m_mask = df[f"{prefix}m_{num}"].isna()

            #combination with logical or selects combines information of all masks
            mask = np.logical_or(iso_mask, eta_mask)
            mask = np.logical_or(mask, pt_mask)
            mask = np.logical_or(mask, m_mask)

            #setting all quantities to nan of incomplete captured objects
            df.loc[mask, f"{prefix}phi_{num}"] = np.nan
            df.loc[mask, f"{prefix}eta_{num}"] = np.nan
            df.loc[mask, f"{prefix}pt_{num}"] = np.nan
            df.loc[mask, f"{prefix}m_{num}"] = np.nan

            n_muon += np.where(mask, 0, 1)#counting how many muons are remaining

        df = df[n_muon >= n_min]#requiring at least 2 muons and jets

    return df


def only_global_muons(df):
    # checks the global flag for each available muon and sets its quantities to zero if it is not global

    n_muon = get_n_occurence(df, "MuonIsGlobal")

    for num in range(1, n_muon+1):
        mask = df[f"MuonIsGlobal_{num}"] != 1
        df.loc[mask, f"pt_{num}"] = np.nan
        df.loc[mask, f"eta_{num}"] = np.nan
        df.loc[mask, f"phi_{num}"] = np.nan
        df.loc[mask, f"m_{num}"] = np.nan

    return df


def get_n_occurence(df, basename):
    #returns the occurence of a column. pt_1, pt_2 is counted as pt
    n = 0
    for col in df.columns: 
        if col.startswith(basename):
            n += 1

    return n


def compactify_objects(df):
    # this function ensures that there a no empty objects in the dataset. if muon3 is empty but muon5 isnt, the quantities from muon5 are moved to the left.
    # at the end empty columns are deleted

    # repeating for jets and muons (prefix is the difference between a muon quantity such as pt and a jet quantity "Jet_pt" )
    for prefix in ["", "Jet_"]:
        n = get_n_occurence(df, f"{prefix}pt")#number of objects

        #this function takes all columns of a quantity such as pt as input and returns it with nans at the end: [1,2,nan,nan,3] -> [1,2,3,nan,nan]
        def shift_left(row):
            non_nans = row[~np.isnan(row)]
            return np.concatenate([non_nans, [np.nan] * (len(row) - len(non_nans))])

        q_length = []#array for applying a short consistency check

        for q in ["pt", "eta", "phi"]:

            q_l = []#will contain the lengths of the single quantitiy columns (q_1, q_2...)
            
            q_cols = [f'{prefix}{q}_{i}' for i in range(1, n+1)]#list of all relevant columns of the quantity
            subset = df[q_cols]#
            q_array = subset.values #2d array of the values columns of the dataframe belonging to a certain quantity

            q_array = np.apply_along_axis(shift_left, axis=1, arr=q_array)#shiftig to the left

            #setting the newly ordered columns
            for num, col in enumerate(q_cols):
                subarray = q_array[:, num]#single column of the dataframe
                l = np.sum(~np.isnan(subarray))#counting not nan entries
                if l > 0:    
                    df.loc[:, col] = subarray#setting array into column if there is data available
                    q_l.append(l)#tracking length
                else:
                    df = df.drop(columns=[col])#removing column if not
            q_length.append(q_l)#adding length array
        
        #now all columns with number n must have the same length (q1_1, q2_1...)
        for num in range(len(q_length[0])):#number of particles
            for mun in range(len(q_length)-1):#number of quantities
                assert q_length[mun][num] == q_length[mun+1][num], "Compactification failed"

    return df  