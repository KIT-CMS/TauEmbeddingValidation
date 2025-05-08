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

def calculate_dr(data_df, emb_df, n_data, n_emb):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset

    dr_arr = np.full(shape=(len(data_df), n_data, n_emb), dtype=float, fill_value=np.nan)

    #looping over all data particle and embedding particle combinations
    for n_d in range(1, n_data+1):
        for n_e in range(1, n_emb+1):
            eta_diff = data_df[f"eta_{n_d}"] - emb_df[f"eta_{n_e}"]
            phi_diff = data_df[f"phi_{n_d}"] - emb_df[f"phi_{n_e}"]
            #calculating the dr value between them for all events
            dr_arr[:, n_d-1, n_e-1] = np.sqrt(np.square(eta_diff) + np.square(phi_diff))

    return dr_arr


def find_closest_emb_particle(dr_arr, particle_id):
    #this function returns the id of the embedding particle that lies closest to the data partile with id "particle_id"

    #num of events
    n_events = dr_arr.shape[0]

    #particle num starts from 0 while the id starts from 1
    particle_num = particle_id-1

    min_index = - np.full(shape=(n_events), fill_value=99, dtype=int)
    min_dr = - np.full(shape=(n_events), fill_value=99., dtype=float)

    for n_e in range(n_events):
        #selecting the distances from the particle to all available embedding particles      
        dr_arr_temp = dr_arr[n_e, particle_num, :]
        try:
            #finding the index of the closest particle in the array
            min_index[n_e] = np.nanargmin(dr_arr_temp)
            min_dr[n_e] = np.nanmin(dr_arr_temp)
        except ValueError:
            # min_index[n_e, n_p] = None
            pass

    #converting the index to id
    min_index[min_index!=-99] += 1

    return min_index, min_dr


def apply_dr_matching(master_df, df, quantities):
    #this function finds swaps the particles in df wherever there is another muon that lies closer to the original muon

    dr = calculate_dr(master_df, df, 2, 2)
    index_1, dr_1 = find_closest_emb_particle(dr, particle_id=1)
    index_2, dr_2 = find_closest_emb_particle(dr, particle_id=2)

    #these are the events where swapping would be conceptually allowed
    mask1 = (index_1-index_2)!=0
    mask1[index_1==99] = False
    mask1[index_2==99] = False

    #these are the events where muon_1 and 2 actually need to be swapped
    mask2 = np.logical_and(index_1==2, index_2==1)

    #and these are the events where the events are being swappend
    mask = np.logical_and(mask1, mask2)

    for quantity in quantities:
        if quantity["dr_matching"]:
            q_name = quantity["target"]
            q1_temp = df[f"{q_name}_1"].copy(deep=True)
            q2_temp = df[f"{q_name}_2"].copy(deep=True)
            df[f"{q_name}_1"] = np.where(mask, q2_temp, q1_temp)
            df[f"{q_name}_2"] = np.where(mask, q1_temp, q2_temp)

    return df