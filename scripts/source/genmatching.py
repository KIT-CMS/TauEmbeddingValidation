import uproot
import os
import pandas as pd
import numpy as np
from source.helper import subtract_columns, get_n_occurence
from source.importer import compactify_objects, get_jet_basenames, get_muon_basenames, get_electron_basenames, get_photon_basenames



#following code is for genmatching
def calculate_dr(df, mode, filter=None, df2=None):
    #this function returns the dr value for all particle combination from embedding
    #the first "n_data" particles of data are compare to the first "n_emb" particles of the embeddign dataset
    #this function is a mess and i know it, however it grew like this and there is no time left to improve its readability
    # i would suggest that instead of df and df2 dicts such as {"eta":[[values], [values]], "phi":[[values], [values]]} are given as argument for both master and target
    # thereby the whole naming issues are no longer a problem and the function is more easily readable
    if mode == "muon":
        n_comp = get_n_occurence(df, "Muon_eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "jet":
        n_comp = get_n_occurence(df, "Jet_eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "electron":
        n_comp = get_n_occurence(df, "Electron_eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "photon":
        n_comp = get_n_occurence(df, "Photon_eta")
        n_target = 2
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "filter":
        n_comp = 2
        n_target = get_n_occurence(df, "Jet_eta")
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "muon_all":
        n_comp = get_n_occurence(df2, "Muon_eta")
        n_target = get_n_occurence(df, "Muon_eta")
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "jet_all":
        # df: emb, df2: data
        n_comp = get_n_occurence(df2, "Jet_eta") # data
        n_target = get_n_occurence(df, "Jet_eta") # emb
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "electron_all":
        # df: emb, df2: data
        n_comp = get_n_occurence(df2, "Electron_eta") # data
        n_target = get_n_occurence(df, "Electron_eta") # emb
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    elif mode == "photon_all":
        # df: emb, df2: data
        n_comp = get_n_occurence(df2, "Photon_eta") # data
        n_target = get_n_occurence(df, "Photon_eta") # emb
        dr_arr = np.full(shape=(len(df), n_target, n_comp), dtype=float, fill_value=np.nan)
    else:
        raise ValueError("Invalid mode selected")

    #looping over all data particle and embedding particle combinations
    for n in range(1, n_target+1):
        #if dr should be calculated between muon, the columns to be used are different from the columns in the jet case. the following clauses assign the names of the columns based on the mode. 
        if mode=="muon":
            comp_phi = "Muon_phi"#this is the name of the columns that the 2 relevant muons are being compared to (simply all muon columns)
            comp_eta = "Muon_eta"
            comp_pt = "Muon_pt"
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
        elif mode=="electron":
            comp_phi = "Electron_phi"
            comp_eta = "Electron_eta"
            comp_pt = "Electron_pt"
            if n == 1:
                master_eta = "LE_eta"#comparing the first jet
                master_phi = "LE_phi"
                master_pt = "LE_pt"
            elif n == 2:
                master_eta = "TE_eta"#then comparing the second jet
                master_phi = "TE_phi"
                master_pt = "TE_pt"
        elif mode=="photon":
            comp_phi = "Photon_phi"
            comp_eta = "Photon_eta"
            comp_pt = "Photon_pt"
            if n == 1:
                master_eta = "LP_eta"#comparing the first jet
                master_phi = "LP_phi"
                master_pt = "LP_pt"
            elif n == 2:
                master_eta = "TP_eta"#then comparing the second jet
                master_phi = "TP_phi"
                master_pt = "TP_pt"
        elif mode=="filter":
            master_eta = "Jet_eta"
            master_phi = "Jet_phi"
            master_pt = "Jet_pt"
            if n == 1:
                comp_phi = "LM_phi"
                comp_eta = "LM_eta"
                comp_pt = "LM_pt"
            elif n == 2:
                comp_phi = "TM_phi"
                comp_eta = "TM_eta"
                comp_pt = "TM_pt"
        # emb auf axis=1, data auf axis=2
        elif mode == "muon_all":
            master_eta = "Muon_eta"
            master_phi = "Muon_phi"
            master_pt = "Muon_pt"
            comp_phi = "Muon_phi"
            comp_eta = "Muon_eta"
            comp_pt = "Muon_pt"
        elif mode == "jet_all":
            master_eta = "Jet_eta"
            master_phi = "Jet_phi"
            master_pt = "Jet_pt"
            comp_phi = "Jet_phi"
            comp_eta = "Jet_eta"
            comp_pt = "Jet_pt"
        elif mode == "electron_all":
            master_eta = "Electron_eta"
            master_phi = "Electron_phi"
            master_pt = "Electron_pt"
            comp_phi = "Electron_phi"
            comp_eta = "Electron_eta"
            comp_pt = "Electron_pt"
        elif mode == "photon_all":
            master_eta = "Photon_eta"
            master_phi = "Photon_phi"
            master_pt = "Photon_pt"
            comp_phi = "Photon_phi"
            comp_eta = "Photon_eta"
            comp_pt = "Photon_pt"
        
        special_cases = ["dr", "pt_ratio", "LM_pt", "TM_pt"]

        for n_m in range(1, n_comp+1):
            if mode == "filter":
                eta_diff = subtract_columns(df[f"{master_eta}_{n}"], df[f"{comp_eta}"], "eta_")
                phi_diff = subtract_columns(df[f"{master_phi}_{n}"], df[f"{comp_phi}"], "phi_")
            elif mode.endswith("_all"):
                eta_diff = subtract_columns(df[f"{master_eta}_{n}"], df2[f"{comp_eta}_{n_m}"], "eta_")
                phi_diff = subtract_columns(df[f"{master_phi}_{n}"], df2[f"{comp_phi}_{n_m}"], "phi_")
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
                    if basename not in special_cases:#in this case the filter is applied on existing columns from df
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
                    elif basename == "LM_pt" and n_target==1:
                        mask1 = df[f"Muon_pt_{n_m}"] < min_val
                        mask2 = df[f"Muon_pt_{n_m}"] > max_val
                        mask = np.logical_or(mask1, mask2)
                        dr_temp[mask] = np.nan
                    elif basename == "TM_pt" and n_target==2:
                        mask1 = df[f"Muon_pt_{n_m}"] < min_val
                        mask2 = df[f"Muon_pt_{n_m}"] > max_val
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

def apply_genmatching(dr_arr, df, mode, purge=False):
    #switches data for those entries where an emb muon closer to the original one is present

    if mode == "muon":
        pt_source = "Muon_pt"
        eta_source = "Muon_eta"
        phi_source = "Muon_phi"
        m_source = "Muon_m"
        pt_target_1 = "LM_pt"
        pt_target_2 = "TM_pt"
        eta_target_1 = "LM_eta"
        eta_target_2 = "TM_eta"
        phi_target_1 = "LM_phi"
        phi_target_2 = "TM_phi"
        m_target_1 = "LM_m"
        m_target_2 = "TM_m"
        # basenames = get_muon_basenames()
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
        # basenames = get_jet_basenames()
    elif mode == "electron":
        pt_source = "Electron_pt"
        eta_source = "Electron_eta"
        phi_source = "Electron_phi"
        m_source = "Electron_m"
        pt_target_1 = "LE_pt"
        pt_target_2 = "TE_pt"
        eta_target_1 = "LE_eta"
        eta_target_2 = "TE_eta"
        phi_target_1 = "LE_phi"
        phi_target_2 = "TE_phi"
        m_target_1 = "LE_m"
        m_target_2 = "TE_m"
        # basenames = get_electron_basenames()
    elif mode == "photon":
        pt_source = "Photon_pt"
        eta_source = "Photon_eta"
        phi_source = "Photon_phi"
        m_source = "Photon_m"
        pt_target_1 = "LP_pt"
        pt_target_2 = "TP_pt"
        eta_target_1 = "LP_eta"
        eta_target_2 = "TP_eta"
        phi_target_1 = "LP_phi"
        phi_target_2 = "TP_phi"
        # m_target_1 = "LP_m"
        # m_target_2 = "TP_m"
        # basenames = get_photon_basenames()
    else:
        raise ValueError("invalid mode selected")
    
    target_length = len(df)
    lm_pt = np.full(target_length, fill_value=np.nan)
    tm_pt = np.full(target_length, fill_value=np.nan)
    lm_phi = np.full(target_length, fill_value=np.nan)
    tm_phi = np.full(target_length, fill_value=np.nan)
    lm_eta = np.full(target_length, fill_value=np.nan)
    tm_eta = np.full(target_length, fill_value=np.nan)
    if not mode =="photon":
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

            dr1 = distances[0,muon_id]#distances to the leading muon
            dr2 = distances[1,muon_id]#distances to the subleading muon
            
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
            if not mode =="photon":
                lm_m[n_event] = event[f"{m_source}_{muon1_id+1}"]
            muon_best_fit[n_event, 0] = muon1_id
            dr_min[n_event, 0] = distances2[0, muon1_id]

            # if purge:
            #     col_list = [f"{bn}_{muon1_id+1}" for bn in basenames]
            #     df.loc[n_event, col_list] = np.nan
        else:
            lm_pt[n_event] = np.nan
            lm_eta[n_event] = np.nan
            lm_phi[n_event] = np.nan
            if not mode =="photon":
                lm_m[n_event] = np.nan
            muon_best_fit[n_event, 0] = np.nan
            dr_min[n_event, 0] = np.nan

        #setting the new value if a valid one could be found - otherwise nan is set
        if ~np.isnan(muon2_id):
            tm_pt[n_event] = event[f"{pt_source}_{muon2_id+1}"]
            tm_eta[n_event] = event[f"{eta_source}_{muon2_id+1}"]
            tm_phi[n_event] = event[f"{phi_source}_{muon2_id+1}"]
            if not mode =="photon":
                tm_m[n_event] = event[f"{m_source}_{muon2_id+1}"]
            muon_best_fit[n_event, 1] = muon2_id
            dr_min[n_event, 1] = distances2[1, muon2_id]

            # if purge:
            #     col_list = [f"{bn}_{muon2_id+1}" for bn in basenames]
            #     df.loc[n_event, col_list] = np.nan
        else:
            tm_pt[n_event] = np.nan
            tm_eta[n_event] = np.nan
            tm_phi[n_event] = np.nan
            if not mode =="photon":
                tm_m[n_event] = np.nan

    matched_df = pd.DataFrame({
        f"{pt_target_1}": pd.Series(lm_pt),
        f"{pt_target_2}": pd.Series(tm_pt),
        f"{eta_target_1}": pd.Series(lm_eta),
        f"{eta_target_2}": pd.Series(tm_eta),
        f"{phi_target_1}": pd.Series(lm_phi),
        f"{phi_target_2}": pd.Series(tm_phi),
        # f"{m_target_1}": pd.Series(lm_m),
        # f"{m_target_2}": pd.Series(tm_m)
    })
    df = pd.concat([df, matched_df], axis=1)
    if mode != "photon":
        m_df = pd.DataFrame({
            f"{m_target_1}": pd.Series(lm_m),
            f"{m_target_2}": pd.Series(tm_m)
        })
        df = pd.concat([df, m_df],  axis=1)
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


def remove_muon_jets(df, dr_arr, cut):
    # removes those jets that are closer than "value" to a muon

    basenames = get_jet_basenames()
    for n_j in range(dr_arr.shape[1]):
        for n_m in range(dr_arr.shape[2]):
            subset = dr_arr[:,n_j,n_m]
            mask = subset<cut
            for bn in basenames:
                df.loc[mask, f"{bn}_{n_j+1}"] = np.nan

    # mask = dr_arr[:,0,0] < cut
    # mask2 = dr_arr[:,0,1] < cut
    # mask = np.logical_or(mask, mask2)

    # df = df.loc[mask]
    
    return df

def remove_zmumu_candidates(df, dr_arr, cut):
    # removes those jets that are closer than "value" to a muon
    basenames = get_muon_basenames()
    for n_j in range(dr_arr.shape[1]):
        for n_m in range(dr_arr.shape[2]):
            subset = dr_arr[:,n_j,n_m]
            mask = subset<cut
            for bn in basenames:
                df.loc[mask, f"{bn}_{n_j+1}"] = np.nan

    # mask = dr_arr[:,0,0] < cut
    # mask2 = dr_arr[:,0,1] < cut
    # mask = np.logical_or(mask, mask2)

    # df = df.loc[mask]
    
    return df


# print(len(data_df), len(emb_df))# def remove_obj(df, ids, basenames):
#     assert ids.shape[1] == 2
#     ids += 1
#     for event in range(ids.shape[0]):
#         temp1 = ids[event,0]
#         temp2 = ids[event,1]
#         if ~np.isnan(temp1):
#             temp1 = int(temp1)
#             col_list = [f"{bn}_{temp1}" for bn in basenames]
#             df.loc[event, col_list] = np.nan
#         if ~np.isnan(temp2):
#             temp2 = int(temp2)
#             col_list = [f"{bn}_{temp2}" for bn in basenames]
#             df.loc[event, col_list] = np.nan

#     return df

def remove_non_muon_jets(df, dr_arr, cut):
    # leaves only muon jets
    counter = 0
    mask = dr_arr<cut
    mask = np.any(mask, axis=2)

    for n_j in range(dr_arr.shape[1]):

        subset = ~mask[:,n_j]

        df.loc[subset, f"Jet_eta_{n_j+1}"] = np.nan
        df.loc[subset, f"Jet_m_{n_j+1}"] = np.nan
        df.loc[subset, f"Jet_phi_{n_j+1}"] = np.nan
        df.loc[subset, f"Jet_pt_{n_j+1}"] = np.nan
        counter += np.sum(subset)

    print(f"removed {counter} jets from {len(df)} events")

    # mask = dr_arr[:,0,0] < cut
    # mask2 = dr_arr[:,0,1] < cut
    # mask = np.logical_or(mask, mask2)

    # df = df.loc[mask]
    
    return df

def remove_nonmatches(df1, df2, mode):
    # removes those jets which were supposed to be matched but couldn't for some some reason
    if mode == "jet":
        basename = "J"
    elif mode == "electron":
        basename = "E"
    elif mode == "photon":
        basename = "P"
    else:
        raise ValueError

    mask1 = df1[f"L{basename}_eta"].isna()
    mask2 = df2[f"L{basename}_eta"].isna()
    mask = np.logical_or(mask1, mask2)

    df1.loc[mask, [f"L{basename}_pt", f"L{basename}_eta", f"L{basename}_phi", f"L{basename}_m"]] = np.nan
    df2.loc[mask, [f"L{basename}_pt", f"L{basename}_eta", f"L{basename}_phi", f"L{basename}_m"]] = np.nan

    mask1 = df1[f"T{basename}_eta"].isna()
    mask2 = df2[f"T{basename}_eta"].isna()
    mask = np.logical_or(mask1, mask2)

    df1.loc[mask, [f"T{basename}_pt", f"T{basename}_eta", f"T{basename}_phi", f"T{basename}_m"]] = np.nan
    df2.loc[mask, [f"T{basename}_pt", f"T{basename}_eta", f"T{basename}_phi", f"T{basename}_m"]] = np.nan

    return df1, df2


def find_unmatchable_objects(dr, emb_df, data_df, mode, cut):

    if mode=="jet_all":
        n_emb = get_n_occurence(emb_df, "Jet_eta_")
        n_data = get_n_occurence(data_df, "Jet_eta_")
        pt_cols_emb = [f"Jet_pt_{n}" for n in range(1,n_emb+1)]
        pt_cols_data = [f"Jet_pt_{n}" for n in range(1,n_data+1)]
        eta_cols_emb = [f"Jet_eta_{n}" for n in range(1,n_emb+1)]
        eta_cols_data = [f"Jet_eta_{n}" for n in range(1,n_data+1)]
        phi_cols_emb = [f"Jet_phi_{n}" for n in range(1,n_emb+1)]
        phi_cols_data = [f"Jet_phi_{n}" for n in range(1,n_data+1)]
        basenames = ["Jet_eta", "Jet_phi", "Jet_pt"]
    elif mode == "muon_all":
        n_emb = get_n_occurence(emb_df, "Muon_eta_")
        n_data = get_n_occurence(data_df, "Muon_eta_")
        pt_cols_emb = [f"Muon_pt_{n}" for n in range(1,n_emb+1)]
        pt_cols_data = [f"Muon_pt_{n}" for n in range(1,n_data+1)]
        eta_cols_emb = [f"Muon_eta_{n}" for n in range(1,n_emb+1)]
        eta_cols_data = [f"Muon_eta_{n}" for n in range(1,n_data+1)]
        phi_cols_emb = [f"Muon_phi_{n}" for n in range(1,n_emb+1)]
        phi_cols_data = [f"Muon_phi_{n}" for n in range(1,n_data+1)]
        basenames = ["Muon_eta", "Muon_phi", "Muon_pt"]
    elif mode == "electron_all":
        n_emb = get_n_occurence(emb_df, "Electron_eta_")
        n_data = get_n_occurence(data_df, "Electron_eta_")
        pt_cols_emb = [f"Electron_pt_{n}" for n in range(1,n_emb+1)]
        pt_cols_data = [f"Electron_pt_{n}" for n in range(1,n_data+1)]
        eta_cols_emb = [f"Electron_eta_{n}" for n in range(1,n_emb+1)]
        eta_cols_data = [f"Electron_eta_{n}" for n in range(1,n_data+1)]
        phi_cols_emb = [f"Electron_phi_{n}" for n in range(1,n_emb+1)]
        phi_cols_data = [f"Electron_phi_{n}" for n in range(1,n_data+1)]
        basenames = ["Electron_eta", "Electron_phi", "Electron_pt"]
    elif mode == "photon_all":
        n_emb = get_n_occurence(emb_df, "Photon_eta_")
        n_data = get_n_occurence(data_df, "Photon_eta_")
        pt_cols_emb = [f"Photon_pt_{n}" for n in range(1,n_emb+1)]
        pt_cols_data = [f"Photon_pt_{n}" for n in range(1,n_data+1)]
        eta_cols_emb = [f"Photon_eta_{n}" for n in range(1,n_emb+1)]
        eta_cols_data = [f"Photon_eta_{n}" for n in range(1,n_data+1)]
        phi_cols_emb = [f"Photon_phi_{n}" for n in range(1,n_emb+1)]
        phi_cols_data = [f"Photon_phi_{n}" for n in range(1,n_data+1)]
        basenames = ["Photon_eta", "Photon_phi", "Photon_pt"]
    else:
        raise ValueError("Invalid mode")
    
    data_unmatched = data_df[pt_cols_data + eta_cols_data + phi_cols_data + ["run", "lumi", "event"]].copy(deep=True)
    emb_unmatched = emb_df[pt_cols_emb + eta_cols_emb + phi_cols_emb + ["run", "lumi", "event"]].copy(deep=True)

    
    dr[dr > cut] = np.nan

    for n_event in range(dr.shape[0]):
        subset = dr[n_event, :, :]#this array contains the distances between all obj in data and all in emb
        while (~np.isnan(subset)).sum() > 0:#only proceeding if there matches
            with np.errstate(all="ignore"):   # suppress warnings
                result = np.nanargmin(subset)   #finds the minimum
                n_e, n_d = np.unravel_index(result, subset.shape)#converts the number into the array index
            #removing muons from dataset
            for bn in basenames:
                data_unmatched.loc[n_event, f"{bn}_{n_d+1}"] = np.nan
                emb_unmatched.loc[n_event, f"{bn}_{n_e+1}"] = np.nan
            # removing all entries along an axis in distance array so that the muons cant be matched twice
            subset[n_e, :] = np.nan
            subset[:, n_d] = np.nan
            


    # print(np.sum(np.isnan(data_unmatched.values)), np.sum(~np.isnan(data_unmatched.values)))
    # print(data_unmatched.columns)
    data_unmatched = compactify_objects(data_unmatched, basenames, n_data)
    emb_unmatched = compactify_objects(emb_unmatched, basenames, n_emb)
    # print(data_unmatched.columns)
    return data_unmatched, emb_unmatched
