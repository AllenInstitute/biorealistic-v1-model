import os
import pickle
import numpy as np


def generate_random_positions(N, layer_range, radial_range):
    radius_outer = radial_range[1]
    radius_inner = radial_range[0]

    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_outer ** 2 - radius_inner ** 2) * np.random.random([N]) + radius_inner ** 2)
    x = r * np.cos(phi)
    z = r * np.sin(phi)

    layer_start = layer_range[0]
    layer_end = layer_range[1]
    # Generate N random z values.
    y = (layer_end - layer_start) * np.random.random([N]) + layer_start

    positions = np.column_stack((x, y, z))

    return positions


def generate_positions_grids(N, x_grids, y_grids, x_len, y_len):
    widthPerTile = x_len/x_grids
    heightPerTile = y_len/y_grids

    X = np.zeros(N * x_grids * y_grids)
    Y = np.zeros(N * x_grids * y_grids)

    counter = 0
    for i in range(x_grids):
        for j in range(y_grids):
            x_tile = np.random.uniform(i*widthPerTile,  (i+1) * widthPerTile,  N)
            y_tile = np.random.uniform(j*heightPerTile, (j+1) * heightPerTile, N)
            X[counter*N:(counter+1)*N] = x_tile
            Y[counter*N:(counter+1)*N] = y_tile
            counter = counter + 1
    return np.column_stack((X, Y))


def get_filter_spatial_size(N, X_grids, Y_grids, size_range):
    spatial_sizes = np.zeros(N * X_grids * Y_grids)
    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            if len(size_range) == 1:
                sizes = np.ones(N) * size_range[0]
            else:
                sizes = np.random.triangular(size_range[0], size_range[0] + 1, size_range[1], N)
            spatial_sizes[counter * N:(counter + 1) * N] = sizes
            counter = counter + 1
    return spatial_sizes


def open_pickle(pkl_file):
    f = open(pkl_file, 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    return u.load()


def get_filter_temporal_params(N, X_grids, Y_grids, model):
    # Total number of cells
    N_total = N * X_grids * Y_grids

    # Jitter parameters
    jitter = 0.025
    lower_jitter = 1 - jitter
    upper_jitter = 1 + jitter

    # Directory of pickle files with saved parameter values
    basepath = 'node_props/lgn_fitted_models/'

    # For two-subunit filter (sONsOFF and sONtOFF)
    sOFF_prs = open_pickle(os.path.join(basepath, 'sOFF_TF4_3.5_-2.0_10.0_60.0_15.0_ic.pkl'))  # best chosen fit for sOFF 4 Hz
    tOFF_prs = open_pickle(os.path.join(basepath, 'tOFF_TF8_4.222_-2.404_8.545_23.019_0.0_ic.pkl'))  # best chosen fit for tOFF 8 Hz
    sON_prs = open_pickle(os.path.join(basepath, 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.pkl'))  # best chosen fit for sON 4 Hz

    # Choose cell type and temporal frequency
    if model == 'sONsOFF_001':

        kpeaks = sOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks =  sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = sOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts =  sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = sOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays =  sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 6.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    elif model == 'sONtOFF_001':
        kpeaks = tOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks = sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = tOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts = sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = tOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays = sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 4.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    else:
        cell_type = model[0: model.find('_')]    #'sON'  # 'tOFF'
        tf_str = model[model.find('_') + 1:]

        # Load pickle file containing params for optimized temporal kernel, it it exists
        file_found = 0
        for fname in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, fname)):
                pkl_savename = os.path.join(basepath, fname)
                if (tf_str in pkl_savename.split('_') and pkl_savename.find(cell_type) >= 0 and pkl_savename.find('.pkl') >= 0):
                    file_found = 1
                    print(pkl_savename)
                    filt_file = pkl_savename

        if file_found != 1:
            print('File not found: Filter was not optimized for this sub-class')

        # savedata_dict = pickle.load(open(filt_file, 'rb'))
        savedata_dict = open_pickle(filt_file)  # pickle.load(open(filt_file, 'rb'))

        kpeaks = savedata_dict['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks_non_dom_0 = np.nan * np.zeros(N_total)
        kpeaks_non_dom_1 = np.nan * np.zeros(N_total)

        wts = savedata_dict['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts_non_dom_0 = np.nan * np.zeros(N_total)
        wts_non_dom_1 = np.nan * np.zeros(N_total)

        delays = savedata_dict['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays_non_dom_0 = np.nan * np.zeros(N_total)
        delays_non_dom_1 = np.nan * np.zeros(N_total)

        sf_sep = np.nan * np.zeros(N_total)
        tuning_angles =  np.nan * np.zeros(N_total)

    return np.column_stack((kpeaks_dom_0, kpeaks_dom_1, wts_dom_0, wts_dom_1, delays_dom_0, delays_dom_1,
                            kpeaks_non_dom_0, kpeaks_non_dom_1, wts_non_dom_0, wts_non_dom_1,
                            delays_non_dom_0, delays_non_dom_1, tuning_angles, sf_sep))

