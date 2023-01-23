"""Query data like optimization parameters and results. Intended to decouple underlying data storage mechanism
and IK-EMOViz."""
import os
import h5py
import json
import warnings
import numpy as np


# This is a very specific way of implementing a query-based system. Ideally, we should have some OOP-based function
# where user defines the results returned by a list of mandatory queries (keys of the QUERY dict).

QUERY = {'N_OBJ': 'n_obj',
         'N_VAR': 'n_var',
         'X_MIN': 'xl',
         'X_MAX': 'xu',
         'MAX_ITER': 'ngen',
         'POP_SIZE': 'pop_size',
         'VAR_LABELS': 'var_labels',
         'OBJ_LABELS': 'obj_label',
         'VARS_IGNORED': 'ignore_vars',
         'INNOV_LATEST_GEN': 'innov_info_latest_gen'}
# Standard queries -> not used currently
PARAM_QUERY = (
    'N_OBJ',
    'N_VAR',
    'X_MIN',
    'X_MAX',
    'MAX_ITER',
    'POP_SIZE'
    'VAR_LABELS',
    'OBJ_LABELS',
    'VARS_IGNORED',
    'INNOV_LATEST_GEN'
)
DATA_QUERY = (
    'F',
    'X',
    'G',
    'RANK',
)


class DemoQuery:
    """A query class example that reads from a demo query."""
    def __init__(self, res_folder):
        self.res_folder = res_folder
        if not os.path.exists(res_folder):
            warnings.warn(f"Path {res_folder} does not exist.")

    def get_iter_data(self, iter, *args):
        """Data pertaining to specific iteration."""
        iter_data = []
        gen_key = f'gen{iter}'
        with h5py.File(os.path.join(self.res_folder, 'optim_state.hdf5'), 'r', libver='latest', swmr=True) as hf:
            current_gen_data = hf[gen_key]
            # TODO: Check for valid query
            for query in args:
                iter_data.append(np.array(current_gen_data[query]))

        return iter_data

    def get(self, *args):
        optim_args_file = os.path.join(self.res_folder, 'optim_args')
        with open(optim_args_file, 'r') as fp:
            optim_args = json.load(fp)

        # HDF5 can have issues with concurrent read/write. Use SWMR for read/write.
        # shutil.copy2(hdf_file_original, temp_path)  # Original arrangement to prevent collision. High space required.
        with h5py.File(os.path.join(self.res_folder, 'optim_state.hdf5'), 'r', libver='latest', swmr=True) as hf:
            for query in args:
                if query == QUERY['N_OBJ']:
                    return hf.attrs[query]
                elif query == QUERY['N_VAR']:
                    return len(hf.attrs[query])
                elif query == QUERY['POP_SIZE']:
                    return optim_args[query]
                elif query == QUERY['VAR_LABELS']:
                    return None
                elif query == QUERY['OBJ_LABELS']:
                    return hf.attrs[query]
                elif query == QUERY['MAX_ITER']:
                    return optim_args[query]
                elif query == QUERY['X_MIN']:
                    return hf.attrs[query]
                elif query == QUERY['X_MAX']:
                    return hf.attrs[query]
                elif query == QUERY['VARS_IGNORED']:
                    return hf.attrs[query]
                elif query == QUERY['INNOV_LATEST_GEN']:
                    return hf.attrs[query]
                elif query == 'GEN_ARR':
                    gen_arr = []
                    for key in hf.keys():
                        gen_no = int(key[3:])
                        gen_arr.append(gen_no)
                    gen_arr.sort()
                    gen_arr = np.array(gen_arr)
                    if hf.attrs['current_gen'] != gen_arr[-1]:
                        print("Mismatch in gen numbering")
                    return gen_arr


if __name__ == '__main__':
    '''Should get the following output:
    2
    129
    [0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
     0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5   0.5  ]
    [ 0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1
      0.1  0.1  0.1  0.1  0.1  0.1  0.1  0.1 29.  29.  29.  29.  29.  29.
     29.  29.  29. ]
    100
    40
    None'''
    test_query = DemoQuery(res_folder=os.path.join('data',
                                                   'nshape9_40pop_100gen_repair_power_freq10_20220930_171014',
                                                   'run1_seed184716924'))
    for key in QUERY:
        print(test_query.get(QUERY[key]))
