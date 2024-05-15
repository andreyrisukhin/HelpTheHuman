"""
Inspect the contents of a data file (hdff)
"""

import h5py

path = "hinf_ir_r128_envs16_nolstm_noswitch_nogoalinfo_convstride1_seed0_5000_follower_testing4.hdf5"
with h5py.File(path, 'r') as f:
    print(f"file: {f}")
    print(f"file.keys(): {f.keys()}")

