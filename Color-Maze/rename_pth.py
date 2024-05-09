"""
.pth files contained the dictionary entries I expected, but the keys had a prefix.
This file edits the key names, removing the prefix.
"""

import torch

path_to_file = 'results/hinf_ir_r128_envs16_nolstm_noswitch_nogoalinfo_convstride1_seed0/'

# Load the original .pth file
original_path = path_to_file + 'follower_iteration=0.pth'  # replace with your file path
data = torch.load(original_path) # Requires a GPU to deserialize.

# Create a new dictionary with renamed keys
new_data = {}
prefix_to_remove = '_orig_mod.'  # Replace with your prefix

for key, value in data.items():
    if key.startswith(prefix_to_remove):
        print(f"matched key: {key}")
        new_key = key[len(prefix_to_remove):]  # Remove the prefix # TODO uncomment once iteration works well
    else:
        print(f"non-matching key: {key}")
        new_key = key
    new_data[new_key] = value

# Save the new dictionary to a new .pth file
new_path = path_to_file + 'defix_follower_iteration=5000.pth'  # specify where to save the modified file
torch.save(new_data, new_path)

print(f'Modified .pth file saved to {new_path}')


