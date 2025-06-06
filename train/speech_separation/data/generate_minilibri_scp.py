import os
import glob

def generate_scp_file(data_dir, output_file):
    print(f"Looking for wav files in: {data_dir}")
    # Get all wav files in mix_both directory
    mix_files = sorted(glob.glob(os.path.join(data_dir, 'mix_both', '*.wav')))
    print(f"Found {len(mix_files)} wav files")
    
    with open(output_file, 'w') as f:
        for mix_file in mix_files:
            # Get the filename without path
            filename = os.path.basename(mix_file)
            
            # Construct paths for s1 and s2
            s1_file = os.path.join(data_dir, 's1', filename)
            s2_file = os.path.join(data_dir, 's2', filename)
            
            # Convert to absolute paths
            mix_file = os.path.abspath(mix_file)
            s1_file = os.path.abspath(s1_file)
            s2_file = os.path.abspath(s2_file)
            
            # Write the line in the same format as WSJ0-2Mix
            f.write(f'{mix_file} {s1_file} {s2_file}\n')

# Get the absolute path to the workspace root
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
print(f"Workspace root: {workspace_root}")

# Generate scp files for train and validation sets
train_dir = os.path.join(workspace_root, 'MiniLibriMix', 'train')
val_dir = os.path.join(workspace_root, 'MiniLibriMix', 'val')

print(f"Train directory: {train_dir}")
print(f"Validation directory: {val_dir}")

generate_scp_file(train_dir, 'tr_minilibri_2mix_16k_min.scp')
generate_scp_file(val_dir, 'cv_minilibri_2mix_16k_min.scp') 