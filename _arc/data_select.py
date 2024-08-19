import json
import os
import pickle
import random
from collections import defaultdict

# Directory containing the ARC files
arc_dir = "dataset/ARC-800-tasks/training"
arc_files = []

# Get the list of files in the directory
for filename in os.listdir(arc_dir):
    if os.path.isfile(os.path.join(arc_dir, filename)):
        arc_files.append(filename)

# List to store eligible arc_data entries
eligible_entries = []

# Process each file
for arc_file in arc_files:
    with open(os.path.join(arc_dir, arc_file), 'r') as file:
        arc_data = json.load(file)

    # Check if 'train' data and 'input' exist
    if 'train' in arc_data and len(arc_data['train']) > 0 and 'input' in arc_data['train'][0]:
        xlength = len(arc_data['train'][0]['input'])
        ylength = len(arc_data['train'][0]['output'])
        if xlength <= 5 and ylength <= 5:
            eligible_entries.append(arc_data)

# Randomly sample 100 entries
sample_size = min(100, len(eligible_entries))
print(f"Sampling {sample_size} entries from {len(eligible_entries)} eligible entries")
sampled_entries = random.sample(eligible_entries, sample_size)

# Split the sampled entries into validation and test sets
half_size = 20
validation_entries = sampled_entries[:20]
test_entries = sampled_entries[20:]

# Save validation entries to a pickle file
with open('sampled_arc_val_data.pkl', 'wb') as val_file:
    pickle.dump(validation_entries, val_file)

# Save test entries to a pickle file
with open('sampled_arc_test_data.pkl', 'wb') as test_file:
    pickle.dump(test_entries, test_file)

# Calculate and print length statistics for the validation set
val_length_counts = defaultdict(int)
for entry in validation_entries:
    length = len(entry['train'][0]['input'])
    val_length_counts[length] += 1

# Output length stats for the validation set
print("Validation Set Length Stats:")
for length, count in sorted(val_length_counts.items()):
    print(f"Length: {length}, Count: {count}")

# Calculate and print length statistics for the test set
test_length_counts = defaultdict(int)
for entry in test_entries:
    length = len(entry['train'][0]['input'])
    test_length_counts[length] += 1

# Output length stats for the test set
print("Test Set Length Stats:")
for length, count in sorted(test_length_counts.items()):
    print(f"Length: {length}, Count: {count}")
