import os

# Function to extract filename from a given path
def get_filename(path):
    return os.path.basename(path)

# Function to get the first subdirectory number (for LibriTTS structure)
def get_first_subdirectory_number(path):
    return path.split('/')[-3]

# Function to read the contents of the lab file corresponding to the wav file
def read_lab_file(wav_filename):
    parts = wav_filename.split('_')
    lab_filename = wav_filename.replace('.wav', '.normalized.txt')
    lab_file_path = os.path.join(directory, parts[0], parts[1], lab_filename)
    try:
        with open(lab_file_path, 'r') as lab_file:
            return lab_file.read().strip()
    except FileNotFoundError:
        return "(lab file not found)"

# Process the input file
input_file = '/home/jeonyj0612/SG-PAC2/config/filelists/7-val-Updated-removed100subset.txt.cleaned.txt'  # Path to your input file
output_file = '/home/jeonyj0612/SG-PAC2/config/filelists/V0-val.txt'  # Path to your output file
directory = '/home/jeonyj0612/SG-PAC/LibriTTS/train-clean-460'
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Split the line by '|'
        parts = line.strip().split('|')
        
        # Extract necessary parts
        third_file = parts[0]  # Note that third file needs to be third in the actual case
        
        third_filename = get_filename(third_file)
        
        # Extract first subdirectory number
        third_subdir = get_first_subdirectory_number(third_file)
        
        # Get lab file contents for the third element
        lab_contents = read_lab_file(third_filename)
        
        output_line = f"{third_filename}|{third_subdir}|{lab_contents}|" + "|".join(parts[-4:]) + "\n"

        # Construct output line: Stage2
        # output_line = f"{first_filename}|{first_subdir}|{third_filename}|{third_subdir}|{lab_contents}|" + "|".join(parts[-4:]) + "\n"
        
        # Write to output file
        outfile.write(output_line)

print("Processing complete.")
