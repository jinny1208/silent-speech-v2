from g2p_en import G2p
import string
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# Initialize G2p
g2p = G2p()

# Function to clean the ARPAbet output by removing special characters
def clean_and_wrap(text):
    # Remove punctuation using str.translate and str.maketrans
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Add curly brackets around the cleaned text
    wrapped_text = f"{{{cleaned_text}}}"
    
    return wrapped_text
# Load FileA and FileB
fileA_path = '/home/jeonyj0612/SG-PAC2/preprocessed_data/LibriTTS/all.txt'
fileB_path = '/home/jeonyj0612/SG-PAC2/config/filelists/V0-val.txt'
output_path = '/home/jeonyj0612/SG-PAC2/config/filelists/V1-val.txt'

# Read FileA into a dictionary for easy lookup
fileA_dict = {}
with open(fileA_path, 'r') as fileA:
    for line in fileA:
        elements = line.strip().split('|')
        fileA_dict[elements[0]] = elements[2]  # Store ARPAbet (fourth element)

# Process FileB and write the output
with open(fileB_path, 'r') as fileB, open(output_path, 'w') as output_file:
    for line in fileB:
        elements = line.strip().split('|')
        fileB_id = elements[0].split('.')[0]  # Get the first element without '.wav'
        sentence = elements[2].strip("'")  # Remove leading and trailing quotes from sentence
        # print(sentence)

        arpabet_list = g2p(sentence)
        arpabet_list = ' '.join(arpabet_list)
        # print(arpabet_list)
        arpabet_list = clean_and_wrap(arpabet_list)

        # Write the ARPAbet and original line from FileB
        new_line = f"{elements[0]}|{elements[1]}|{arpabet_list}|{elements[2]}|{'|'.join(elements[3:])}\n"
        output_file.write(new_line)

print("Process completed. Output saved to:", output_path)
