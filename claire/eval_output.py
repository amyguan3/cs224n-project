import numpy as np

# Function to read and split the text in the file by a specific delimiter
def split_text_by_delimiter(file_path, delimiter='[SEP_DIAL]'):
    scores = []
    file1 = open(file_path, 'r')
    lines = file1.readlines()
    for line in lines:
        line = line.strip()
        score = float(line.split(delimiter)[2])
        scores.append(score)
    dial_score = np.mean(scores)
    return dial_score
    
 
count = 0
# Strips the newline character

# Use the function and print the result
dials = [
    "aus",
    "gbr",
    "ind_n",
    "ind_s",
    "irl",
    "kenya",
    "nga",
    "nzl",
    "phl",
    "usa",
    "zaf",
]

for dial in dials:
    split_content = split_text_by_delimiter(dial + "_outs.txt")
    print(dial, split_content)