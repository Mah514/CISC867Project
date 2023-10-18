import csv

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]

def get_binary_encoding(labels):
    encoding = [0] * len(DISEASES)
    for label in labels:
        if label in DISEASES:
            encoding[DISEASES.index(label)] = 1
    return encoding

def read_csv(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return {row[0]: row[1].split('|') for row in reader}

def read_file(file_name):
    with open(file_name, 'r') as f:
        return [line.strip() for line in f.readlines()]

def write_file(file_name, data):
    with open(file_name, 'w') as f:
        for line in data:
            f.write(line + '\n')

def main():
    csv_data = read_csv('Data_Entry_2017_v2020.csv')
    test_images = read_file('test_list.txt')
    
    formatted_test_data = []
    for image in test_images:
        if image in csv_data:
            diseases = csv_data[image]
            encoding = get_binary_encoding(diseases)
            formatted_line = image + ' ' + ' '.join(map(str, encoding))
            formatted_test_data.append(formatted_line)

    write_file('formatted_test_list.txt', formatted_test_data)

if __name__ == "__main__":
    main()
