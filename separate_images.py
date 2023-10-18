import os
import shutil

def read_file(file_name):
    with open(file_name, 'r') as f:
        # Fetch only the image names
        return [line.strip().split()[0] for line in f.readlines()]

def move_files(file_list, source_folder, destination_folder):
    for file in file_list:
        src_path = os.path.join(source_folder, file)
        dest_path = os.path.join(destination_folder, file)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"File {file} not found in {source_folder}")

def main():
    # Place this script in the same directory as the images folder
    source_dir = "images" 

    train_dir = "training_images"
    test_dir = "testing_images"

    # Make the directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read the train and test image lists
    train_images = read_file("Xray14_val_official.txt")
    test_images = read_file("test_list.txt")

    # Move the train and test images to their respective directories
    move_files(train_images, source_dir, train_dir)
    move_files(test_images, source_dir, test_dir)

if __name__ == "__main__":
    main()
