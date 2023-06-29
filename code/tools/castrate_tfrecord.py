import os
import random
import shutil

def move_files(source_dir, destination_dir):
    # Get the list of files in the source directory
    files = os.listdir(source_dir)
    
    # Calculate the number of files to move (approximately one-third)
    num_files_to_move = len(files) // 3
    
    # Select random files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Move the selected files to the destination directory
    for file_name in files_to_move:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copy(source_path, destination_path)
        print(f"Moved: {file_name}")
    
    print("Files moved successfully.")

# Example usage
source_directory = "/trainingData/sage/CIC-IDS2018/tfrecord/valid"
destination_directory = "/trainingData/sage/CIC-IDS2018/castration/valid"

move_files(source_directory, destination_directory)
