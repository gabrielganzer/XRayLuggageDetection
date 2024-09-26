import os
import shutil
import random

def move_random_files(source_folder, destination_folder, percentage=20):
    # Get the list of all jpg files in the source folder
    jpg_files = [file for file in os.listdir(source_folder) if file.lower().endswith('.jpg')]

    # Calculate the number of files to move (20% of total jpg files)
    num_files_to_move = int(len(jpg_files) * (percentage / 100))

    # Randomly select jpg files to move
    jpg_files_to_move = random.sample(jpg_files, num_files_to_move)

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Move selected jpg files and corresponding txt files to the destination folder
    for jpg_file_name in jpg_files_to_move:
        txt_file_name = jpg_file_name.replace('.jpg', '.txt')

        jpg_source_path = os.path.join(source_folder, jpg_file_name)
        txt_source_path = os.path.join(source_folder, txt_file_name)

        jpg_destination_path = os.path.join(destination_folder, jpg_file_name)
        txt_destination_path = os.path.join(destination_folder, txt_file_name)

        try:
            shutil.move(jpg_source_path, jpg_destination_path)
            shutil.move(txt_source_path, txt_destination_path)
        except:
            print(txt_source_path)

if __name__ == "__main__":
    # Set your source and destination folders
    source_folder_path = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/train"
    destination_folder_path = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/val"

    # Set the percentage of files to move (default is 20%)
    move_random_files(source_folder_path, destination_folder_path, percentage=10)