import os
import shutil
from tqdm import tqdm

def move_unmatched_files(source_folder, destination_folder):
    for file_name in tqdm(os.listdir(source_folder)):
        if file_name.lower().endswith('.jpg'):
            txt_file = os.path.join(source_folder, file_name[:-4] + '.txt')

            if not os.path.exists(txt_file):
                image_file = os.path.join(source_folder, file_name)
                destination_image_file = os.path.join(destination_folder, file_name)
                shutil.move(image_file, destination_image_file)
                print(f"Moved: {file_name} to {destination_folder}")

        elif file_name.lower().endswith('.txt'):
            image_file = os.path.join(source_folder, file_name[:-4] + '.jpg')

            if not os.path.exists(image_file):
                txt_file = os.path.join(source_folder, file_name)
                destination_txt_file = os.path.join(destination_folder, file_name)
                shutil.move(txt_file, destination_txt_file)
                print(f"Moved: {file_name} to {destination_folder}")

if __name__ == "__main__":
    # Set the path to your source folder containing .jpg and .txt files
    source_folder_path = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/XML"

    # Set the path to your destination folder
    destination_folder_path = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/trash"

    move_unmatched_files(source_folder_path, destination_folder_path)