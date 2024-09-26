import glob, os

# Current directory
current_dir = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/train"
# Create and/or truncate training.txt
file_train = open('/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/train.txt', 'w')

# Populate training.txt
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    file_train.write("/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/train" + "/" + title + '.jpg' + "\n")

current_dir = "/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/val"

# Create and/or truncate train.txt and test.txt
file_validation = open('/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/val.txt', 'w')

# Populate train.txt and test.txt

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    file_validation.write("/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/val" + "/" + title + '.jpg' + "\n")
