import glob  # to read files
import os
import xml.etree.ElementTree as ET  # to read the xml data from annotations files of images

import cv2
import numpy as np
from tqdm import tqdm  # to check the progress

with open("/home/gabriel.ganzer@kryptus.lan/Documentos/final-project/dataset/labels.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# read annotation data from xml
def read_xml_data(xml_file):
    file = open(xml_file, "r")
    tree = ET.parse(file)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    objects = []

    for obj in root.iter("object"):
        cls = obj.find("name").text
        bbox = [int(round(float(x.text))) for x in obj.find("bndbox")]
        objects.append([cls, bbox])

    return width, height, objects


# Convert XML data to YOLO format
def xml_to_yolo(bbox, w, h):
    """
    Yolo coordinates are calculated by normalizing the Bounding Box coordinates.
    It is important to know the Width and Height of the image whose normalized coordinates are to be calculated
    for YOLO format.
    bbox = Only xmin, ymin, xmax, ymax are passed in the list. These are the actual values read from the xml file
    """
    x_c = ((bbox[2] + bbox[0]) / 2) / w
    y_c = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h

    return [x_c, y_c, width, height]


def cv_to_yolo(main_bbox, ht, wd):
    """
    This function works similar to the above xml_to_yolo(). The only point of difference is the bbox size.
    main_bbox = The BB coords along with the index is passed. This is the output of the rotated images.
    """
    bb_width = main_bbox[3] - main_bbox[1]
    bb_height = main_bbox[4] - main_bbox[2]
    bb_cx = (main_bbox[1] + main_bbox[3]) / 2
    bb_cy = (main_bbox[2] + main_bbox[4]) / 2

    return (
        main_bbox[0],
        round(bb_cx / wd, 6),
        round(bb_cy / ht, 6),
        round(bb_width / wd, 6),
        round(bb_height / ht, 6),
    )


class SimpleAugmentation:
    """
    This class works for images where augmentation of image doesnot affect the bounding box coordinates of the object
    """

    def __init__(self, filename, image_ext):
        """
        Initializing the Image
        """
        self.filename = filename
        self.image_ext = image_ext

        # read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

    def simpleBB(self):
        """
        This function helps in saving the augmented images and generate its corresponding YOLO text data.
        Files are saved with name as 'filename_aug' and its corresponding extention.
        """
        # Read XML from data
        W, H, objects = read_xml_data(self.filename + ".xml")
        res = []

        for i, obj in enumerate(objects):
            yolo_bbox = xml_to_yolo(obj[1], W, H)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            res.append(f"{classes.index(obj[0])} {bbox_string}")

        # Save augmented image in new folder
        f = self.filename
        file = f.split("/")

        cv2.imwrite(
            os.path.join(f"{output_dir}/{file[1]}_aug{self.image_ext}"), self.image
        )
        filename = file[1] + "_aug.txt"

        # save yolo format text file in the same folder as its augmented image
        if res:
            with open(os.path.join(f"{output_dir}/{filename}"), "w", encoding="utf-8") as f:
                f.write("\n".join(res))

    def contrastImage(self):
        """
        To imporve the contrast effect of the images.
        1. Convert image to gray scale
        2. Perform Equlization Hist, to improve the contrast of the image
        """
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.equalizeHist(gray)

    def sharpenImage(self):
        """
        To sharpen the effects of the object in the image
        1. Define Kernel to sharpen (https://setosa.io/ev/image-kernels/)
        2. Using filter2D() for sharpening
        """
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        self.image = cv2.filter2D(self.image, -1, kernel)

    def blurImage(self):
        """
        To remove noise from image.
        As most of the Image data have low light intensity, Gaussain Blur is used, to remove noise without affecting
        the image

        kernel size = 15,15 is used as the image is large. This value can be increased or decreased based on the
        desired output.
        """
        self.image = cv2.GaussianBlur(self.image, (15, 15), 0)

    def modified_image(self):
        """
        This is a helper function, to call the above mentioned functionalities on an image. It can be improved by using
        config files,on deciding the scale to which it is to be implemented.

        I wanted to implement the result with less hassle.
        """
        self.contrastImage()
        self.sharpenImage()
        self.blurImage()
        return self.image


# This class performes rotation of images, as well the finds the bounding box coordinates of rotated object
# Defining a class to perform Augmentation.
# Here ony Rotating the images is considered
# Perform YOLO Rotation
class yoloRotate:
    def __init__(self, filename, image_ext, angle):
        """
        1. Checking the image paths
        2. Initializing the Images
        3. Defining the transformation Matrix
        """
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + ".xml")

        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = (
            self.angle * np.pi / 180
        )  # Converting angle(in degree) to radian

        # define a transformation matrix
        self.rot_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )

    def rotateYoloBB(self):
        """
        This function focuses on finding the new coordinates of the BB after Image Rotation.
        """
        new_ht, new_wd = self.rotate_image().shape[:2]

        # Read XML from data
        W, H, objects = read_xml_data(self.filename + ".xml")

        # Store Bounding Box coordinates from an image
        new_bbox = []

        # To store the differnt object classes, later used to generate the classes.txt file used for YOLO training.
        global classes

        # Convert XML data to YOLO format
        for i, obj in enumerate(objects):
            if obj[0] not in classes:
                print("Class not found")
                classes.append(obj[0])

            # From XML
            xmin = obj[1][0]
            ymin = obj[1][1]
            xmax = obj[1][2]
            ymax = obj[1][3]

            # Calculating coordinates of corners of the Bounding Box
            top_left = (xmin - W / 2, -H / 2 + ymin)
            top_right = (xmax - W / 2, -H / 2 + ymin)
            bottom_left = (xmin - W / 2, -H / 2 + ymax)
            bottom_right = (xmax - W / 2, -H / 2 + ymax)

            # Calculate new coordinates (after rotation)
            new_top_left = []
            new_bottom_right = [-1, -1]

            for j in (top_left, top_right, bottom_left, bottom_right):
                # Generate new corner coords by multiplying it with the transformation matrix (2,2)
                new_coords = np.matmul(self.rot_matrix, np.array((j[0], -j[1])))

                x_prime, y_prime = (
                    new_wd / 2 + new_coords[0],
                    new_ht / 2 - new_coords[1],
                )

                # Finding the new top-left coords, by finding the minimum of calculated x and y-values
                if len(new_top_left) > 0:
                    if new_top_left[0] > x_prime:
                        new_top_left[0] = x_prime
                    if new_top_left[1] > y_prime:
                        new_top_left[1] = y_prime
                else:  # for first iteration, lists are empty, therefore directly append
                    new_top_left.append(x_prime)
                    new_top_left.append(y_prime)

                # Finding the new bottom-right coords, by finding the maximum of calculated x and y-values
                if new_bottom_right[0] < x_prime:
                    new_bottom_right[0] = x_prime
                if new_bottom_right[1] < y_prime:
                    new_bottom_right[1] = y_prime

            # i(th) index of the object
            new_bbox.append(
                [
                    classes.index(obj[0]),
                    new_top_left[0],
                    new_top_left[1],
                    new_bottom_right[0],
                    new_bottom_right[1],
                ]
            )

        return new_bbox

    def rotate_image(self):
        """
        This function focuses on rotating the image to a particular angle.
        """
        height, width = self.image.shape[:2]
        img_c = (width / 2, height / 2)  # Image Center Coordinates

        rotation_matrix = cv2.getRotationMatrix2D(
            img_c, self.angle, 1.0
        )  # Rotating Image along the actual center

        abs_cos = abs(rotation_matrix[0, 0])  # Cos(angle)
        abs_sin = abs(rotation_matrix[0, 1])  # sin(angle)

        # New Width and Height of Image after rotation
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract the old image center and add the new center coordinates
        rotation_matrix[0, 2] += bound_w / 2 - img_c[0]
        rotation_matrix[1, 2] += bound_h / 2 - img_c[1]

        # rotating image with transformed matrix and new center coordinates
        rotated_matrix = cv2.warpAffine(self.image, rotation_matrix, (bound_w, bound_h))

        return rotated_matrix


if __name__ == "__main__":
    input_dir = "train"
    output_dir = "XML"

    files = glob.glob(os.path.join(input_dir, "*.jpg"))
    for filename in tqdm(files):
        file = filename.split(".")

        image_name = file[0]
        image_ext = "." + file[1]

        # Simple Data Augmentation (blur,sharpen,contrast)
        aug = SimpleAugmentation(image_name, image_ext)
        modified_img = aug.modified_image()
        aug.simpleBB()

        # Rotated Image Augmentation
        angles = [45, 90, 180, 225, 270]
        for angle in angles:
            im = yoloRotate(image_name, image_ext, angle)
            bbox = im.rotateYoloBB()  # new BBox values, after rotation
            rotated_image = im.rotate_image()  # rotated_image

            # writing into new folder
            f = image_name
            file = f.split("/")

            cv2.imwrite(
                os.path.join(f"{output_dir}/{file[1]}_{str(angle)}{image_ext}"),
                rotated_image,
            )
            file_name = file[1] + "_" + str(angle) + ".txt"
            # saving the Bbox ccoordinates in YOLO format in same folder as its augmented image data
            for i in bbox:
                with open(os.path.join(output_dir, file_name), "a") as fout:
                    fout.writelines(
                        " ".join(
                            map(
                                str,
                                cv_to_yolo(
                                    i,
                                    rotated_image.shape[0],
                                    rotated_image.shape[1],
                                ),
                            )
                        )
                        + "\n"
                    )
