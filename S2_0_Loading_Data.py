# Imports: reading files
import xml.etree.ElementTree as ET
import os

# Imports: preparing the data
import pandas as pd
import numpy as np
import re

# Imports: image cropping
from PIL import Image

# Imports: test/train partitioning
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

#__________________________________________________________________________________

"""
Our dataset is filled with image with poentially multiple faces in each image. To load data, we need to:

1. crop faces from images using given bound boxes in data
2. treat each face as a row
3. convert these face images into arrays of rgb data

The load_data() function is going to build the main dataframe we get from completeing steps 1, 2, and 3.

----PARAMETERS----

Optional:
- test_percent: size of the test set as a decimal
- color_setting: the format of the color arrays 
- image_size: the size of the image arrays
- get_main_df


----RETURNS----
- A tuple: (train_x, test_x, train_y, test_y)
"""
def load_data(test_percent=0.2, color_setting="RGB", image_size=(50, 50), get_main_df=False):
    # Standard Directory where annotations are
    directory = os.fsencode("./annotations")

    # Build list of files and their pathways ex. ("./annotations/maksssksksss0.xml")
    fileNamesList = []
    for file in os.listdir(directory):
        filename = os.fsencode(file)
        if (re.search(".gitignore", str(filename))) == None:
            fileNamesList.append("./annotations/" + str(filename).strip()[2:-1])

    # objects will be a list of faces with everything we need to know about them.
    # we have which image they are contained in, the boundbox, the label, etc.
    objects = []

    index = 0
    for fileName in fileNamesList:
        tree = ET.parse(fileName)
        root = tree.getroot()

        for child in root:
            # find faces
            # for each face:
            if child.tag == "object":
                
                # store information about the face's original image
                childDict = {}
                childDict["index"] = index
                childDict["source_image_path"] = "./images/" + root.find("filename").text
                childDict["image_width"] = root.find("size").find("width").text
                childDict["image_height"] = root.find("size").find("height").text
                index += 1

                # add bound boxes for each face
                for attribute in child:
                    if (attribute.tag == "bndbox"):
                        childDict["bndbox"] = {}
                        for coordinates in child.find("bndbox"):
                            childDict["bndbox"][coordinates.tag] = child.find("bndbox").find(coordinates.tag).text

                    else:
                        childDict[attribute.tag] = child.find(attribute.tag).text

                # add record to list of objects
                objects.append(childDict)


    # structure of the dataframe we will build. Each row is a face.
    main_df = pd.DataFrame(columns = ["label", "xmin", "ymin", "xmax", "ymax", "height", "width", "path"])

    for records in objects:
        main_df = pd.concat(
            [
                main_df, 
                pd.DataFrame({
                    "label": [records["name"]],
                    "xmin": [int(records["bndbox"]["xmin"])],
                    "ymin": [int(records["bndbox"]["ymin"])],
                    "xmax": [int(records["bndbox"]["xmax"])],
                    "ymax": [int(records["bndbox"]["ymax"])],
                    "height": [int(records["image_height"])],
                    "width": [int(records["image_width"])],
                    "path": [records["source_image_path"]]
                })
            ],
            axis = 0
        )

    # pd.concat ruins the index of the DF, this line fixes the index numbers
    main_df.reset_index(inplace = True, drop = True)

    if (get_main_df):
        return main_df

    train, test = train_test_split(main_df, test_size = test_percent)

    train_y = np.array(pd.get_dummies(train["label"]).values)
    test_y = np.array(pd.get_dummies(test["label"]).values)

    train_x = extract_img_array_from_df(train, color_setting, image_size)
    test_x = extract_img_array_from_df(test, color_setting, image_size)

    test_x = np.array(test_x).astype(int)
    train_x = np.array(train_x).astype(int)


    return (train_x, test_x, train_y, test_y)


def extract_img_array_from_df(df, color_setting, image_size):
    img_array = []

    for index, row in df.iterrows():
        im = Image.open(row["path"])

        im_processed = im.crop(
            (
                # left, bottom, right, top
                row["xmin"],
                row["ymin"],
                row["xmax"],
                row["ymax"]
            )
        ).convert(
            # we have to convert because we have some RGB and some RGBA images
            color_setting
        ).resize(
            # this choice is important
            # based on Tao's finding that the vast majority of images are less than 50x50
            # but there are some images that are much higher resolution
            image_size
        )

        img_array.append(img_to_array(im_processed))

    return img_array


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = load_data()
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)

    main_df = load_data(get_main_df=True)
    

