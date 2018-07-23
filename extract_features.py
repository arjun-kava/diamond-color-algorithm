from color_descriptor import *

from os import listdir
from os.path import isfile, join


# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

indexFile = "color_algorithm.csv"

# open the output index file for writing
output = open(indexFile, "w")

# fetch training data
training_path = 'data/train'
only_files = [f for f in listdir(training_path) if isfile(join(training_path, f))]

for image in only_files:

    image_path = join(training_path, image)

    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    imageID = image_path[image_path.rfind("/") + 1:]
    image = cv2.imread(image_path)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # describe the image
    features = cd.describe(hsv)

    # write the features to file
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageID, ",".join(features)))

output.close()