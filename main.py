from searcher import *
from color_descriptor import *

from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))

    # fetch training data
    training_path = 'data/test'
    only_files = [f for f in listdir(training_path) if isfile(join(training_path, f))]

    for image in only_files:

        # extract full image path
        image_path = join(training_path, image)
        print("image_path", image_path)

        # load the query image and describe it
        query = cv2.imread(image_path)
        features = cd.describe(query)

        # perform the search
        indexFile = "color_algorithm.csv"
        searcher = Searcher(indexFile)
        results = searcher.search(features)

        # display the query
        cv2.imshow("Query", query)

        topK = results[0]
        score = topK[0]
        resultID = topK[1]
        print("score", score)
        print("resultID", resultID)