"""Script to produce cropped JPEGs to localize logos"""

import pandas as pd
from collections import defaultdict
from PIL import Image
from contextlib import suppress

#base_path = 'flickr_logos_27_dataset/flickr_logos_27_dataset_images/'
#save_path = 'flickr_logos_27_dataset/cropped_images/'


def crop(img_df: pd.DataFrame, base_path: str, save_path: str, ignore_fail: bool = False) -> dict:
    """
    This function converts a given pickle file to a DataFrame, and saves cropped images given a tuple of coordinates to
    localise. The columns of the Dataframe include the images and the coordinates.
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    @param ignore_fail: A bool that tells the function whether to ignore an error if due to bad coordinates.
    """

    MAX_NUMBER_OF_ITERATION = 5
    PICKLE_FILE_STARTING_COLUMN = 2
    TOTAL_NUMBER_OF_COLUMNS_TO_CROP = 4

    img_dict = defaultdict(list)
    for elem in range(1, MAX_NUMBER_OF_ITERATION):
        image_path = base_path + str(img_df.iloc[elem, 0])
        saved_location = save_path + str(img_df.iloc[elem, 0])
        coords = tuple(img_df.iloc[elem, PICKLE_FILE_STARTING_COLUMN:(PICKLE_FILE_STARTING_COLUMN +
                                                                      TOTAL_NUMBER_OF_COLUMNS_TO_CROP)])
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)

        # Now we try to save the cropped image to the saved path if the image coordintes were valid and the crop was
        # successful, other we except a SystemError from the PIL package if the coordinates are invalid.

        if ignore_fail:
            try:
                cropped_image.save(saved_location)
                img_dict['images'].append(str(img_df.iloc[elem, 0]))
                img_dict['label'].append(str(img_df.iloc[elem, 1]))
            except SystemError:
                # We expect assertion errors to be caused by bad coordinates, so we will skip those.
                pass
        else:
            cropped_image.save(saved_location)
            img_dict['images'].append(str(img_df.iloc[elem, 0]))
            img_dict['label'].append(str(img_df.iloc[elem, 1]))

    # If all successful we return a dictionary containing the image name and the classification label.
    return img_dict
