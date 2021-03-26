import pandas as pd
import os
from PIL import Image
from ImageAI.imageai.Detection import ObjectDetection
from collections import defaultdict
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

df1 = pd.read_pickle('streetstyle_df.pkl')
coord_dict = defaultdict(list)
base_path = "cropped_streetstyle5k/"
curr_path = "streetstyle27k/"


def crop_streetstyle(image_path, img_coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(img_coords)
    cropped_image.save(saved_location)


def opt_person_coord(element):
    prob_0 = 0
    opt_coords = None
    if element[1][0]['name'] == 'person':
        coords = element[1][0]['box_points']
        prob_0 = element[1][0]['percentage_probability']
        opt_coords = tuple(coords)

    for x in element[1]:
        if (x['name'] == 'person') and (x['percentage_probability'] > prob_0):
            prob_0 = x['percentage_probability']
            coords = x['box_points']
            opt_coords = tuple(coords)
    return opt_coords


def find_name(x):
    t1 = x.split('/')[-1]
    t1 = base_path + t1
    return t1

df1['img_names'] = df1['img_paths']
df1['img_names'] = df1['img_names'].apply(find_name)


for val in range(3500,8500):
    curr_img_path = curr_path + str(df1.iloc[val,23])

    detected = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,
                                                                           curr_img_path),
                                                  extract_detected_objects=False, output_type='array')
    try:
        person_coord = opt_person_coord(detected)
        if person_coord:
            #crop_streetstyle(curr_img_path,person_coord,crop_img_path)
            coord_dict['img_paths'].append(df1.iloc[val,23])
            coord_dict['coords'].append(person_coord)
        else:
            pass
    except:
        pass

coord_df = pd.DataFrame.from_dict(coord_dict)
print(coord_df.head())
