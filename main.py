import os
import argparse
from ultralytics import YOLO
import numpy as np
model = YOLO('./best_A2024_noAC_int8_openvino_model/')
CONF_THR = 0.86

def process(res):
    conf = res[0].boxes.conf.numpy()
    bbox = res[0].boxes.xywhn
    if len(conf) > 0:

        if conf.max() > CONF_THR :
            return bbox[np.argmax(conf)].numpy()
        else: return []
    else:
        return []


def inference(root):
    result = {}
    for filename in os.listdir(root):
        if filename[-3:] != 'txt' and ('.ipynb_checkpoints' not in filename ) :
            res = model.predict(root + filename)
            bbox = process(res)
            print(f'bbox = {bbox}')
            if len(bbox) != 0:
                x, y = bbox[:2]
                result[filename] = [x, y]
    return result

def z_to_w(z):
    w = z * 18 / 16
    return w

def z_to_h(z):
    h = z * 12 / 16
    return h

def calculate_coordinates(x_gps, y_gps, z_gps, x_obj, y_obj):
    w_real = z_to_w(z_gps)
    h_real = z_to_h(z_gps)
    x_obj_real = x_gps - w_real * (x_obj - 0.5)
    y_obj_real = y_gps + h_real * (y_obj - 0.5)
    return x_obj_real, y_obj_real

# Firstly parse coordinates, then NN output
def parse_coordinates(path):
    global img_info_list
    with open(path, 'r') as coordinates_file:
        coordinates = coordinates_file.readlines()
    for img_info_line in coordinates:
        name, x_gps, y_gps, z_gps = img_info_line.split(' ')
        x_gps, y_gps, z_gps = list(map(float, [x_gps, y_gps, z_gps]))
        img_info = {
            'name': name,
            'gps': [x_gps, y_gps, z_gps],
        }
        img_info_list += [img_info]

# Firstly parse coordinates, then NN output
def parse_nn_output(nn_output):
    global img_info_list
    for info_id in range(len(img_info_list)):
      if img_info_list[info_id]["name"] in nn_output:
        x_obj, y_obj = nn_output[f'{img_info_list[info_id]["name"]}']
        x_gps, y_gps, z_gps = img_info_list[info_id]['gps']
        x_obj_real, y_obj_real = calculate_coordinates(x_gps, y_gps, z_gps, x_obj, y_obj)
        img_info_list[info_id]['obj'] = [x_obj_real, y_obj_real]
      else:
        img_info_list[info_id]['obj'] = None

def show_results():
    global img_info_list
    for img_info in img_info_list:

        try:
            print(f'Name:\t{img_info["name"]}')
        except:
            print(f'Name:\tNone')
        try:
            print(f'GPS:\t{img_info["gps"]}')
        except:
            print(f'GPS:\tNone')
        try:
            print(f'Object:\t{img_info["obj"]}')
        except:
            print(f'Object:\tNone')


img_info_list = []

def run(root):
    nn_output = inference(root)
    parse_coordinates(root + 'coordinates.txt')
    parse_nn_output(nn_output)


    list_results = []

    for dict_info in img_info_list:
        if dict_info['obj'] is not None:
            list_results.extend([dict_info['obj']])

    x, y = [], []
    for [x_step, y_step] in list_results:
        x.append(x_step)
        y.append(y_step)

    q_x = np.percentile(np.array(x), 50)
    q_y = np.percentile(np.array(y), 50)

    clear_results = []
    for [x_step, y_step] in list_results:
        if q_x * 0.25 < x_step < q_x* 1.75 and \
           q_y * 0.25 < y_step < q_y * 1.75:
            clear_results.append([x_step, y_step])


    print(list_results)
    print(f'Result:\n\n\n___________\n\n{np.mean(list_results, axis=0)}')
    print(f'Cleat result:\n\n\n___________\n\n{np.mean(clear_results, axis=0)}')

    # show_results()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_images', default = './IMAGES/')
    args = parser.parse_args()
    run(args.path_images)
