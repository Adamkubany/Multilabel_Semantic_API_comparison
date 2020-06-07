
# from keras.models import Model
import numpy as np
import os
import pandas as pd

from config import imagenet_classes as img_clsses
from config import DATASET_PATHS, SAVE_FILE_FLAG, PRETRAINED_MODEL_PATH, api_numbers

global DATASET, DATA_PATH, IMAGES_PATH, IMAGES_DATA_FILE


def inference_deepdetect():
    """
    API explanation link : https://www.deepdetect.com/quickstart-server/?os=windows&source=docker&compute=cpu&gpu=gtx&backend=caffe%2Ctsne%2Cxgboost&deepdetect=server
    CURL assisting link : https://curl.trillworks.com/

     :return:
     df : dataframe with image objects
     df_unavailable : dataframe with images which were not inferred
     """
    import requests
    # check docker server alive
    # response = requests.get('http://localhost:8080/info')
    # set a service
    # data = '{"mllib":"caffe","description":"image classification service","type":"supervised","parameters":{"input":{"connector":"image"},"mllib":{"nclasses":1000}},"model":{"repository":"/opt/models/ggnet/"}}'
    # response = requests.put('http://localhost:8080/services/imageserv', data=data).json()
    # print(response)
    api = 'deepdetect'
    api_num = api_numbers[api]

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id', 'note'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} from {}. {} / {}'.format(img.img_id, api, count, images))
        image_url = img.url
        try:
            data = '{"service":"imageserv", "parameters":{"output":{"best":5}, "mllib":{"gpu":true}}, "data":["' + image_url + '"]}'
            res = requests.post('http://localhost:8080/predict', data=data).json()
        except:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id}, ignore_index=True)
            continue
        if res['status']['code'] == 400:
            print('img "{}" is unavailable, server ERROR 400'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': res['status']['msg']}, ignore_index=True)
            continue

        for label in res['body']['predictions'][0]['classes']:
            df = df.append({'img_id': img.img_id, 'label': label['cat'][label['cat'].find(' ') + 1:], 'conf_level': label['prob'], 'run_date': 2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_imagenet(algo_choice=1, top=5, print_every=10):
    """
    API explanation link : https://keras.io/applications/
    :param algo_choice: numerical choice of algos 0-4 for ['InceptionResNetV2', 'mobilenet_v2', 'vgg19', 'resnet_imgnet', 'Inception-v3']
    :param top: top labels, default 5
    :param print_every: print every # images
    :return:
    """

    from keras.preprocessing import image

    algos = ['InceptionResNetV2', 'mobilenet_v2', 'vgg19', 'resnet_imgnet', 'Inception-v3']
    algo = algos[algo_choice]
    api_num = api_numbers[algo]

    print('algo {} top {}'.format(algo, top))

    # algo framework
    if algo == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        from keras.applications.inception_resnet_v2 import preprocess_input
        base_model = InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    elif algo == 'mobilenet_v2':
        from keras.applications.mobilenet_v2 import MobileNetV2
        from keras.applications.mobilenet_v2 import preprocess_input
        base_model = MobileNetV2(input_shape=None, alpha=1.0, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    elif algo == 'vgg19':
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        base_model = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    elif algo == 'resnet_imgnet':
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
    elif algo == 'Inception-v3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        base_model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    image_list = [f for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    img_count = 0
    for img_file_name in image_list:
        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {} for algo {} top {}'.format(img_file_name, img_count, len(image_list), algo, top))
        cur_img_path = os.path.join(IMAGES_PATH, img_file_name)
        if algo in ['InceptionResNetV2', 'Inception-v3']:
            img = image.load_img(cur_img_path, target_size=(299, 299))  # for InceptionResNetV2 and Inception-v3
        elif algo in ['mobilenet_v2', 'vgg19', 'resnet_imgnet']:
            img = image.load_img(cur_img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_labels_coef = base_model.predict(x)[0]
        top_labels_indx = img_labels_coef.argsort()[-top:][::-1]
        for indx in top_labels_indx:
            label_dict = {'img_id': img_file_name[:img_file_name.find('.')], 'label': img_clsses[indx], 'conf_level': img_labels_coef[indx], 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)
    if algo_choice == 4:  # due to legacy naming issues
        algo = 'tensorflow'
    df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, algo)), index=None)
    # df.to_csv(os.path.join(DATA_PATH, algo + '_obj_inference_data.csv'), index=None)
    return print('Done!')


def inference_coco(algo_choice=1, print_every=10):
    """
    API explanation link : https://imageai.readthedocs.io/en/latest/
    ResNet model download link : https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
    YOLO model download link : https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
    :param algo_choice: numerical choice of algos 0-1 for ['resnet_coco', 'yolo_v3_coco']
    :param print_every:
    :param print_every: print every # images
    """
    from imageai.Detection import ObjectDetection

    algos = ['resnet_coco', 'yolo_v3_coco']
    algo = algos[algo_choice]
    api_num = api_numbers[algo]

    print('algo {}'.format(algo))
    # model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'Sources', 'pretrained_models')
    model_name = ["resnet50_coco_best_v2.0.1.h5", "yolo.h5"]
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])

    detector = ObjectDetection()

    if algo == 'resnet_coco':
        detector.setModelTypeAsRetinaNet()

    elif algo == 'yolo_v3_coco':
        detector.setModelTypeAsYOLOv3()

    detector.setModelPath(os.path.join(PRETRAINED_MODEL_PATH, model_name[algo_choice]))
    detector.loadModel()
    image_list = [f for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
    img_count = 0
    for img_file_name in image_list:
        img_name = img_file_name[:img_file_name.find('.')]
        # cur_img_path = img_path + '/' + img_file_name
        cur_img_path = os.path.join(IMAGES_PATH, img_file_name)

        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {}, algo {}'.format(img_file_name, img_count, len(image_list), algo))
        detections = detector.detectObjectsFromImage(input_image=cur_img_path, output_image_path='results/temp_image.jpg', minimum_percentage_probability=10)
        for label in detections:
            label_dict = {'img_id': img_name, 'label': label['name'], 'conf_level': label['percentage_probability'], 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)

    df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, algo)), index=None)
    # df.to_csv(os.path.join(DATA_PATH, algo + '_obj_inference_data.csv'), index=None)
    print('Done!')


def yolo_imagenet(print_every=10):
    """
    install darknet : https://pjreddie.com/darknet/install/
    API explanation link : https://pjreddie.com/darknet/imagenet/#darknet53_448

    :param print_every: print every # images
    :return:
    """
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    algo = 'yolo_v3'
    api_num = api_numbers[algo]
    image_list = [f for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
    img_count = 0
    for img_file_name in image_list:
        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {}'.format(img_file_name, img_count, len(image_list)))
        img_name = img_file_name[:img_file_name.find('.')]
        cur_img_path = os.path.join(IMAGES_PATH, img_file_name)
        img_data = os.popen("./darknet classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53.weights {}".format(cur_img_path)).read()
        img_data_ = img_data.split('\n')[:-1]
        for label_data in img_data_:
            label = label_data[label_data.find(':') + 2:]
            conf = label_data[:label_data.find('%')]
            label_dict = {'img_id': img_name, 'label': label, 'conf_level': conf, 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)

    df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, algo)), index=None)
    # df.to_csv(os.path.join(DATA_PATH, algo + '_obj_inference_data.csv'), index=None)
    print('Done!')


def main():
    global DATASET, DATA_PATH, IMAGES_PATH, IMAGES_DATA_FILE, ORIGINAL_METADATA_PATH
    correct_db = False
    db_dict =  {'0': 'OPEN_IMAGE',
                '1': 'VISUAL_GENOME'}
    while not correct_db:
        print('Please choose dataset:')
        for db in db_dict.keys():
            print('For {} dataset, choose {}'.format(db_dict[db], db))
        db_choice = input("Your choice: ")
        if db_choice in db_dict.keys():
            correct_db = True
            DATASET = db_dict[db_choice]
            DATA_PATH = DATASET_PATHS[DATASET]['DATA_PATH']
            IMAGES_PATH = DATASET_PATHS[DATASET]['IMAGES_PATH']
            IMAGES_DATA_FILE = DATASET_PATHS[DATASET]['IMAGES_DATA_FILE']
        else:
            print('ERROR: "{}" is a wrong input, please try again...'.format(db_choice))
            print(50 * '-')
    correct_input = False
    choice_dict = {"1": ['InceptionResNetV2', 'ImageNet'],
                   "2": ['mobilenet_v2', 'ImageNet'],
                   "3": ['VGG19', 'ImageNet'],
                   "4": ['ResNet50', 'ImageNet'],
                   "5": ['InceptionV3', 'ImageNet'],
                   "6": ['YOLO V3 (require linux)', 'ImageNet'],
                   "7": ['ResNet50', 'COCO'],
                   "8": ['YOLO V3', 'COCO'],
                   }
    while not correct_input:
        print("Open-source API inference for the '{}' dataset. Choose from the following options:".format(DATASET))
        for c in choice_dict.keys():
            print("For '{}' model pre-trained with {} dataset, choose {}".format(choice_dict[c][0], choice_dict[c][1], c))
        choice = input("Your choice: ")
        if choice in choice_dict.keys():
            print("You chose the '{}' model pre-trained with {} dataset".format(choice_dict[choice][0], choice_dict[choice][1]))
            choice = int(choice)
            if choice <= 5:
                inference_imagenet(algo_choice=choice-1)
            elif choice == 6:
                yolo_imagenet()
            elif choice in [7, 8]:
                inference_coco(algo_choice=choice-7)
            correct_input = True
        else:
            print('ERROR: "{}" is a wrong input, please try again...'.format(choice))
            print(50 * '-')


if __name__ == '__main__':
    main()
