
# from keras.models import Model
import numpy as np
import os
import pandas as pd

from config import imagenet_classes as img_clsses
from config import SAVE_PATH, PROJECT_ROOT

img_path = 'first1000images'


def inference_imagenet(algo_choice=1, top=5, print_every=10):
    # reference code from:
    # https://keras.io/applications/
    from keras.preprocessing import image

    algos = ['InceptionResNetV2', 'mobilenet_v2', 'VGG19', 'ResNet50']
    algo = algos[algo_choice]
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
    elif algo == 'VGG19':
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg19 import preprocess_input
        base_model = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
        # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    elif algo == 'ResNet50':
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

    image_list = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    img_count = 0
    for img_file_name in image_list:
        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {} for algo {} top {}'.format(img_file_name, img_count, len(image_list), algo, top))
        img_name = img_file_name[:img_file_name.find('.')]
        cur_img_path = img_path + '/' + img_file_name
        if algo in ['InceptionResNetV2']:
            img = image.load_img(cur_img_path, target_size=(299, 299))  # for InceptionResNetV2
        elif algo in ['mobilenet_v2', 'VGG19', 'ResNet50']:
            img = image.load_img(cur_img_path, target_size=(224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_labels_coef = base_model.predict(x)[0]
        top_labels_indx = img_labels_coef.argsort()[-top:][::-1]
        for indx in top_labels_indx:
            label_dict = {'img_id': img_name, 'label': img_clsses[indx], 'conf_level': img_labels_coef[indx], 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)

    df.to_csv(os.path.join(SAVE_PATH, algo + '_obj_inference_data.csv'), index=None)
    return print('Done!')


def inference_coco(algo_choice=1, print_every=10):
    from imageai.Detection import ObjectDetection

    algos = ['resnet_coco', 'yolo_v3_coco']
    algo = algos[algo_choice]
    print('algo {}'.format(algo))
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'Sources', 'pretrained_models')
    model_name = ["resnet50_coco_best_v2.0.1.h5", "yolo.h5"]
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    cur_dir = os.getcwd()

    detector = ObjectDetection()

    if algo == 'resnet_coco':
        detector.setModelTypeAsRetinaNet()

    elif algo == 'yolo_v3_coco':
        detector.setModelTypeAsYOLOv3()

    detector.setModelPath(os.path.join(model_path, model_name[algo_choice]))
    detector.loadModel()
    image_list = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    img_count = 0
    for img_file_name in image_list:
        img_name = img_file_name[:img_file_name.find('.')]
        # cur_img_path = img_path + '/' + img_file_name
        cur_img_path = os.path.join(cur_dir, img_path, img_file_name)

        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {}, algo {}'.format(img_file_name, img_count, len(image_list), algo))
        detections = detector.detectObjectsFromImage(input_image=cur_img_path, output_image_path='results/temp_image.jpg', minimum_percentage_probability=10)
        for label in detections:
            label_dict = {'img_id': img_name, 'label': label['name'], 'conf_level': label['percentage_probability'], 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)

    df.to_csv(os.path.join(SAVE_PATH, algo + '_obj_inference_data.csv'), index=None)
    print('Done!')


def yolo_imagenet(print_every=10):
    # https://pjreddie.com/darknet/imagenet/#darknet53_448
    # img_path = 'first1000images'
    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    algo = 'yolo_v3_imagenet'
    image_list = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    img_count = 0
    for img_file_name in image_list:
        img_count += 1
        if img_count % print_every == 0:
            print('image {}, count {} / {}'.format(img_file_name, img_count, len(image_list)))
        img_name = img_file_name[:img_file_name.find('.')]
        cur_img_path = img_path + '/' + img_file_name
        img_data = os.popen("./darknet classifier predict cfg/imagenet1k.data cfg/darknet53.cfg darknet53.weights {}".format(cur_img_path)).read()
        img_data_ = img_data.split('\n')[:-1]
        for label_data in img_data_:
            label = label_data[label_data.find(':') + 2:]
            conf = label_data[:label_data.find('%')]
            label_dict = {'img_id': img_name, 'label': label, 'conf_level': conf, 'run_date': '2020', 'api': algo}
            df = df.append(label_dict, ignore_index=True)
    df.to_csv(os.path.join(SAVE_PATH, algo + '_obj_inference_data.csv'), index=None)
    print('Done!')


def main():
    correct_input = False
    choice_dict = {"1": ['InceptionResNetV2', 'ImageNet'],
                   "2": ['mobilenet_v2', 'ImageNet'],
                   "3": ['VGG19', 'ImageNet'],
                   "4": ['ResNet50', 'ImageNet'],
                   "5": ['YOLO', 'ImageNet'],
                   "6": ['ResNet50', 'COCO'],
                   "7": ['YOLO', 'COCO'],
                   }
    while not correct_input:
        correct_options = []
        for c in choice_dict:
            correct_options.append(c)
            print("For '{}' model pre-trained with {} dataset, choose {}".format(choice_dict[c][0], choice_dict[c][1], c))
        choice = input("Your choice: ")
        if choice in correct_options:
            print("You chose the '{}' model pre-trained with {} dataset".format(choice_dict[choice][0], choice_dict[choice][1]))
            choice = int(choice)
            if choice <= 4:
                inference_imagenet(algo_choice=choice-1)
            elif choice == 5:
                yolo_imagenet()
            elif choice in [6, 7]:
                inference_coco(algo_choice=choice-6)
            correct_input = True
        else:
            print('ERROR: "{}" is a wrong input, try again...'.format(choice))
            print('---------------')


if __name__ == '__main__':
    main()
