
import pandas as pd
import os
import io
import sys

from config import DATASET_PATHS, AUTH, SAVE_FILE_FLAG, api_numbers, top_labels

global DATASET, DATA_PATH, IMAGES_PATH, IMAGES_DATA_FILE, ORIGINAL_METADATA_PATH


def inference_imagga():
    """
    API explanation link : https://docs.imagga.com/?python#getting-started-signup
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    import requests
    api = 'imagga'
    api_num = api_numbers[api]
    api_key = AUTH[api]['api_key']
    api_secret = AUTH[api]['api_secret']

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} form {}. {} / {}'.format(img.img_id, api, count, images))
        image_url = img.url
        response = requests.get('https://api.imagga.com/v2/tags?image_url=%s' % image_url, auth=(api_key, api_secret))
        res = response.json()
        if res.get('result') is None:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id}, ignore_index=True)
            continue
        for label in res['result']['tags']:
            df = df.append({'img_id': img.img_id, 'label': label['tag']['en'], 'conf_level': label['confidence'], 'run_date':2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_ibm():
    """
    API explanation link : https://cloud.ibm.com/apidocs/visual-recognition/visual-recognition-v3?code=python
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    # from ibm_watson import VisualRecognitionV4
    # from ibm_watson.visual_recognition_v4 import AnalyzeEnums, FileWithMetadata
    from ibm_watson import VisualRecognitionV3
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    api = 'ibmwatson'
    api_num = api_numbers[api]
    authenticator = IAMAuthenticator(AUTH[api]['apikey'])
    visual_recognition = VisualRecognitionV3(version='2018-03-19', authenticator=authenticator)

    visual_recognition.set_service_url(AUTH[api]['url'])

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} from {}. {} / {}'.format(img.img_id, api, count, images))
        image_url = img.url
        try:
            res = visual_recognition.classify(url=image_url).get_result()
        except:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id}, ignore_index=True)
            continue

        labels = dict()
        for label in res['images'][0]['classifiers'][0]['classes']:
            labels[label['class']] = label['score']
        sorted_labels = sorted(labels, key=labels.get, reverse=True)
        for label in sorted_labels:
            df = df.append({'img_id': img.img_id, 'label': label, 'conf_level': labels[label], 'run_date': 2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_clarifai():
    """
    API explanation links :
        https://portal.clarifai.com/signup
        https://docs.clarifai.com/api-guide/api-overview/api-clients
        actual script: https://github.com/Clarifai/clarifai-python-grpc/
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel

    from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
    from clarifai_grpc.grpc.api.status import status_code_pb2
    channel = ClarifaiChannel.get_json_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    api = 'clarifai'
    api_num = api_numbers[api]
    metadata = (('authorization', AUTH[api]['authorization']),)

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} from {}. {} / {}'.format(img.img_id, api, count, images))
        image_url = img.url
        try:
            request = service_pb2.PostModelOutputsRequest(
                model_id=AUTH[api]['model_id'],
                inputs=[resources_pb2.Input(data=resources_pb2.Data(image=resources_pb2.Image(url=image_url)))])

            res = stub.PostModelOutputs(request, metadata=metadata)
        except:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id}, ignore_index=True)
            continue
        if res.status.code != status_code_pb2.SUCCESS:
            print('img "{}" is unavailable, ERROR 2'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id}, ignore_index=True)
            continue
            # raise Exception("Request failed, status code: " + str(response.status.code))
        for label in res.outputs[0].data.concepts:
            df = df.append({'img_id': img.img_id, 'label': label.name, 'conf_level': label.value, 'run_date': 2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_microsoft():
    """
    API explanation links :
        https://docs.microsoft.com/bs-cyrl-ba/azure/cognitive-services/computer-vision/quickstarts/python-disk
        https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/client-library?pivots=programming-language-python
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
    import requests

    api = 'microsoft_oxford'
    api_num = api_numbers[api]
    endpoint = AUTH[api]['endpoint']
    subscription_key = AUTH[api]['subscription_key']
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # for local image when the url does not work
    analyze_url = endpoint + "vision/v3.0/analyze"
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}

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
            res = computervision_client.detect_objects(image_url)
            objects = res.objects
        except:
            print('img "{}" is unavailable for URL service, try local image'.format(img.img_id))
            # local image (does not return conf level)
            image_path = os.path.join(IMAGES_PATH, img.img_id + '.jpg')
            image_data = open(image_path, "rb").read()
            res = requests.post(analyze_url, headers=headers, params=params, data=image_data).json()

            objects = []
            for i, label in enumerate(res['description']['tags'][:top_labels]):
                objects.append(type('objclass', (object,), {'object_property': label, 'confidence': 0.8})())

            # df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'error'}, ignore_index=True)
            # continue
        if len(objects) == 0:
            print('img "{}" has no objects'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'no objects'}, ignore_index=True)
            continue
        print('img {} produce {} objetcs'.format(img.img_id, len(objects)))
        for label in objects:
            df = df.append({'img_id': img.img_id, 'label': label.object_property, 'conf_level': label.confidence, 'run_date': 2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_google():
    """
    API explanation links : https://cloud.google.com/vision/docs/quickstart-client-libraries
                            https://console.cloud.google.com/
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    # Imports the Google Cloud client library
    from google.cloud import vision
    from google.cloud.vision import types
    api = 'googlevision'
    api_num = api_numbers[api]
    client = vision.ImageAnnotatorClient.from_service_account_json(AUTH[api]['config_json'])

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id', 'note'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} from {}. {} / {}'.format(img.img_id, api, count, images))
        # image_url = img.url
        file_name = os.path.join(IMAGES_PATH, '{}.jpg'.format(img.img_id))
        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()
        image = types.Image(content=content)

        try:
            res = client.label_detection(image=image)
        except:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'error'}, ignore_index=True)
            continue
        if len(res.label_annotations) == 0:
            print('img "{}" has no objects'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'no objects'}, ignore_index=True)
            continue

        for label in res.label_annotations:
            df = df.append({'img_id': img.img_id, 'label': label.description, 'conf_level': label.score, 'run_date': 2020, 'api': api}, ignore_index=True)
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)
    return df, df_unavailable


def inference_wolfram():
    """
    wolfram language download site : https://www.wolfram.com/language/
    API explanation links :
        Announcing the Wolfram Client Library for Python : https://blog.wolfram.com/2019/05/16/announcing-the-wolfram-client-library-for-python/
        general python API usage explanation : https://reference.wolfram.com/language/WolframClientForPython/
    :return:
    df : dataframe with image objects
    df_unavailable : dataframe with images which were not inferred
    """
    from wolframclient.evaluation import WolframLanguageSession
    session = WolframLanguageSession()
    from wolframclient.language import wl, wlexpr

    api = 'wolfram'
    api_num = api_numbers[api]

    # from PIL import Image
    # img = Image.open(file_name)

    df = pd.DataFrame(columns=['img_id', 'label', 'conf_level', 'run_date', 'api'])
    df_unavailable = pd.DataFrame(columns=['img_id', 'note'])

    img_data = pd.read_csv(IMAGES_DATA_FILE)
    images = len(img_data)
    count = 0
    for _, img in img_data.iterrows():
        count += 1
        print('infering image id {} from {}. {} / {}'.format(img.img_id, api, count, images))
        file_name = os.path.join(IMAGES_PATH, '{}.jpg'.format(img.img_id))
        try:
            im = session.evaluate(wl.Import(file_name))
            res_entities = session.evaluate(wl.ImageIdentify(im, wl.All, 5))
            res = session.evaluate(wl.EntityValue(res_entities, "Name"))
            res = list(res)
            # session.evaluate(wl.ImageDimensions(im))

        except:
            print('img "{}" is unavailable'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'error'}, ignore_index=True)
            continue
        if len(res) == 0 or len(res) == 1:
            print('img "{}" has no objects'.format(img.img_id))
            df_unavailable = df_unavailable.append({'img_id': img.img_id, 'note': 'no objects'}, ignore_index=True)
            continue
        # if len(res) > 2:
        # print('{} img "{}" has "{}" objects {}'.format(5*'*', img.img_id, len(res), 5*'*'))
        #     break
        for i, label in enumerate(res):
        # df = df.append({'img_id': img.img_id, 'label': res[1][:res[1].find(':')], 'conf_level': 1, 'run_date': 2020, 'api': api}, ignore_index=True)
            df = df.append({'img_id': img.img_id, 'label': label, 'conf_level': (len(res) - i) / len(res), 'run_date': 2020, 'api': api}, ignore_index=True)
    session.terminate()
    if SAVE_FILE_FLAG:
        df.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}.csv'.format(api_num, api)), index=None)
        df_unavailable.to_csv(os.path.join(DATA_PATH, '1000new_img_{}_{}_unavailable.csv'.format(api_num, api)), index=None)


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
            ORIGINAL_METADATA_PATH = DATASET_PATHS[DATASET]['ORIGINAL_METADATA_PATH']
        else:
            print('ERROR: "{}" is a wrong input, please try again...'.format(db_choice))
            print(50 * '-')
    correct_input = False
    choice_dict = {
                   "1": 'imagga',
                   "2": 'ibm',
                   "3": 'clarifai',
                   "4": 'microsoft',
                   "5": 'google',
                   "6": 'wolfram',
                   }
    while not correct_input:
        print("Commercial API inference for the '{}' dataset. Choose from the following options:".format(DATASET))
        for c in choice_dict.keys():
                print("For inference with the '{}' API, choose {}".format(choice_dict[c], c))
        choice = input("Your choice: ")
        if choice in choice_dict.keys():
            print("You chose the '{}' option".format(choice_dict[choice]))
            getattr(sys.modules[__name__], "inference_%s" % choice_dict[choice])()
            correct_input = True
        else:
            print('ERROR: "{}" is a wrong input, please try again...'.format(choice))
            print(50 * '-')


if __name__ == '__main__':
    main()
