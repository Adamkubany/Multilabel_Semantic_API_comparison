import pandas as pd
import os

from config import DATASET_PATHS

global DATASET, DATA_PATH, IMAGES_PATH, IMAGES_DATA_FILE, ORIGINAL_METADATA_PATH


def open_image_dataset_etl():
    """
    Open Images dataset v6 : https://storage.googleapis.com/openimages/web/download.html
    image label download link : https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv
    image data download link : https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
    classes download link : https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
    :return:
    """
    # load data
    print('Start ETL for the Open Images dataset...')
    print('Loading data...')
    image_list = [f[:-4] for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
    classes = pd.read_csv(os.path.join(ORIGINAL_METADATA_PATH, 'class-descriptions-boxable.csv'), header=None)
    classes.columns = ['ClassID', 'ClassName']
    classes['ClassNameClean'] = [cls if cls.find('(') == -1 else cls[:cls.find('(') - 1] for cls in classes.ClassName.iloc]
    img_labels = pd.read_csv(os.path.join(ORIGINAL_METADATA_PATH, 'train-annotations-human-imagelabels-boxable.csv'), low_memory=False)
    img_data = pd.read_csv(os.path.join(ORIGINAL_METADATA_PATH, 'train-images-boxable-with-rotation.csv'), low_memory=False)

    # ##### check that each image exist in original data files
    # count = 0
    # images = len(image_list)
    # image_list_data = img_data.ImageID.values
    # image_list_labels = img_labels.loc[img_labels.Confidence != 0].ImageID.unique()
    # for img_id in image_list:
    #     count += 1
    #     print('image id : {}. {}/{}'.format(img_id, count, images))
    #     if img_id not in image_list_data:
    #         print('image {} not in image data'.format(img_id))
    #     if img_id not in image_list_labels:
    #         print('image {} not in image labels'.format(img_id))
    # print('done')

    # transform data
    print('Transform data...')
    img_data_1000 = img_data.loc[(img_data.ImageID.isin(image_list))]
    img_data_1000 = img_data_1000.rename(columns={'ImageID': 'img_id', 'OriginalURL': 'url'})
    img_labels_1000 = img_labels.loc[(img_labels.ImageID.isin(image_list)) & (img_labels.Confidence > 0)]
    img_labels_1000 = img_labels_1000.rename(columns={"LabelName": "LabelNameID"})
    img_labels_1000['img_id'] = img_labels_1000.ImageID
    img_labels_1000["names"] = [classes.loc[classes.ClassID == lbl]['ClassName'].values[0] for lbl in img_labels_1000.LabelNameID]

    img_labels_1000_distinct = pd.DataFrame(columns=['img_id', 'names'])
    count, n = 0, len(img_data_1000)
    for img_id in image_list:
        count += 1
        print('Extracting unique labels for image id {}. img {}/{}'.format(img_id, count, n))
        for distinct_object in img_labels_1000.loc[img_labels_1000.img_id == img_id].names.sort_values().unique():
            img_labels_1000_distinct = img_labels_1000_distinct.append({'img_id': img_id, 'names': distinct_object}, ignore_index=True)

    # save new data files
    print('Saving data...')
    pd.DataFrame(image_list, columns=['img_id']).to_csv(os.path.join(DATA_PATH, '1000new_images.csv'), index=None)
    img_labels_1000_distinct[['img_id', 'names']].to_csv(os.path.join(DATA_PATH, '1000new_img_objects_dist.csv'), index=None)
    img_data_1000.to_csv(os.path.join(DATA_PATH, '1000new_img_data.csv'), index=None)
    print('ETL done!')


def visual_genome_dataset_etl():
    """
    visual genome dataset link : https://visualgenome.org/api/v0/api_home.html
    image data download link : https://visualgenome.org/static/data/dataset/image_data_v1.json.zip
    image objects download link : https://visualgenome.org/static/data/dataset/objects_v1.json.zip
    :return:
    """
    # load data
    print('Start ETL for the Visual Genome dataset...')
    print('Loading data...')
    image_list = [f[:-4] for f in os.listdir(IMAGES_PATH) if os.path.isfile(os.path.join(IMAGES_PATH, f))]
    img_data = pd.read_json(os.path.join(ORIGINAL_METADATA_PATH, 'image_data.json'))
    img_labels_raw = pd.read_json(os.path.join(ORIGINAL_METADATA_PATH, 'objects.json'))

    # transform data
    print('Transform data...')
    img_data = img_data.rename(columns={'id': 'img_id'})
    img_data_1000 = img_data.loc[(img_data.img_id.isin(image_list))]
    img_labels_raw = img_labels_raw.rename(columns={'id': 'img_id'})
    img_labels_raw_1000 = img_labels_raw.loc[(img_data.img_id.isin(image_list))].reset_index()

    img_labels_1000 = pd.DataFrame(columns=['img_id', 'names'])
    img_labels_1000_distinct = pd.DataFrame(columns=['img_id', 'names'])

    count = 0
    n = len(img_data)
    for index, img in img_labels_raw_1000.iterrows():
        count += 1
        print('Extracting labels for image id {}. img {}/{}'.format(img.img_id, count, n))
        for obj in img_labels_raw_1000.iloc[index].objects:
            img_labels_1000 = img_labels_1000.append({'img_id': img.img_id, 'names': obj['names'][0].replace('"', '')}, ignore_index=True)
        for distinct_object in img_labels_1000.loc[img_labels_1000.img_id == img.img_id].names.sort_values().unique():
            img_labels_1000_distinct = img_labels_1000_distinct.append({'img_id': img.img_id, 'names': distinct_object}, ignore_index=True)

    # save data
    print('Saving data...')
    pd.DataFrame(image_list, columns=['img_id']).to_csv(os.path.join(DATA_PATH, '1000new_images.csv'), index=None)
    img_data_1000.to_csv(os.path.join(DATA_PATH, '1000new_img_data.csv'), index=None)
    img_labels_1000_distinct[['img_id', 'names']].to_csv(os.path.join(DATA_PATH, '1000new_img_objects_dist.csv'), index=None)
    print('ETL done!')


def main():
    global DATASET, DATA_PATH, IMAGES_PATH, IMAGES_DATA_FILE, ORIGINAL_METADATA_PATH
    correct_db = False
    db_dict =  {'0': 'OPEN_IMAGE',
                '1': 'VISUAL_GENOME'}
    while not correct_db:
        print('Please choose dataset to ETL:')
        for db in db_dict.keys():
            print('For {} dataset, choose {}'.format(db_dict[db], db))
        db_choice = input("Your choice: ")
        if db_choice in db_dict.keys():
            print("You chose to perform ETL on the '{}' dataset".format(db_dict[db_choice]))
            correct_db = True
            DATASET = db_dict[db_choice]
            DATA_PATH = DATASET_PATHS[DATASET]['DATA_PATH']
            IMAGES_PATH = DATASET_PATHS[DATASET]['IMAGES_PATH']
            IMAGES_DATA_FILE = DATASET_PATHS[DATASET]['IMAGES_DATA_FILE']
            ORIGINAL_METADATA_PATH = DATASET_PATHS[DATASET]['ORIGINAL_METADATA_PATH']
            if DATASET == "OPEN_IMAGE":
                open_image_dataset_etl()
            else:
                visual_genome_dataset_etl()


if __name__ == '__main__':
    main()

