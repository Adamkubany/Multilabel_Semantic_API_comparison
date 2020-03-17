import mysql.connector

import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
from pyemd import emd
import csv
import os
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from config import PROJECT_ROOT, SAVE_PATH

# init params
# w2v_file_path_local = 'C:/Users/Adam/Google Drive/!PhD/Sources/WordEmbeddings/GoogleNews-vectors-negative300.bin'
w2v_file_path_local = os.path.join(PROJECT_ROOT, 'GoogleNews-vectors-negative300.bin')
global w2v_model
stop_words = set(stopwords.words('english'))

con = mysql.connector.connect(user='root', password='', host='localhost', database='infomedia')

APIs = ['1000new_img_4_clarifai', '1000new_img_7_googlevision', '1000new_img_2_ibmwatson', '1000new_img_1_imagga',
        '1000new_img_5_microsoft_oxford', '1000new_img_6_wolfram', '1000new_img_8_caffe',
        '1000new_img_9_deepdetect', '1000new_img_12_InceptionResNetV2', '1000new_img_11_tensorflow',
        '1000new_img_13_mobilenet_v2', '1000new_img_10_overfeat', '1000new_img_17_resnet_imgnet', '1000new_img_15_resnet_coco',
        '1000new_img_18_vgg19', '1000new_img_14_yolo_v3', '1000new_img_16_yolo_v3_coco']
api_names = {'1000new_img_1_imagga': 'Imagga',
             '1000new_img_2_ibmwatson': 'IBM Watson',
             '1000new_img_4_clarifai': 'Clarifai',
             '1000new_img_5_microsoft_oxford': 'Microsoft Computer Vision',
             '1000new_img_6_wolfram': 'Wolfram',
             '1000new_img_7_googlevision': 'Google Cloud Vision',
             '1000new_img_8_caffe': 'CaffeNet',
             '1000new_img_9_deepdetect': 'DeepDetect',
             '1000new_img_10_overfeat': 'Overfeat',
             '1000new_img_11_tensorflow': 'Inception-v3',
             '1000new_img_12_InceptionResNetV2': 'InceptionResNet-v2',
             '1000new_img_13_mobilenet_v2': 'MobileNet-v2',
             '1000new_img_14_yolo_v3': 'YOLO-v3',
             '1000new_img_15_resnet_coco': 'ResNet50_coco',
             '1000new_img_16_yolo_v3_coco': 'YOLO-v3_coco',
             '1000new_img_17_resnet_imgnet': 'ResNet50',
             '1000new_img_18_vgg19': 'VGG19',
             '1000new_img_objects_dist': 'True Labels'}


def get_word_vec(name):
    """
    get the word2vec vector

    :param name: label name
    :return: return either the word2vec embedding vector or False
    """
    global w2v_model
    try:
        word_vec = np.array(w2v_model[name])
    except KeyError:
        word_vec = False
    return word_vec


def sanitize_name(text, san_char):
    """
    sanitize string from unwanted chars

    :param text: input string
    :param san_char: chars to remove
    :return: cleaned string
    """
    filtered_text = []
    remove_chars = ['-', '%', '#', '@', '!', '?', '.', ',', '/', '\\', '"', "'", '[', ']', '(', ')']
    keep_words = ['can', 'all']

    for ch in remove_chars:
        text = text.replace(ch, san_char)

    for w in word_tokenize(text):
        if w not in stop_words or w in keep_words:
            filtered_text.append(w)

    filtered_text = ' '.join(filtered_text)

    return filtered_text


def sanitize_synset(synsets):
    if synsets != []:
        clean_synset = synsets[0].split('.')[0]
    else:
        clean_synset = ''
    return clean_synset


def get_w2v_embd(names, synsets=['']):
    """
    choosing the first word with word2vec vector for labels
    :param names:
    :param synsets:
    :return: word name and vec or 'unknown' word with its vector
    """
    name = 'unknown'
    word_vec = False
    if type(names) != list:
        names = [names]
    for n in names:  # for the case of multiple names ['aa', 'bb']
        n = sanitize_name(n, '')  # remove unwanted chars and stopwords
        if n in 'a b c d e f g h i j k l m n o p q r s t u v w x y z empty' or n.replace(" ", "") == '':
            continue
        for nn in [n, n.replace('a_', ''), n.replace('the_', '')]:
            name = nn
            word_vec = get_word_vec(name)
            if word_vec is False:
                name = nn.replace("_", "")  # remove spaces
                word_vec = get_word_vec(name)
            if word_vec is False:
                name = nn.title()  # cap the first letter of each word
                word_vec = get_word_vec(name)
            if word_vec is not False:
                break
        if word_vec is not False:
            break
    if word_vec is False:
        name = 'unknown'
        word_vec = get_word_vec(name)

    return word_vec


def clean_label(label):
    """
    clean the SQL retrieved labels
    :param label:
    :return:
    """
    label = re.sub(' ', '_', label.strip())  # remove whitespace from the beginning and end of a string, and change middle space to _
    rx = re.compile('\W')
    # remove .,:;"()[]{}-+?!|<>^!@#$%^&* chars (keep unicode). remove space. to lower case
    label = rx.sub('', label).lower()
    # remove _
    # label = re.sub('_', ' ', label)
    return label


def get_img_lables(imgid, ap, top_lbls=None, clean_unknowns=False):
    """
    Retrieve image labels

    :param imgid:
    :param ap: api
    :param top_lbls:
    :param clean_unknowns:
    :return: label list
    """
    if clean_unknowns:
        unknown_vec = get_w2v_embd('unknown').reshape(1, -1)
    cursor = con.cursor(buffered=True)
    if top_lbls is None:
        top_lbls = ""
    else:
        top_lbls = "order by conf_level DESC limit 0 ," + str(top_lbls)
    cursor.execute("select * from {api_name} where img_id={image_id} {tp}"
                   .format(api_name=ap, image_id=imgid, tp=top_lbls))
    rows = cursor.fetchall()
    label_list = []
    label_string = ''
    row_count = -1
    for row in rows:
        row_count += 1
        label_list.append([])
        for word in row[1].split(','):  # split in case 2 labels in one cell sep by ,
            w = clean_label(word)
            if not clean_unknowns or (clean_unknowns and not np.array_equal(unknown_vec, get_w2v_embd(w).reshape(1, -1))):
                label_list[row_count].append(w)
                label_string += " " + w
    cursor.close()
    return label_list, label_string.strip()


def get_glove_embd(word):
    """
    Retrieve GLOVE embedding from SQL server
    :param word:
    :return:
    """
    cursor = con.cursor(buffered=True)
    cursor.execute("select * from glove6b_words_vectors_par where word='{wrd}'".format(wrd=word))
    row = cursor.fetchone()
    if row is not None:
        word_vector = row[2:len(row)]
    else:
        cursor.execute("select * from glove6b_words_vectors where word='{wrd}'".format(wrd=word))
        row_ = cursor.fetchone()
        if row_ is not None:
            word_vector = row_[2:len(row_)]
            rang = list(range(1, 51))
            tbl_columns = "`" + "`, `".join(str(x) for x in (['word_id', 'word'] + rang)) + "`"
            word_vector_vals = "'" + "', '".join(str(x) for x in row_) + "'"
            q = ("INSERT INTO `glove6b_words_vectors_par` ({columns}) VALUES ({vals})"
                 .format(columns=tbl_columns, vals=word_vector_vals))
            cursor.execute(q)
            con.commit()
            print('word "' + word + '" inserted')
        else:  # unseen word enter to partial table
            # print 'no word vector for word "' + word + '"'
            word_vector = [0] * 50
    return word_vector


def get_word_ed_matrix(word_list1, word_list2=None, embedding='w2v'):
    """
    Calc Eucldean distance matrix

    :param word_list1:
    :param word_list2:
    :param embedding:
    :return:
    """
    if embedding == 'w2v':
        get_embd = get_w2v_embd
    elif embedding == 'glove':
        get_embd = get_glove_embd
    vectors_list1 = [get_embd(w).tolist() for w in word_list1]
    if word_list2 is not None:
        vectors_list2 = [get_embd(w).tolist() for w in word_list2]
    else:
        vectors_list2 = vectors_list1
    # now we have the two vector lists
    ed_word_mat = euclidean_distances(vectors_list1, vectors_list2)
    if sum(sum(ed_word_mat)) == 0:  # all words were not in GLOVE
        print('words {} are not in embeddings'.format(word_list1))
        ed_word_mat = np.ones((len(vectors_list1), len(vectors_list2)))
    return ed_word_mat


def get_nbow_vecotr(word_string, label_dict):
    """
    Calc nBOW vector
    :param word_string:
    :param label_dict:
    :return:
    """
    n_bow = label_dict.transform([word_string])  # nBOW vector
    n_bow = n_bow.toarray().ravel()
    n_bow = n_bow.astype(np.double)
    n_bow /= n_bow.sum()
    return n_bow


def average_precision(predictions, gts_, embedding='', th=0.3):
    """
    Calc average precision and true positive hits, in semantic case use a threshold
    :param predictions:
    :param gts_: ground truths
    :param embedding:
    :param th: threshold
    :return: average precision, true positives
    """
    gts = gts_.copy()
    accum_precision = 0
    pred_count = 0
    tp = 0

    if embedding:
        if embedding == 'w2v':
            get_embd = get_w2v_embd
        elif embedding == 'glove':
            get_embd = get_glove_embd
        # gts = [get_embd(w).reshape(1, -1) for w in gts]

    for label_preds_ in predictions:
        pred_count += 1
        same = False
        cur_precision = 0
        # label_preds_ = label_preds.split(', ')
        for label_pred in label_preds_:
            gt_place = -1
            for label_gt_ in gts:
                gt_place += 1
                for label_gt in label_gt_:
                    if embedding:
                        if cosine_similarity(get_embd(label_gt).reshape(1, -1), get_embd(label_pred).reshape(1, -1))[0][0] > th:
                            same = True
                            gts.pop(gt_place)
                            break
                    else:
                        if label_gt == label_pred:
                            same = True
                            gts.pop(gt_place)
                            break
                if same:
                    break
            if same:
                break
        if same:
            tp += 1
            cur_precision = tp / pred_count
        accum_precision += cur_precision
    if tp == 0:
        ap = 0
    else:
        ap = accum_precision / tp
    return ap, tp


def metrics(tp, pred_len, gt_len, labels_num):
    """
    Calc metrics per image
    :param tp:
    :param pred_len:
    :param gt_len:
    :param labels_num:
    :return:
    """
    fn = gt_len - tp
    fp = pred_len - tp
    tn = labels_num - tp - fn - fp
    if tp == 0:
        recall, precision, f1 = 0, 0, 0
    else:
        recall = tp / gt_len
        precision = tp / pred_len
        f1 = 2 * recall * precision / (recall + precision)
    accuracy = tp / (gt_len + pred_len - tp)
    accuracy_balanced = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    return recall, precision, f1, accuracy, accuracy_balanced


def label_freq(save_fig=True, rng=200, verbose=False):
    """
    Calc the label frequency for each API
    :param save_fig:
    :param rng:
    :param verbose:
    :return: api_labels_freq, api_labels_distinct and save the graph to disk
    """
    # APIs = ['1000new_img_12_InceptionResNetV2']
    apis = APIs.copy()

    img_gt_table = '1000new_img_objects_dist'
    apis.append(img_gt_table)

    cursor_imgs = con.cursor(buffered=True)
    cursor_imgs.execute("select * from 1000new_images")
    images = cursor_imgs.fetchall()
    api_labels_freq, api_labels_distinct, api_labels = {}, {}, {}
    print_count = 50
    for api in apis:
        if api == '1000new_img_objects_dist':
            top_labels = None
        else:
            top_labels = 5
        img_counter = 0
        labels = []
        for img in images:
            img_counter += 1
            img_id = img[0]
            if img_counter % print_count == 0:
                print('image id {} # {} / {} api {} top {}'.format(img_id, img_counter, len(images), api, top_labels))
            pred_labels, _ = get_img_lables(img_id, api, top_labels)
            for label in pred_labels:
                labels = labels + label  # for multi label per object
        api_labels_freq[api] = sorted(list(Counter(labels).values()), reverse=True)
        api_labels_distinct[api] = list(Counter(labels).keys())
        api_labels[api] = labels
    if not verbose:
        for api in apis:
            print("Api {} have {} / {} unique labels".format(api, len(api_labels_distinct[api]), len(api_labels[api])))

    # rng = 200
    if save_fig:
        x = [i for i in range(rng)]
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        for api in apis:
            if api in ['1000new_img_15_resnet_coco', '1000new_img_16_yolo_v3_coco']:
                continue
            y = api_labels_freq[api][:rng] + [0] * (rng - len(api_labels_freq[api][:rng]))
            ax.plot(x, y, label=api_labels_freq[api])
        ax.set_ylabel("Frequency")
        # ax.set_title("{} Most Frequent Labels".format(rng))
        ax.legend()
        fig.savefig(os.path.join(SAVE_PATH, 'label_freq.jpg'), format='jpg', quality=100, dpi=1000, bbox_inches='tight')
    return api_labels_freq, api_labels_distinct


def count_unknowns():
    """
    Count the "unknown" labels per API
    :return: save CSV file with results
    """
    # APIs = ['1000new_img_objects_dist']
    img_gt_table = '1000new_img_objects_dist'
    apis = APIs.copy()
    apis.append(img_gt_table)

    print_count = 1
    embd_choice = 1
    embedding = ['glove', 'w2v']

    embedding = embedding[embd_choice]

    if embedding == 'w2v':
        get_embd = get_w2v_embd
    elif embedding == 'glove':
        get_embd = get_glove_embd
    # Main
    print('using "{}" embeddings'.format(embedding))

    unknown_vec = get_embd('unknown').reshape(1, -1)
    df = pd.DataFrame(columns=['API', 'images', 'label_number', 'total labels', 'unknowns'])
    cursor_imgs = con.cursor(buffered=True)
    cursor_imgs.execute("select * from 1000new_images")
    images = cursor_imgs.fetchall()

    for api in apis:
        if api == '1000new_img_objects_dist':
            top_labels = None
        else:
            top_labels = 5
        img_counter = 0
        label_counter = 0
        unknown_counter = 0
        total_label_counter = 0
        for img in images:
            img_id = img[0]
            # if img_counter ==2:
            #     break
            # img_id = 43
            if img_counter % print_count == 0:
                print('image id {} # {} / {} api {} top {}'.format(img_id, img_counter, len(images), api, top_labels))
            pred_labels, _ = get_img_lables(img_id, api, top_labels)
            if pred_labels:  # in case no such img for api
                img_counter += 1
                label_counter += len(pred_labels)
                for pred_label_ in pred_labels:
                    total_label_counter += len(pred_label_)
                    label_ok = False
                    for label in pred_label_:
                        label_vec = get_embd(label).reshape(1, -1)
                        if not np.array_equal(label_vec, unknown_vec):
                            # print('label "{}" is OK'.format(label))
                            label_ok = True
                            break

                        else:
                            print('the label "{}" is unknown in {} embeddings, count is {}/{}.'.format(label, embedding, unknown_counter, label_counter))

                    if not label_ok:
                        unknown_counter += 1
        df = df.append({'API': api, 'images': img_counter, 'label_number': label_counter, 'total labels': total_label_counter, 'unknowns': unknown_counter}, ignore_index=True)
    df.to_csv(os.path.join(SAVE_PATH, 'unknown-labels_with_total.csv'))
    cursor_imgs.close()

    con.close()


def api_example_based_metrics():
    """
    Calc the example-based and semantic metrics
    :return: save CSV with results to disk
    """
    top_predictions_options = [5, 3, 1]
    # APIs = ['1000new_img_6_wolfram']
    # top_predictions_options = [1]
    img_gt_table = '1000new_img_objects_dist'

    similarity_th = 0.4
    print_count = 1
    embd_choice = 1
    embedding = ['glove', 'w2v']
    labels_count = 3728

    # Main
    embedding = embedding[embd_choice]
    print('using "{}" embeddings'.format(embedding))

    all_file = open(os.path.join(SAVE_PATH, 'all_APIs_similarity.csv'), 'w')
    all_write = csv.writer(all_file)
    all_write.writerow(['api', 'labels', 'top_labels', 'imgs_in_api', 'api_recall', 'api_recall_semantic', 'api_precision', 'api_precision_semantic',
                        'api_f1', 'api_f1_semantic', 'api_ap', 'api_ap_sem', 'img_acc', 'img_acc_semantic', 'img_acc_bl', 'img_acc_bl_semantic',
                        'api_wmd'])

    cursor_imgs = con.cursor(buffered=True)
    cursor_imgs.execute("select * from 1000new_images")
    images = cursor_imgs.fetchall()
    for api in APIs:
        for top_labels in top_predictions_options:
            print("API: " + api + " top labels: " + str(top_labels))
            counter = 0
            imgs_in_api = 0
            api_wmd, api_wmd_clean, api_ap, api_ap_sem, api_recall, api_recall_semantic, api_precision, api_precision_semantic, api_f1, api_f1_semantic, api_acc, api_acc_semantic, api_acc_bl, api_acc_bl_semantic = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            api_file = open(os.path.join(SAVE_PATH, 'similarity_{}_top_{}.csv'.format(api, str(top_labels))), 'w')
            api_write = csv.writer(api_file)
            api_write.writerow(['img_id', 'tp', 'tp_sem', 'pred_count', 'gt_count', 'img_recall', 'img_recall_semantic', 'img_precision', 'img_precision_semantic', 'img_f1', 'img_f1_semantic',
                                'img_ap', 'img_ap_semantic', 'img_acc', 'img_acc_semantic', 'img_acc_bl', 'img_acc_bl_semantic', 'img_wmd'])

            for img in images:
                counter += 1
                img_id = img[0]
                # img_id = 285633
                if counter % print_count == 0:
                    print('image id {} # {} / {} api {} top {}'.format(img_id, counter, len(images), api, top_labels))
                gt_labels, gt_string = get_img_lables(img_id, img_gt_table)
                pred_labels, pred_string = get_img_lables(img_id, api, top_labels)

                if len([l[0] for l in pred_labels if len(l) != 0]) != 0:  # in case no such img for api
                    imgs_in_api += 1

                    cur_label_dic = CountVectorizer().fit([gt_string, pred_string])
                    if len(cur_label_dic.get_feature_names()) == 1:
                        img_wmd = 0
                    else:
                        euclidean_dis_mat = get_word_ed_matrix(cur_label_dic.get_feature_names(), embedding=embedding)
                        img_wmd = emd(get_nbow_vecotr(gt_string, cur_label_dic), get_nbow_vecotr(pred_string, cur_label_dic), np.float64(euclidean_dis_mat))
                    api_wmd += img_wmd

                    img_ap, img_tp = average_precision(pred_labels, gt_labels)
                    img_recall, img_precision, img_f1, img_acc, img_acc_bl = metrics(img_tp, len(pred_labels), len(gt_labels), labels_count)

                    api_ap += img_ap
                    api_recall += img_recall
                    api_precision += img_precision
                    api_f1 += img_f1
                    api_acc += img_acc
                    api_acc_bl += img_acc_bl

                    img_ap_semantic, img_tp_semantic = average_precision(pred_labels, gt_labels, embedding, similarity_th)
                    img_recall_semantic, img_precision_semantic, img_f1_semantic, img_acc_semantic, img_acc_bl_semantic = metrics(img_tp_semantic, len(pred_labels), len(gt_labels), labels_count)

                    api_ap_sem += img_ap_semantic
                    api_recall_semantic += img_recall_semantic
                    api_precision_semantic += img_precision_semantic
                    api_f1_semantic += img_f1_semantic
                    api_acc_semantic += img_acc_semantic
                    api_acc_bl_semantic += img_acc_bl_semantic

                    api_write.writerow([img_id, img_tp, img_tp_semantic, len(pred_labels), len(gt_labels), img_recall, img_recall_semantic, img_precision, img_precision_semantic, img_f1, img_f1_semantic,
                                        img_ap, img_ap_semantic, img_acc, img_acc_semantic, img_acc_bl, img_acc_bl_semantic, img_wmd])
                    api_file.flush()
            api_file.close()
            api_wmd /= imgs_in_api
            api_ap /= imgs_in_api
            api_ap_sem /= imgs_in_api
            api_recall /= imgs_in_api
            api_recall_semantic /= imgs_in_api
            api_precision /= imgs_in_api
            api_precision_semantic /= imgs_in_api
            api_f1 /= imgs_in_api
            api_f1_semantic /= imgs_in_api
            api_acc /= imgs_in_api
            api_acc_semantic /= imgs_in_api
            api_acc_bl /= imgs_in_api
            api_acc_bl_semantic /= imgs_in_api
            all_write.writerow([api, labels_count, top_labels, imgs_in_api, api_recall, api_recall_semantic, api_precision, api_precision_semantic, api_f1, api_f1_semantic,
                                api_ap, api_ap_sem, api_acc, api_acc_semantic, api_acc_bl, api_acc_bl_semantic, api_wmd])
            all_file.flush()
    print("Done!")
    all_file.close()
    cursor_imgs.close()
    con.close()


def main():
    global w2v_model
    correct_input = False
    choice_dict = {"1": "to calc example-based and semantic metrics", "2": "to count the unknown labels", "3": "3 to calc the labels frequencies"}
    while not correct_input:
        print("Choose from the following options:")
        print("Type 1 to calc example-based and semantic metrics.")
        print("Type 2 to count the unknown labels.")
        print("Type 3 to calc the labels frequencies.")
        choice = input("Your choice: ")
        if choice in ["1", "2", "3"]:
            print("You chose {}.".format(choice_dict[choice]))
            print("Please wait while loading the word2vec model...")
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_file_path_local, binary=True)
            if choice == "1":
                api_example_based_metrics()
            elif choice == "2":
                count_unknowns()
            elif choice == "3":
                label_freq()
            correct_input = True
        else:
            print('ERROR: "{}" is a wrong input, try again...'.format(choice))
            print('---------------')


if __name__ == '__main__':
    main()
