import tensorflow as tf
import pandas as pd
import os
#from distutils.sysconfig import get_python_lib
#print(get_python_lib())
import numpy as np
import pandas as pd
import os

def generateFeedingDataCSV():
    print("generateFeedingDataCSV 0")
    feeding_data_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    feeding_data_csv_path = feeding_data_path + "labels.csv"
    if not os.path.isfile(feeding_data_csv_path):
        
        print("generateFeedingDataCSV 1")
        input_data_path = feeding_data_path + "input_data/"

        csv_path = input_data_path + "labels.csv"
        dog_breed_df = pd.read_csv(csv_path)
        updated = False

        index_list = dog_breed_df['breed'].value_counts().index
        key_value_mapping = {}
        current_i = 0

        print("generateFeedingDataCSV 2")
        for index in index_list:
            key_value_mapping[index] = current_i
            current_i = current_i+1

        dog_breed_df['breed_int'] = dog_breed_df['breed'].map(key_value_mapping)
        updated = True

        dog_breed_df["filepath"] = dog_breed_df["id"].apply(lambda x: input_data_path + "images/train/" + x + ".jpg")
        updated = True

        print("generateFeedingDataCSV updated")
        print(updated)
        if updated:
            dog_breed_df.to_csv(feeding_data_csv_path)


def _parse_function_X(filename, labels):
    #将图片数据转换成矩阵输入
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, 
        [tf.flags.FLAGS.image_size, tf.flags.FLAGS.image_size], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.to_float(image_resized)
    #label_matrix = tf.to_int32(tf.one_hot(label, tf.flags.FLAGS.category_size))

    return image_resized, labels

def generateTrainingData():
    print("generateTrainingData")
    feeding_data_path = os.path.dirname(os.path.abspath(__file__)) + "/"
    feeding_data_csv_path = feeding_data_path + "labels.csv"
    if not os.path.isfile(feeding_data_csv_path):
        generateFeedingDataCSV()

    current_path = os.path.dirname(os.path.abspath(__file__))
    csvfile = current_path + "/labels.csv"

    image_breed_list = pd.read_csv(csvfile)

    pathList = image_breed_list["filepath"].tolist()
    labelList = image_breed_list['breed_int'].tolist()

    filenames = tf.constant(pathList)
    labels = tf.constant(labelList)

    #imageDataset = tf.data.Dataset.from_tensor_slices(filenames)
    #imageDataset = imageDataset.map(_parse_function_X)
    #labelsDataset = tf.data.Dataset.from_tensor_slices(labels)
    #return imageDataset, labelsDataset

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function_X)
    return dataset

def train_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''
    train_dataset = generateTrainingData()
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.repeat(tf.flags.FLAGS.num_epochs)
    train_dataset = train_dataset.batch(tf.flags.FLAGS.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()
    features, labels = train_iterator.get_next()        
    return features, labels

def eval_input_fn():
    '''
    训练输入函数，返回一个 batch 的 features 和 labels
    '''
    eval_dataset = generateTrainingData()
    eval_dataset = eval_dataset.repeat(1)
    eval_dataset = eval_dataset.batch(10)
    eval_iterator = eval_dataset.make_one_shot_iterator()
    features, labels = eval_iterator.get_next()        
    return features, labels
    