import tensorflow as tf
import pandas as pd
import os
import Settings
#from distutils.sysconfig import get_python_lib
#print(get_python_lib())
def _parse_function_X(filename, labels):
    #将图片数据转换成矩阵输入
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [Settings.IMAGE_WIDTH, Settings.IMAGE_HEIGHT], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image_resized = tf.to_float(image_resized)
    #label_matrix = tf.to_int32(tf.one_hot(label, Settings.CATEGORY_SIZE))

    #label_matrix = tf.constant(labels)

    #print("label_matrix")
    #print(label_matrix)

    #label_matrix = tf.to_int32(labels)#tf.data.Dataset.from_tensor_slices(label)
    return image_resized, labels

def generateTrainingData():
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
    #train_dataset = tf.data.TFRecordDataset(FLAGS.train_dataset)
    #train_dataset = train_dataset.map(parser)
    # num_epochs 为整个数据集的迭代次数
    #train_dataset = train_dataset.repeat(FLAGS.num_epochs)
    #train_dataset = train_dataset.batch(FLAGS.batch_size)
    #train_iterator = train_dataset.make_one_shot_iterator()

    #features, labels = train_iterator.get_next()

    train_dataset = generateTrainingData()
    train_dataset = train_dataset.shuffle(1000)
    train_dataset = train_dataset.repeat(Settings.NUM_EPOCHS)
    train_dataset = train_dataset.batch(Settings.BATCH_SIZE)
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