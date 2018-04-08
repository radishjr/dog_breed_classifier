import tensorflow as tf
import pandas as pd
import CNNNetwork as CNN
import DatasetGenerater as DG
import matplotlib.pyplot as plt
import os

tf.flags.DEFINE_integer('category_size', 120, 'batch size, default: 50')
tf.flags.DEFINE_integer('num_epochs', 10000, 'batch size, default: 50')
tf.flags.DEFINE_integer('batch_size', 100, 'batch size, default: 50')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 128')

tf.logging.set_verbosity(tf.logging.INFO)

current_path = os.path.dirname(os.path.abspath(__file__)) + "/"

dog_breed_classifier = tf.estimator.Estimator(
    model_fn=CNN.cnn_model_fn, model_dir=current_path + "tmp/dog_breed_model_100_128")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, 
    every_n_iter=10)

summary_hook = tf.train.SummarySaverHook(
    save_steps=10,
    #summary_op="tf.summary.merge_all"
    scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all())
    )
    
with tf.Session() as sess:

    eval_results = dog_breed_classifier.evaluate(
        input_fn=DG.eval_input_fn,
        hooks=[
            logging_hook, 
            summary_hook
        ])
    print(eval_results)