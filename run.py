import tensorflow as tf
import pandas as pd
import CNNNetwork as CNN
import DatasetGenerater as DG
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__)) + "/"
model_dir = current_path + "tmp/dog_breed_model_100"

tf.flags.DEFINE_integer('category_size', 120, 'batch size, default: 120')
tf.flags.DEFINE_integer('num_epochs', 100000, 'batch size, default: 100000')
tf.flags.DEFINE_integer('batch_size', 200, 'batch size, default: 200')
tf.flags.DEFINE_integer('image_size', 128, 'image size, default: 128')

tf.logging.set_verbosity(tf.logging.INFO)


dog_breed_classifier = tf.estimator.Estimator(
    model_fn=CNN.cnn_model_fn, model_dir=model_dir)

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
    #for i in range(30):

    #next_x_item = x_dataset_iterator.get_next()
    #next_y_item = y_dataset_iterator.get_next()
    
    #x_data = sess.run(next_x_item)
    #y_data = sess.run(next_y_item)

    # train_input_fn_1 = tf.estimator.inputs.numpy_input_fn(
    #    x={"x": x_data},
    #    y=y_data,
    #    batch_size=50,
    #    num_epochs=None,
    #    shuffle=True)

    train_classifier = dog_breed_classifier.train(
        input_fn=DG.train_input_fn,
        steps=tf.flags.FLAGS.num_epochs,
        hooks=[
            logging_hook, 
            summary_hook
        ])

    saver = tf.train.Saver()
    saver.save(sess, 'model/model.ckpt')
    print("train finished, evaluate")

    eval_results = dog_breed_classifier.evaluate(
        input_fn=DG.eval_input_fn,
        hooks=[
            logging_hook, 
            summary_hook
        ])
    print(eval_results)