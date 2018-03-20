import tensorflow as tf
import pandas as pd
import CNNNetwork as CNN
import DatasetGenerater as DG
import matplotlib.pyplot as plt 
import Settings

tf.logging.set_verbosity(tf.logging.INFO)

dog_breed_classifier = tf.estimator.Estimator(
    model_fn=CNN.cnn_model_fn, model_dir="/Users/Rambo/Desktop/faceoff/dog_breed_data/tmp/dog_breed_model3")

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



    # tensorboard_dir = 'tensorboard/mnist3'   # 保存到新的目录
    # if not os.path.exists(tensorboard_dir):
    #     os.makedirs(tensorboard_dir)

    # merged_summary = tf.summary.merge_all()   # 使用tf.summary.merge_all()，可以收集以上定义的所有信息
    # writer = tf.summary.FileWriter(tensorboard_dir)
    # writer.add_graph(session.graph)




    train_classifier = dog_breed_classifier.train(
        input_fn=DG.train_input_fn,
        steps=Settings.NUM_EPOCHS,
        hooks=[
            logging_hook, 
            summary_hook
        ])

    print("train finished, evaluate")

    eval_results = dog_breed_classifier.evaluate(
        input_fn=DG.eval_input_fn,
        hooks=[
            logging_hook, 
            summary_hook
        ])
    print(eval_results)