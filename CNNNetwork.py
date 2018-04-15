import tensorflow as tf
import time
import datetime

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input data:
    # [0]: 128*128, 3 channels
    # [1]: output label
    
    #initial_value must have a shape specified: Tensor("input/Reshape:0", shape=(?, 48, 48, 3), dtype=float32)

    if tf.flags.FLAGS.use_gpu:
        device = '/device:GPU:0'
    else:
        device = '/device:CPU:0'

    with tf.device(device):
        with tf.name_scope('input') as scope:
            input_layer_x = features
            input_layer_y = labels
            tf.summary.image('input', input_layer_x, 3)
            
            # print("input_layer_y")
            # print(input_layer_y)

        #with tf.name_scope('Hidden'):
        with tf.name_scope('Convolution_Layer_1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer_x,
                filters=32, #提取的特征数量，conv1的第三个维度 e.g. 3通道
                data_format='channels_last',
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            #print("conv1")
            #print(conv1)

        with tf.name_scope('Pooling_Layer_1'):
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2], 
                                    strides=2 #每隔多少行取样
                                )
            #print("pool1")
            #print(pool1)
        with tf.name_scope('Convolution_Layer_2'):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            #print("conv2")
            #print(conv2)

        with tf.name_scope('Pooling_Layer_2') as scope:
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            #print("pool2")
            #print(pool2)

        # with tf.name_scope('Convolution_Layer_3'):
        #     # Convolutional Layer #2 and Pooling Layer #2
        #     conv3 = tf.layers.conv2d(
        #         inputs=pool2,
        #         filters=128,
        #         kernel_size=[5, 5],
        #         padding="same",
        #         activation=tf.nn.relu)
        #     #print("conv2")
        #     #print(conv2)

        # with tf.name_scope('Pooling_Layer_3') as scope:
        #     pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        #     #print("pool3")
        #     #print(pool3)

        with tf.name_scope('Fully_Connection_Layer') as scope:
            # Dense Layer 

            # print(pool3.shape)
            # shape_1 = pool3.shape[1] * pool3.shape[2] * 128
            # pool3_flat = tf.reshape(pool3, [-1, shape_1])
            # dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu, name="fullyConnected")
            
            print(pool2.shape)
            shape_1 = pool2.shape[1] * pool2.shape[2] * 64
            pool2_flat = tf.reshape(pool2, [-1, shape_1])
            #print("pool2_flat")
            #print(pool2_flat)
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="fullyConnected")
            # print("dense")
            # print(dense)

            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training= mode == tf.estimator.ModeKeys.TRAIN,
                name="dropout")
            print("dropout")
            print(dropout)
            
        with tf.name_scope('Output_layer') as scope:  
            # Logits Layer
            logits = tf.layers.dense(inputs=dropout, units=tf.flags.FLAGS.category_size)
            
            print("logits")
            print(logits)

        #with tf.name_scope('Prediction') as scope:  
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        accuracy = tf.metrics.accuracy(
                labels=input_layer_y, predictions=predictions["classes"], name="accuracy")
    
    class LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""
        def __init__(self, log_steps=None):
            self.log_steps = log_steps

        def begin(self):
            self._step = -1
            self._start_time = time.time()

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss, input_layer_y, predictions, accuracy])  # Asks for loss value.

        def after_run(self, run_context, run_values):
            if self._step % self.log_steps == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value, input_layer_y_value, predictions_value, accuracy_value= run_values.results
                examples_per_sec = self.log_steps * tf.flags.FLAGS.batch_size / duration
                sec_per_batch = float(duration / self.log_steps)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                print ("accuracy_0:")
                print (accuracy_value[0])
                print ("accuracy_1:")
                print (accuracy_value[1])
                #print (format_str % (datetime.datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))
                #print ("input_layer_y_value")
                #print (input_layer_y_value)
                #print ("predictions_value[\"classes\"]")
                #print (predictions_value["classes"])
                for i in range(predictions_value["classes"].shape[0]):
                    if input_layer_y_value[i] == predictions_value["classes"][i]:
                        print("index %d: %d" %(i, input_layer_y_value[i]))

            
    logger_hook = LoggerHook(
        log_steps=10
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, 
            evaluation_hooks=[logger_hook])

    with tf.name_scope('LOSS') as scope:  
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=input_layer_y, logits=logits)
        tf.summary.scalar("cross entropy loss", loss)    

    #correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(input_layer_y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy_0", accuracy[0])
    tf.summary.scalar("accuracy_1", accuracy[1])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.flags.FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss, 
            train_op=train_op,
            predictions=predictions,
            training_hooks=[logger_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": accuracy}
    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            predictions=predictions)
