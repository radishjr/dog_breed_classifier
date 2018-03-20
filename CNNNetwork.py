import tensorflow as tf
import Settings

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input data:
    # [0]: 28*28, 3 channels
    # [1]: output label
    
    #initial_value must have a shape specified: Tensor("input/Reshape:0", shape=(?, 48, 48, 3), dtype=float32)

    with tf.name_scope('input') as scope:
        input_layer_x = features
        input_layer_y = labels
        tf.summary.image('input', input_layer_x, 3)

    with tf.name_scope('Hidden'):
        with tf.name_scope('Convolution_Layer_1'):
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer_x,
                filters=32, #提取的特征数量，conv1的第三个维度 e.g. 3通道
                data_format='channels_last',
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            print("conv1")
            print(conv1)
  
        with tf.name_scope('Pooling_Layer_1'):
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, 
                                    pool_size=[2, 2], 
                                    strides=2 #每隔多少行取样
                                )
            print("pool1")
            print(pool1)
        with tf.name_scope('Convolution_Layer_2'):
            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)
            print("conv2")
            print(conv2)

        with tf.name_scope('Pooling_Layer_2') as scope:
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            print("pool2")
            print(pool2)
        with tf.name_scope('Fully_Connection_Layer') as scope:
            # Dense Layer 
            print(pool2.shape)
            shape_1 = pool2.shape[1] * pool2.shape[2] * 64
            #shape_1 = int(Settings.IMAGE_HEIGHT / 4 * Settings.IMAGE_WIDTH / 4 * 64)
            pool2_flat = tf.reshape(pool2, [-1, shape_1])
            # print("pool2_flat")
            # print(pool2_flat)
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="fully connected")
            # print("dense")
            # print(dense)

            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training= mode == tf.estimator.ModeKeys.TRAIN,
                name="dropout")
            print("dropout")
            print(dropout)
        
    with tf.name_scope('Output_layer') as scope:  
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=Settings.CATEGORY_SIZE)
        print("logits")
        print(logits)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope('LOSS') as scope:  
        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=input_layer_y, logits=logits)
        tf.summary.scalar('cross entropy loss', loss) 

    accuracy = tf.metrics.accuracy(
            labels=input_layer_y, predictions=predictions["classes"])
    #tf.summary.scalar('accuracy', accuracy)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": accuracy}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
