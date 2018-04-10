# from tensorflow.contrib.tensorboard.plugins import projector
# import tensorflow as tf

# # Create randomly initialized embedding weights which will be trained.
# N = 10000 # Number of items (vocab size).
# D = 200 # Dimensionality of the embedding.

# w = 20
# h = 20

# embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
# # Specify the width and height of a single thumbnail.
# embedding.sprite.single_image_dim.extend([w, h])

# embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')

# # Format: tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto
# config = projector.ProjectorConfig()

# # You can add multiple embeddings. Here we add only one.
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
# # Link this tensor to its metadata file (e.g. labels).
# embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')


# # Use the same LOG_DIR where you stored your checkpoint.
# summary_writer = tf.summary.FileWriter(LOG_DIR)

# # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# # read this file during startup.
# projector.visualize_embeddings(summary_writer, config)