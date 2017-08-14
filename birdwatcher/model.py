import tensorflow as tf
import pdb


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def _activation_summary(scope_name, x):
    tf.summary.histogram(scope_name + '/activations', x)
    tf.summary.scalar(scope_name + '/sparsity', tf.nn.zero_fraction(x))

def _image_summary(name, x):
    tf.summary.image(name, tf.reshape(x, [-1, tf.to_int32(x.shape[1]), tf.to_int32(x.shape[2]), 1]))

def _batch_normalize(name, x):
    batch_normalization_axis = 3 # image channel
    return tf.layers.batch_normalization(x,
            axis=batch_normalization_axis,
            name=name)

def evaluation(logits, labels):
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

def inference(images, keep_prob):
    x_image = tf.reshape(images, [-1, 256, 512, 1])
    _image_summary("input_images", x_image)


    # conv (64 5x5 kernels, stride size 1x2)
    with tf.variable_scope('conv1') as scope:
        x_image = _batch_normalize("input_batch_norm", x_image)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 2, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _image_summary(scope.name, conv1)
        _activation_summary(scope.name, conv1)

    with tf.variable_scope('max_pool1') as scope:
        pool1 = max_pool_2x2(conv1)
        _image_summary(scope.name, pool1)


    # conv (64 5x5 kernels, stride size 1x1)
    with tf.variable_scope('conv2') as scope:
        pool1 = _batch_normalize("pool1_batch_norm", pool1)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64],
                initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _image_summary(scope.name, conv2)
        _activation_summary(scope.name, conv2)

    with tf.variable_scope('max_pool2') as scope:
        pool2 = max_pool_2x2(conv2)
        _image_summary(scope.name, pool2)


    # conv (128 5x5 kernels, stride size 1x1)
    with tf.variable_scope('conv3') as scope:
        pool2 = _batch_normalize("pool1_batch_norm", pool2)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [128],
                initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _image_summary(scope.name, conv3)
        _activation_summary(scope.name, conv3)

    with tf.variable_scope('max_pool3') as scope:
        pool3 = max_pool_2x2(conv3)
        _image_summary(scope.name, pool3)


    # conv (256 5x5 kernels, stride size 1x1)
    with tf.variable_scope('conv4') as scope:
        pool3 = _batch_normalize("pool3_batch_norm", pool3)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [256],
                initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _image_summary(scope.name, conv4)
        _activation_summary(scope.name, conv4)

    with tf.variable_scope('max_pool4') as scope:
        pool4 = max_pool_2x2(conv4)
        _image_summary(scope.name, pool4)


    # conv (256 5x5 kernels, stride size 1x1)
    with tf.variable_scope('conv5') as scope:
        pool4 = _batch_normalize("pool4_batch_norm", pool4)
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 256], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [256],
                initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _image_summary(scope.name, conv5)
        _activation_summary(scope.name, conv5)

    with tf.variable_scope('max_pool5') as scope:
        pool5 = max_pool_2x2(conv5)
        _image_summary(scope.name, pool5)


    # Fully connected layer 1 -- after 2 round of downsampling, our 256x512 image
    # is down to 8x8x256 feature maps -- maps this to 1024 features.
    with tf.variable_scope('fully_connected') as scope:
        pool5 = _batch_normalize("pool5_batch_norm", pool5)
        W_fc1 = weight_variable([8 * 8 * 256, 1024])
        b_fc1 = bias_variable([1024])

        pool5_flat = tf.reshape(pool5, [-1, 8 * 8 * 256])

    with tf.name_scope('dropout1'):
        flat_drop = tf.nn.dropout(pool5_flat, keep_prob)
        fc1 = tf.nn.relu(tf.matmul(flat_drop, W_fc1) + b_fc1)
        _activation_summary(scope.name, fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Map the 1024 features to 166 classes, one for each bird 
    with tf.variable_scope('softmax_linear') as scope:
        W_fc2 = weight_variable([1024, 166])
        b_fc2 = bias_variable([166])

        y_conv = tf.matmul(fc1_drop, W_fc2) + b_fc2
        _activation_summary(scope.name, y_conv)
    return y_conv

def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer()
    return optimizer.minimize(loss)

