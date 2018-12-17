import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests

#=================
# Hyperparamaters
#=================
l2_reg = 1e-3
std_dev = 0.01
lr_rate = 0.0005
kp = 0.5
epochs = 40
batch_size = 20
#=================

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # load vgg model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # get graph
    graph = tf.get_default_graph()

    # get the above layers
    input_tensor      = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor  = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1 conv of vgg layer7
    layer_7A_out = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))
	# To build the decoder portion of FCN-8, we’ll upsample the input to the original image size.
	# The shape of the tensor after the final convolutional transpose layer will be 4-dimensional:
	# (batch_size, original_height, original_width, num_classes).
	# Let’s implement those transposed convolutions we discussed earlier as follows:

	# upsample
    layer_4A_in1 = tf.layers.conv2d_transpose(layer_7A_out, num_classes, 4, strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))

    # 1x1 conv of vgg layer4
    layer_4A_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))
	# The transpose convolutional layers increase the height and width dimensions of the 4D input Tensor.

	# skip connection
    layer_4A_out = tf.add(layer_4A_in1, layer_4A_in2)

	# upsample
    layer_3A_in1 = tf.layers.conv2d_transpose(layer_4A_out, num_classes, 4, strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))

	# 1x1 conv of vgg layer 3
    layer_3A_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))

	# skip connection
    layer_3A_out = tf.add(layer_3A_in1, layer_3A_in2)

	# upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer_3A_out, num_classes, 16, strides=(8, 8), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), kernel_initializer=tf.random_normal_initializer(stddev=std_dev))

	# Skip Connections

	# The final step is adding skip connections to the model. In order to do this we’ll combine the output of two layers.
	# The first output is the output of the current layer. The second output is the output of a layer further back in the network,
	# typically a pooling layer. In the following example we combine the result of the previous layer with the result
	# of the 4th pooling layer through elementwise addition (tf.add).
    #output = tf.add(output, vgg_layer4_out)


    tf.Print(nn_last_layer, [tf.shape(nn_last_layer)])

    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    # FCN-8 - Classification & Loss
	# The final step is to define a loss. That way, we can approach training a FCN just like we would approach training a normal classification CNN.
	# In the case of a FCN, the goal is to assign each pixel to the appropriate class.
	# We already happen to know a great loss function for this setup, cross entropy loss!
	# Remember the output tensor is 4D so we have to reshape it to 2D:
    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    correct_label = tf.reshape(correct_label, (-1,num_classes))
	
	# logits is now a 2D tensor where each row represents a pixel and each column a class.
	# From here we can just use standard cross entropy loss:
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, logits=False, image_shape=False, data_dir=False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    runs_dir = './test_epoch_results'
    plot_dir = './plots'
    val_dir = './val_epoch_results'
    losses = []

    sess.run(tf.global_variables_initializer())

    print("*********************************************************")
    print("Starting training")
    print("*********************************************************")

    for i in range(epochs):
    	epoch = i+1
    	print("epoch {} ...".format(epoch))
    	for image, label in get_batches_fn(batch_size):
    		# Training
    		_, loss = sess.run([train_op, cross_entropy_loss], feed_dict={input_image: image,
    			                                                        correct_label: label,
    			                                                           keep_prob: kp,
    			                                                           learning_rate: lr_rate})
    		print("loss = {:.3f}".format(loss))
    		losses.append(loss)

    	if((epoch % 10) == 0):
            print("#--------------------------------------------------------")
            print("Inference & Validation samples after epoch", epoch)
            print("#--------------------------------------------------------")
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, epoch)
            helper.save_val_samples(val_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, epoch)

    helper.plot_loss(plot_dir, losses, "loss_vs_epoch")

tests.test_train_nn(train_nn)


def run():
    data_dir = './data'
    num_classes = 2
    image_shape = (160, 576)
    # tests.test_for_kitti_dataset(data_dir)


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        runs_dir = './test_output'
        plot_dir = './plots'
        val_dir  = './val_output'

	# TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # TODO: Train NN using the train_nn function
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate, logits=logits, image_shape=image_shape, data_dir=data_dir)
        # TODO: Save inference data using helper.save_inference_samples
        print("*********************************************************")
        print("Final validation & inference samples ")
        print("*********************************************************")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, epochs)
        helper.save_val_samples(val_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, epochs)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
