from datasets import cifar10
from datasets import flowers
from datasets import imagenet
from datasets import mnist
from datasets import mydata
from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.contrib import slim

def load_batch(dataset, batch_size=12, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = vgg_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels




flowers_data_dir = "/home/robin/Dataset/flowers"
mnist_data_dir = "/home/robin/Dataset/mnist"
cifar10_data_dir = "/home/robin/Dataset/cifar10"
imagenet_data_dir = "/home/robin/Dataset/imaget/output_tfrecord"

if __name__ == "__main__":
    with tf.Graph().as_default(): 
#         dataset = flowers.get_split('train', flowers_data_dir) #load_batch(dataset,height=224, width=224)
#         dataset = mnist.get_split("train",mnist_data_dir) #load_batch(dataset,height=28, width=28)
#         dataset = cifar10.get_split("train",cifar10_data_dir) #load_batch(dataset,height=32, width=32)
        dataset = imagenet.get_split('validation', imagenet_data_dir) #load_batch(dataset,height=224, width=224)


        
        batch_image, batch_raw_image, batch_labels =load_batch(dataset,height=224, width=224,is_training=True)
        one_hot_batch_labels = slim.one_hot_encoding(batch_labels, dataset.num_classes)
        print(batch_image, batch_raw_image,batch_labels)
        
        
        data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset, common_queue_capacity=32, common_queue_min=1)
        image, label = data_provider.get(['image', 'label'])
        one_hot_labels = slim.one_hot_encoding(label, dataset.num_classes)
  
                        
        with tf.Session() as sess:    
            with slim.queues.QueueRunners(sess):
                for i in range(10):
                    np_image,np_raw_image, np_labels,one_hot_labels = sess.run([batch_image, batch_raw_image, batch_labels, one_hot_batch_labels])
                    #print(np_image[0])
                    _,height, width, _ = np_image.shape
                    class_name = dataset.labels_to_names[np_labels[0]]
                                                                                                                            
#                     plt.figure()
#                     plt.imshow(np_image[0])
#                     plt.title('%s, %d x %d' % (name, height, width))
#                     plt.axis('off')
#                     plt.show()


                    print(class_name, np_labels[0], one_hot_labels[0])
                    cv2.imshow('process:',np_image[0]/255)
                    cv2.imshow("raw:",np_raw_image[0]/255)
   
                    cv2.waitKey(0)
                    
                    print("process:", np_image[0])
                    print("--------------")
                    print("raw", np_raw_image[0])
                    
                    
                    
                    
