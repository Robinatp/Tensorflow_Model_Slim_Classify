from preprocessing import inception_preprocessing
import tensorflow as tf

from tensorflow.contrib import slim
from datasets import flowers
from nets import squeezenet
import math

# This might take a few minutes.
train_dir = 'tmp/tfslim_model/'
print('Will save model to %s' % train_dir)
flowers_data_dir="tmp/flower_photos/"


def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
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
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
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



def train():

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = flowers.get_split('train', flowers_data_dir)
        images, _, labels = load_batch(dataset)
      
        # Create the model:
        logits ,_= squeezenet.squeezenet(images, num_classes=dataset.num_classes, is_training=True)
     
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()
    
        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total Loss', total_loss)
      
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
    
        # Run the training:
        final_loss = slim.learning.train(
          train_op,
          logdir=train_dir,
          number_of_steps=100, # For speed, we just do 1 epoch
          save_interval_secs=600,
          save_summaries_secs=6000,
          log_every_n_steps =1,)
      
        print('Finished training. Final batch loss %d' % final_loss)
    
def eval():
    # This might take a few minutes.
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.DEBUG)
        
        dataset = flowers.get_split('validation', flowers_data_dir)
        images, _, labels = load_batch(dataset)
        
        logits,_ = squeezenet.squeezenet(images, num_classes=dataset.num_classes, is_training=False)
        predictions = tf.argmax(logits, 1)
        
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'eval/Recall@5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
        })
    
        num_batches = math.ceil(dataset.num_samples / 32)
        print('Running evaluation Loop...')
        checkpoint_path = tf.train.latest_checkpoint(train_dir)
        metric_values = slim.evaluation.evaluate_once(
            master='',
            checkpoint_path=checkpoint_path,
            logdir=train_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            final_op=names_to_values.values())
    
        names_to_values = dict(zip(names_to_values.keys(), metric_values))
        for name in names_to_values:
            print('%s: %f' % (name, names_to_values[name]))
        
if __name__ == "__main__":
    
#     train()
    eval()            
    
