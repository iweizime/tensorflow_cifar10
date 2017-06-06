from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import argparse
import math
import time
import sys
import cv2
import os

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

import cifar10
from cifar10_input import IMAGE_SIZE

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', 'saved_cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
                         
tf.app.flags.DEFINE_boolean('extract', False,
                         """Whether to run eval only once.""")

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

def evaluate(images, labels=None):
  with tf.Graph().as_default():

    image_holder = tf.placeholder(tf.uint8, [32, 32, 3])
    resized_image = tf.cast(image_holder, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(image_holder,
                      IMAGE_SIZE, IMAGE_SIZE)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # batch size = 1
    float_image_batch = tf.reshape(float_image, [1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # inference model.
    logits  = cifar10.inference(float_image_batch, False)
    softmax = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

      total = 0
      correct = 0
      statics = {}
      for cls in CLASSES:
          statics[cls] = 0

      for i, image in enumerate(images):
        total += 1
        softmax_out = sess.run(softmax, feed_dict={image_holder: image})
        index = np.argmax(softmax_out)
        if labels:
            if labels[i] == index:
                correct += 1
        statics[CLASSES[index]] += 1

      for cls in CLASSES:
          print('%-12s: %4d/%d'%(cls, statics[cls], total))
      if labels:
          print('%d/%d, %.1f%%'%(correct, total, correct*100/float(total)))




def main(argv=None):  # pylint: disable=unused-argument
  images = []
  labels = []
  isLabel = True
  paths = []
  for image_path in args.images:
    path = Path(image_path)
    if path.is_file():
        paths.append(str(path))
    elif path.is_dir():
        paths = paths + [str(pic) for pic in path.glob('**/*') if pic.is_file()]
  
  for image_path in paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))
    images.append(image)
    if isLabel:
        base_path = os.path.basename(image_path)
        try:
            label = int(base_path[0])
            labels.append(label)
        except ValueError:
            isLabel = False
  if isLabel:
      evaluate(images, labels)
  else:
      evaluate(images)

  # with open('test_batch', 'rb') as fo:
  #     image_label_dict = pickle.load(fo, encoding='bytes')

  # labels = image_label_dict[b'labels']
  # labels = list(labels)
  # data = image_label_dict[b'data']
  # images = []

  # statics = {}
  # for i in range(10):
  #     statics[i] = 0

  # for i, image in enumerate(data):
  #     image = image.reshape((3, 32, 32))
  #     image = image.transpose((1, 2, 0))
  #     if FLAGS.extract:
  #         label = labels[i]
  #         filename = str(label) + '_' + str(statics[label])+'.jpg'
  #         statics[label] += 1
  #         cv2.imwrite('test/'+filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  #     images.append(image)
  # evaluate(images, labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--images', nargs='+', metavar='IMAGE_PATH', dest='images', help="images to be tested")
  args, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
