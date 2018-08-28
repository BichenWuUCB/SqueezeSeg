# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016
#-*- coding: utf-8 -*-

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from imdb import kitti
from utils.util import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")

tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")

tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeseg/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")

tf.app.flags.DEFINE_string('net', 'squeezeSeg',
                           """Neural net architecture. """)

tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")

tf.app.flags.DEFINE_integer('summary_step', 50,
                            """Number of steps to save summary.""")

tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")

tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def train():
  """Train SqueezeSeg model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():
    assert FLAGS.net == 'squeezeSeg', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    if FLAGS.net == 'squeezeSeg':
      mc = kitti_squeezeSeg_config()
      # ed: SqueezeSeg을 본격적으로 training 하기 전에 전이학습을 위해 SqueezeNet의 pretrained_model을 불러오는듯
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeSeg(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # ed: Model의 여러 정보를 저장하기 위한 코드
    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')

      # ed: parameter size를 기록하는 코드
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      # ed: output activation이 정확히 뭔지 모르겠지만 그걸 기록해주는 코드
      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))


      # ed: Flop Count를 기록해주는 코드
      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

    f.close()


    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))



    def enqueue(sess, coord):
      with coord.stop_on_exception():
        while not coord.should_stop():
          # ed: 여기가 Input (.npy) 파일들을 처리하는 코드인듯
          # read batch input
          lidar_per_batch, lidar_mask_per_batch, label_per_batch,\
              weight_per_batch = imdb.read_batch()

          feed_dict = {
              model.ph_keep_prob: mc.KEEP_PROB,
              model.ph_lidar_input: lidar_per_batch,
              model.ph_lidar_mask: lidar_mask_per_batch,
              model.ph_label: label_per_batch,
              model.ph_loss_weight: weight_per_batch,
          }

          # ed: placeholder에 데이터를 넣어주는 코드
          #     FIFOQueue라는 함수를 사용해서 여러 input들을 병렬적으로 처리하는듯
          sess.run(model.enqueue_op, feed_dict=feed_dict)


    saver = tf.train.Saver(tf.all_variables())
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    # ed: sess 초기화
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    coord = tf.train.Coordinator()
    enq_threads = []

    for _ in range(mc.NUM_ENQUEUE_THREAD):
      eqth = threading.Thread(target=enqueue, args=[sess, coord])
      eqth.start()
      enq_threads.append(eqth)

    # ed: 특정 시간이상 연산이 초과되면 assertion을 내주기 위한 코드인듯
    run_options = tf.RunOptions(timeout_in_ms=60000)

    try:
      # ed: 학습하는 코드
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        # ed: 50번 마다 실행되고 마지막 step에서 실행되는 제어문
        if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps-1:
          op_list = [
              model.lidar_input, model.lidar_mask, model.label, model.train_op,
              model.loss, model.pred_cls, summary_op
          ]

          # ed: 50번 step과 마지막 step에만 실행되는 코드, 학습이 끝나고 성능을 알아보기 위해 실행하는 듯하다
          #     이런식으로 Queue를 사용해서 일괄적으로 placeholder들에 feeding을 할 수 있는듯하다
          lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
              _, loss_value, pred_cls, summary_str = sess.run(op_list,
                                                              options=run_options)

          # ed: label, pred_cls에 의해서 class가 정해진 곳에 colorize를 해주는 코드
          label_image = visualize_seg(label_per_batch[:6, :, :], mc)
          pred_image = visualize_seg(pred_cls[:6, :, :], mc)

          # ed: IOU를 계산하는 코드
          # Run evaluation on the batch
          ious, _, _, _ = evaluate_iou(
              label_per_batch, pred_cls * np.squeeze(lidar_mask_per_batch),
              mc.NUM_CLASS)


          feed_dict = {}

          # Assume that class-0 is the background class
          for i in range(1, mc.NUM_CLASS):
            feed_dict[model.iou_summary_placeholders[i]] = ious[i]

          iou_summary_list = sess.run(model.iou_summary_ops[1:], feed_dict)


          # ed: 여기서 summary 형식으로 visualize 해주는건 뭘까? ==> tensorboard를 위한 코드
          # Run visualization
          viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]

          viz_summary_list = sess.run(
              viz_op_list, 
              feed_dict={
                  model.depth_image_to_show: lidar_per_batch[:6, :, :, [4]],
                  model.label_to_show: label_image,
                  model.pred_image_to_show: pred_image,
              }
          )

          # Add summaries
          summary_writer.add_summary(summary_str, step)

          for sum_str in iou_summary_list:
            summary_writer.add_summary(sum_str, step)

          for viz_sum in viz_summary_list:
            summary_writer.add_summary(viz_sum, step)

          # force tensorflow to synchronise summaries
          summary_writer.flush()

        else:
          # ed: 실제 학습을 하는 코드
          _, loss_value = sess.run(
              [model.train_op, model.loss], options=run_options)


        # ed: 알고리즘 수행시간 체크
        duration = time.time() - start_time

        # ed: 여러 loss value 중 invalid한 값이 없어야 한다
        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                    class_loss)


        # ed: 10번에 한번씩 print 해주는 코드
        if step % 10 == 0:
          num_images_per_step = mc.BATCH_SIZE
          images_per_sec = num_images_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                        'sec/batch)')

          print (format_str % (datetime.now(), step, loss_value,
                               images_per_sec, sec_per_batch))
          sys.stdout.flush()


        # ed: default=1000 번에 한번씩 model의 가중치를 저장한다
        # Save the model checkpoint periodically.
        if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps-1:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


    except Exception, e:
      coord.request_stop(e)

    finally:
      coord.request_stop()
      sess.run(model.q.close(cancel_pending_enqueues=True))
      # Wait for all the threads to terminate.
      coord.join(enq_threads)



def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)

  tf.gfile.MakeDirs(FLAGS.train_dir)

  train()


if __name__ == '__main__':
  # ed: run main() code
  tf.app.run()
