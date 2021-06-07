# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
import resnet_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10, cifar100, or signlang.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('ref_log_root', '',
                           'Directory to keep the reference checkpoints trained for facenet. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_bool('gray', True, "To train model with gray scale images")
tf.app.flags.DEFINE_integer('maxsteps', 100000, "Maximum steps to train model")
tf.app.flags.DEFINE_integer('ckptinterval', 1000, "Maximum steps to train model")


def train(hps):
    """Training loop."""
    tf.reset_default_graph()
    images, labels = cifar_input.build_input(
        FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode, FLAGS.gray, hps[1])
    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()

    # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    # sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True))
    saver = tf.train.Saver()
    try:
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.ref_log_root)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.ref_log_root)
        else:
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
    except Exception as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)

    if FLAGS.mode == "freeze":
        tf.train.write_graph(sess.graph_def, FLAGS.log_root, "model.pbtxt")
        print("Saved model.pbtxt at", FLAGS.log_root)
        sys.exit()
    tf.train.start_queue_runners(sess)

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.1

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate * 1.0})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 20000:
                self._lrn_rate = 0.1
            elif train_step < 35000:
                self._lrn_rate = 0.01
            elif train_step < 50000:
                self._lrn_rate = 0.001
            elif train_step < 60000:
                self._lrn_rate = 0.0001
            else:
                self._lrn_rate = 0.00001

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            save_summaries_steps=0,
            save_checkpoint_steps=FLAGS.ckptinterval,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        confusion_matrix = np.zeros((hps[1], hps[1]))
        while not mon_sess.should_stop() and mon_sess.run(model.global_step) < FLAGS.maxsteps:
            _, confusion = mon_sess.run([model.train_op, model.confusion_matrix])
            confusion_matrix = np.add(confusion_matrix, np.array(confusion))
            if mon_sess.run(model.global_step) % FLAGS.ckptinterval == 0 and mon_sess.run(model.global_step) != 0:
                print("Confusion_Matrix :\n {}".format(confusion_matrix.astype(np.int)))
                confusion_matrix = np.zeros((hps[1], hps[1]))


def evaluate(hps):
    """Eval loop."""
    images, labels = cifar_input.build_input(
        FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode, FLAGS.gray, hps[1])
    model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
    model.build_graph()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    # ==============================================================================
    # Store Graph
    # ==============================================================================
    # tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pb", as_text=False)
    # tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pbtx", as_text=True)
    # ==============================================================================

    best_precision = 0.0
    first = True
    while True:
        if first:
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)
                continue
            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
                continue
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            # tf.train.write_graph(sess.graph_def, "./", "motor_eval.pb", as_text=False)
            # tf.train.write_graph(sess.graph_def, "./", "motor_eval.pbtx", as_text=True)
            first = False

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step, logits, in_image, fire1, fire2, fire3, fire4, fire5,
             fire6) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step, model.logits, model._images, model.fire1, model.fire2, model.fire3,
                 model.fire4, model.fire5, model.fire6])
            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        np.set_printoptions(threshold=0.5)

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if FLAGS.mode == 'train':
        batch_size = 128
    elif FLAGS.mode == 'eval':
        batch_size = 100
    if FLAGS.mode == 'freeze':
        batch_size = 1

    if FLAGS.dataset == 'signlang':
        num_classes = 2 + 1  # broken, normal, unknown

    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,  # 2*3*this
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom'  # sgd, mom, adam, rmsprop; mom=ok. adam=fail
                               )  # resNet enable

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
            FLAGS.mode = 'freeze'  # generate inference graph
            train(hps)
        elif FLAGS.mode == 'freeze':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
