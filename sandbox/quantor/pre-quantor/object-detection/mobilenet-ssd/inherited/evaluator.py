# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""

import logging
import os
import subprocess
import numpy as np
import tensorflow as tf

# from object_detection import eval_util
from inherited import eval_util
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.metrics import coco_evaluation
from object_detection.utils import object_detection_evaluation

# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    'open_images_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
}

EVAL_DEFAULT_METRIC = 'pascal_voc_detection_metrics'


def _extract_predictions_and_losses(model,
                                    create_input_dict_fn,
                                    ignore_groundtruth=False):
  """Constructs tensorflow detection graph and returns output tensors.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    prediction_groundtruth_dict: A dictionary with postprocessed tensors (keyed
      by standard_fields.DetectionResultsFields) and optional groundtruth
      tensors (keyed by standard_fields.InputDataFields).
    losses_dict: A dictionary containing detection losses. This is empty when
      ignore_groundtruth is true.
  """
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
  preprocessed_image, true_image_shapes = model.preprocess(
      tf.to_float(original_image))
  prediction_dict = model.predict(preprocessed_image, true_image_shapes)
  detections = model.postprocess(prediction_dict, true_image_shapes)

  groundtruth = None
  losses_dict = {}
  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    if fields.InputDataFields.groundtruth_group_of in input_dict:
      groundtruth[fields.InputDataFields.groundtruth_group_of] = (
          input_dict[fields.InputDataFields.groundtruth_group_of])
    if fields.DetectionResultFields.detection_masks in detections:
      groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
          input_dict[fields.InputDataFields.groundtruth_instance_masks])
    label_id_offset = 1
    model.provide_groundtruth(
        [input_dict[fields.InputDataFields.groundtruth_boxes]],
        [tf.one_hot(input_dict[fields.InputDataFields.groundtruth_classes]
                    - label_id_offset, depth=model.num_classes)])
    losses_dict.update(model.loss(prediction_dict, true_image_shapes))

  result_dict = eval_util.result_dict_for_single_example(
      original_image,
      input_dict[fields.InputDataFields.source_id],
      detections,
      groundtruth,
      class_agnostic=(
          fields.DetectionResultFields.detection_classes not in detections),
      scale_to_absolute=True)
  return result_dict, losses_dict


def get_evaluators(eval_config, categories):
  """Returns the evaluator class according to eval_config, valid for categories.

  Args:
    eval_config: evaluation configurations.
    categories: a list of categories to evaluate.
  Returns:
    An list of instances of DetectionEvaluator.

  Raises:
    ValueError: if metric is not in the metric class dictionary.
  """
  eval_metric_fn_keys = eval_config.metrics_set
  if not eval_metric_fn_keys:
    eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
  evaluators_list = []
  for eval_metric_fn_key in eval_metric_fn_keys:
    if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
      raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
    evaluators_list.append(
        EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories=categories))
  return evaluators_list


def evaluate(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, graph_hook_fn=None, evaluator_list=None):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
    graph_hook_fn: Optional function that is called after the training graph is
      completely built. This is helpful to perform additional changes to the
      training graph such as optimizing batchnorm. The function should modify
      the default graph.
    evaluator_list: Optional list of instances of DetectionEvaluator. If not
      given, this list of metrics is created according to the eval_config.

  Returns:
    metrics: A dictionary containing metric names and values from the latest
      run.
  """

  model = create_model_fn()

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  tensor_dict, losses_dict = _extract_predictions_and_losses(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      ignore_groundtruth=eval_config.ignore_groundtruth)

  def _process_batch(tensor_dict, sess, batch_index, counters,
                     losses_dict=None):
    """Evaluates tensors in tensor_dict, losses_dict and visualizes examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      batch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.
      losses_dict: Optional dictonary of scalar loss tensors.

    Returns:
      result_dict: a dictionary of numpy arrays
      result_losses_dict: a dictionary of scalar losses. This is empty if input
        losses_dict is None.
    """
    try:
      if not losses_dict:
        losses_dict = {}
      result_dict, result_losses_dict = sess.run([tensor_dict, losses_dict])
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}
    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    if batch_index < eval_config.num_visualizations:
      tag = 'image-{}'.format(batch_index)
      eval_util.visualize_detection_results(
          result_dict,
          tag,
          global_step,
          categories=categories,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=eval_config.visualize_groundtruth_boxes,
          groundtruth_box_visualization_color=eval_config.
          groundtruth_box_visualization_color,
          min_score_thresh=eval_config.min_score_threshold,
          max_num_predictions=eval_config.max_num_boxes_to_visualize,
          skip_scores=eval_config.skip_scores,
          skip_labels=eval_config.skip_labels,
          keep_image_id_for_visualization_export=eval_config.
          keep_image_id_for_visualization_export)
    return result_dict, result_losses_dict

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)

  if graph_hook_fn: graph_hook_fn()

  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

  if not evaluator_list:
    evaluator_list = get_evaluators(eval_config, categories)

  metrics = eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      summary_dir=eval_dir,
      evaluators=evaluator_list,
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(1 if eval_config.ignore_groundtruth else
                                 eval_config.max_evals
                                 if eval_config.max_evals else None),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      losses_dict=losses_dict)

  return metrics


# _extract_anchros_and_losses
def _extract_anchros_and_losses(model,
                                create_input_dict_fn,
                                quantize=False,
                                ignore_groundtruth=False):
  """Constructs tensorflow detection graph and returns output tensors.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    prediction_groundtruth_dict: A dictionary with postprocessed tensors (keyed
      by standard_fields.DetectionResultsFields) and optional groundtruth
      tensors (keyed by standard_fields.InputDataFields).
    losses_dict: A dictionary containing detection losses. This is empty when
      ignore_groundtruth is true.
  """
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
  preprocessed_image, true_image_shapes = model.preprocess(
      tf.to_float(original_image))
  prediction_dict = model.predict(preprocessed_image, true_image_shapes)
  detections = model.postprocess(prediction_dict, true_image_shapes)

  if quantize:
    from tensorflow.contrib.quantize import experimental_create_eval_graph
    experimental_create_eval_graph()
    # g = tf.get_default_graph()
    # print(g.get_operations())

  groundtruth = None
  losses_dict = {}
  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    if fields.InputDataFields.groundtruth_group_of in input_dict:
      groundtruth[fields.InputDataFields.groundtruth_group_of] = (
          input_dict[fields.InputDataFields.groundtruth_group_of])
    if fields.DetectionResultFields.detection_masks in detections:
      groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
          input_dict[fields.InputDataFields.groundtruth_instance_masks])
    label_id_offset = 1
    model.provide_groundtruth(
        [input_dict[fields.InputDataFields.groundtruth_boxes]],
        [tf.one_hot(input_dict[fields.InputDataFields.groundtruth_classes]
                    - label_id_offset, depth=model.num_classes)])
    losses_dict.update(model.loss(prediction_dict, true_image_shapes))

  result_dict = eval_util.result_dict_for_single_example(
      original_image,
      input_dict[fields.InputDataFields.source_id],
      detections,
      groundtruth,
      class_agnostic=(
          fields.DetectionResultFields.detection_classes not in detections),
      scale_to_absolute=True)

  # model.preprocess
  result_dict['preprocessed_image'] = preprocessed_image
  result_dict['true_image_shapes'] = true_image_shapes

  # model.predict
  result_dict['class_predictions_with_background'] = (
      prediction_dict['class_predictions_with_background'])
  result_dict['feature_maps'] = prediction_dict['feature_maps']
  result_dict['preprocessed_inputs'] = prediction_dict['preprocessed_inputs']
  result_dict['box_encodings'] = prediction_dict['box_encodings']
  result_dict['anchors'] = prediction_dict['anchors']

  # model.detections DEBUG ONLY
  result_dict['detection_boxes_all'] = detections['detection_boxes_all']

  # print('YMK in _extract_anchros_and_losses')
  # import ipdb
  # ipdb.set_trace()
  return result_dict, losses_dict


# evaluate_with_anchors
def evaluate_with_anchors(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir, graph_hook_fn=None, evaluator_list=None,
             evaluate_with_run_tflite=False, quantize=False, tflite_outputs=None,
             tensorflow_dir=''):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.
    graph_hook_fn: Optional function that is called after the training graph is
      completely built. This is helpful to perform additional changes to the
      training graph such as optimizing batchnorm. The function should modify
      the default graph.
    evaluator_list: Optional list of instances of DetectionEvaluator. If not
      given, this list of metrics is created according to the eval_config.

  Returns:
    metrics: A dictionary containing metric names and values from the latest
      run.
  """

  model = create_model_fn()

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  tensor_dict, losses_dict = _extract_anchros_and_losses(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      quantize=quantize,
      ignore_groundtruth=eval_config.ignore_groundtruth)

  def _prepare_run_tflite_commands(eval_dir, tflite_model, inference_type):
	return [tensorflow_dir + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
			'--tflite_file={}'.format(tflite_model),
			'--batch_xs={}'.format(os.path.join(eval_dir, 'batch_xs.npz')),
			'--batch_ys={}'.format(os.path.join(eval_dir, 'output_ys.npz')),
			'--use_npz={}'.format('true'),
			'--inference_type={}'.format(inference_type)]

  def _process_batch_steps(tensor_dict, sess, batch_index, counters,
                           losses_dict=None):
    # first step: model.preprocess,
    #   with some fields keep in tensor_dict from input_fn
    preprocess_tensor_dict = {}
    preprocess_keys = ['preprocessed_image', 'true_image_shapes']
    preprocess_keys.extend(['original_image', 'key',
                            'groundtruth_is_crowd', 'groundtruth_is_crowd',
                            'groundtruth_group_of', 'groundtruth_classes',
                            'groundtruth_boxes', 'groundtruth_difficult'])
    for k in preprocess_keys:
      preprocess_tensor_dict[k] = tensor_dict[k]
    preprocess_result_dict = sess.run(preprocess_tensor_dict)

    feed_dict = {}
    # second step: model.predict
    for k in preprocess_keys:
      feed_dict[preprocess_tensor_dict[k]] = preprocess_result_dict[k]

    predict_tensor_dict = {}
    predict_keys = ['class_predictions_with_background', 'feature_maps',
                     'preprocessed_inputs', 'box_encodings', 'anchors']
    for k in predict_keys:
      predict_tensor_dict[k] = tensor_dict[k]
    predict_result_dict = sess.run(predict_tensor_dict,
            feed_dict=feed_dict)
    if evaluate_with_run_tflite:
      if quantize:
        # save quantized [-1.0, 1.0] preprocessed_inputs to npz
        #   real_value = (quantized_input_value - mean_value) / std_value
        #   => quantized_input_value = real_value * std_value + mean_value
        #   => q = uint8(r * 127.0 + 128.0)
        run_tflite_dir = os.path.join(eval_dir, 'run_tflite')
        batch_xs_fn = os.path.join(run_tflite_dir, 'batch_xs.npz')
        inputs_r =  predict_result_dict['preprocessed_inputs']
        inputs_q = inputs_r * 127.0 + 128.0
        inputs_q = inputs_q.astype('uint8')
        kwargs = {'Preprocessor/sub': inputs_q}
        np.savez(batch_xs_fn, **kwargs)

        # get outputs with run_tflite
        cmds = _prepare_run_tflite_commands(run_tflite_dir,
			os.path.join(run_tflite_dir, 'uint8_model.lite'), 'uint8')
        subprocess.check_output(cmds)

        # read outputs where
        ys = np.load(os.path.join(run_tflite_dir, 'output_ys.npz'))
        for output in tflite_outputs:
          qv = ys[output[1]]
          fv = qv.astype('float32')
          fv = (fv - output[3]) * output[2]
          predict_result_dict[output[0]] = fv
        '''
        q_class = ys['concat_1']
        f_class = q_class.astype('float32')
        f_class = (f_class - 153) * 0.374646
        predict_result_dict['class_predictions_with_background'] = f_class
        q_box = ys['Squeeze']
        f_box = q_box.astype('float32')
        f_box = (f_box- 154) * 0.0692892
        predict_result_dict['box_encodings'] = f_box
        '''
      else:
        # save preprocessed_inputs to npz
        run_tflite_dir = os.path.join(eval_dir, 'run_tflite')
        batch_xs_fn = os.path.join(run_tflite_dir, 'batch_xs.npz')
        kwargs = {'Preprocessor/sub': predict_result_dict['preprocessed_inputs']}
        np.savez(batch_xs_fn, **kwargs)

        # get outputs with run_tflite
        cmds = _prepare_run_tflite_commands(run_tflite_dir,
			os.path.join(run_tflite_dir, 'float_model.lite'), 'float')
        subprocess.check_output(cmds)

        # read outputs where
        #   'class_predictions_with_background' => 'concat_1'
        #   'box_encodings' => 'Squeeze'
        ys = np.load(os.path.join(run_tflite_dir, 'output_ys.npz'))
        for output in tflite_outputs:
          predict_result_dict[output[0]] = ys[output[1]]
        '''
        predict_result_dict['class_predictions_with_background'] = ys['concat_1']
        predict_result_dict['box_encodings'] = ys['Squeeze']
        '''

    # third step: model.postprocess
    # predict_feed_dict = {}
    for k in predict_keys:
      if type(predict_tensor_dict[k]) == list:
        # print(type(predict_tensor_dict[k]))
        for i, a in enumerate(predict_tensor_dict[k]):
          # print(a)
          # print(predict_result_dict[k][i].shape)
          feed_dict[a] = predict_result_dict[k][i]
      else:
        # print(predict_tensor_dict[k])
        feed_dict[predict_tensor_dict[k]] = predict_result_dict[k]

    # just get all tensor_dict with feed_dict={prediction_dict}
    try:
      result_dict, result_losses_dict = sess.run([tensor_dict, {}],
                                                 feed_dict=feed_dict)
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}

    '''
    try:
     if not losses_dict:
       losses_dict = {}
     result_dict, result_losses_dict = sess.run([tensor_dict, losses_dict])
     counters['success'] += 1
    except tf.errors.InvalidArgumentError:
     logging.info('Skipping image')
     counters['skipped'] += 1
     return {}
    '''

    # print(result_dict.keys())
    # print('YMK original_image.shape {}'.format(result_dict['original_image'].shape))
    # print('YMK anchors[0] {}'.format(result_dict['anchors'][0]))
    # save to npys
    # import numpy as np
    # for k in result_dict.keys():
    #   if k == 'feature_maps':
    #     continue
    #   np.save('tests/post-processing/npys/{}.npy'.format(k), result_dict[k])
    # import ipdb
    # ipdb.set_trace()

    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    if batch_index < eval_config.num_visualizations:
      tag = 'image-{}'.format(batch_index)
      eval_util.visualize_detection_results(
          result_dict,
          tag,
          global_step,
          categories=categories,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=eval_config.visualize_groundtruth_boxes,
          groundtruth_box_visualization_color=eval_config.
          groundtruth_box_visualization_color,
          min_score_thresh=eval_config.min_score_threshold,
          max_num_predictions=eval_config.max_num_boxes_to_visualize,
          skip_scores=eval_config.skip_scores,
          skip_labels=eval_config.skip_labels,
          keep_image_id_for_visualization_export=eval_config.
          keep_image_id_for_visualization_export)
    return result_dict, result_losses_dict

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)

  if graph_hook_fn: graph_hook_fn()

  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)

  if not evaluator_list:
    evaluator_list = get_evaluators(eval_config, categories)

      # batch_processor=_process_batch,
      # losses_dict=losses_dict)
  metrics = eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      summary_dir=eval_dir,
      evaluators=evaluator_list,
      batch_processor=_process_batch_steps,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(1 if eval_config.ignore_groundtruth else
                                 eval_config.max_evals
                                 if eval_config.max_evals else None),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''),
      losses_dict=None)

  return metrics
