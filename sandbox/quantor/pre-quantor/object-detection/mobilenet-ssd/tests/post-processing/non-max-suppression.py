import numpy as np


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def _filter_background_and_threshold(scores, score_threshold=1e-8, bgclass=0):
  '''
    return filtered and sorted score list of tuple
      in [(index, argmax, max), ...]
  '''
  # with argmax and max (1917, 38) -> (1917, 2)
  argmaxs = np.argmax(scores, axis=1)
  maxs = np.max(scores, axis=1)

  filtered_scores = []
  for i in range(len(argmaxs)):
    if argmaxs[i] == 0:
      continue
    if maxs[i] < score_threshold:
      continue
    # print('index {}, argmaxs {} value {}'.format(i, argmaxs[i], maxs[i]))
    filtered_scores.append((i, argmaxs[i], maxs[i]))

  sorted_scores = sorted(filtered_scores, key=lambda x: x[2], reverse=True)
  return sorted_scores


def _ious(box_a, box_b):
  '''
    box_a = (ind, cat, score, bbox)
    iou = iou_area / bbox_a_area
  '''
  # not in same cat, iou = 0.0
  if box_a[1] != box_b[1]:
    return 0.0
  bbox_a = box_a[3] 
  bbox_b = box_b[3] 
  bbox_a_area = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])

  ymin = max(bbox_a[0], bbox_b[0])
  xmin = max(bbox_a[1], bbox_b[1])
  ymax = min(bbox_a[2], bbox_b[2])
  xmax = min(bbox_a[3], bbox_b[3])
  iou_area = (ymax - ymin) * (xmax - xmin)
  iou = iou_area / bbox_a_area
  # print('[ymin, xmin, ymax, xmax] = [{}, {}, {}, {}]'.format(ymin, xmin, ymax, xmax))
  # print('iou = {}'.format(iou))
  return iou


def _non_max_suppression(scores, bboxes, original_scale_ratio, iou_threshold=0.6):
  orig_ratio = np.array([original_scale_ratio[0], original_scale_ratio[1],
                         original_scale_ratio[0], original_scale_ratio[1]])
  nms_bboxes = []
  if len(scores) == 0:
    return nms_bboxes

  sorted_scores = sorted(scores, key=lambda x: x[2], reverse=True)
  for (ind, cat, score) in sorted_scores:
    bbox = bboxes[ind] * orig_ratio
    # print('ind {} cat {} score {}'.format(ind, cat, score))
    # print('bbox {}'.format(bbox))

    cur = (ind, cat, score, bbox)

    # add to nms_bboxes if no cat in nms_bboxes
    if len(list(filter(lambda n: n[1] == cat, nms_bboxes))) == 0:
      nms_bboxes.append(cur)
      continue

    # add to nms_bboxes if iou < iou_threshold
    _nms_ious = list(map(lambda n: _ious(n , cur), nms_bboxes))
    _nms_ious = list(filter(lambda n: n > iou_threshold, _nms_ious))
    if len(_nms_ious) == 0:
      nms_bboxes.append(cur)
      continue

  # print(nms_bboxes)
  return nms_bboxes


def main():
  # TODO: argparse
  
  # Make sure preprocessed_inputs is [N, 300, 300, 3] with range (-1.0, 1.0)
  original_image = np.load('npys/original_image.npy')
  preprocessed_inputs = np.load('npys/preprocessed_inputs.npy')

  # uses detection_boxes_all first
  # detection_boxes_all = np.load('npys/detection_boxes_all.npy')
  detection_boxes_all = np.load('box_decoded.npy')

  # detection_scores, detection_classes detection_boxes
  detection_scores = np.load('npys/detection_scores.npy')
  detection_classes = np.load('npys/detection_classes.npy')
  detection_boxes = np.load('npys/detection_boxes.npy')

  # class_predictions_with_background
  class_predictions_with_background = np.load('npys/class_predictions_with_background.npy')

  # TODO: only support score_converter SIGMOID
  scores = _sigmoid(class_predictions_with_background)

  # filter out background => argmax == 0
  filtered_scores = _filter_background_and_threshold(scores)

  # original_ratio
  original_height = original_image.shape[1]
  original_width = original_image.shape[2]
  wh_min = min(original_height, original_width)
  original_scale_ratio = [original_height, original_width,
                          original_height / wh_min,
                          original_width / wh_min]

  detections = _non_max_suppression(filtered_scores,
                                    np.squeeze(detection_boxes_all),
                                    original_scale_ratio)

  print('detections')
  for d in detections:
    print(d)

  print()
  print('detection_boxes[0:{}]'.format(len(detections)))
  for l in range(len(detections)):
    print('({}, {}, {}, {})'.format(l, detection_classes[l],
                               detection_scores[l], detection_boxes[l]))

  # detection_boxes_all = np.squeeze(detection_boxes_all)
  # import ipdb
  # ipdb.set_trace()
  # print(detection_boxes_all)


if __name__ == '__main__':
  main()
