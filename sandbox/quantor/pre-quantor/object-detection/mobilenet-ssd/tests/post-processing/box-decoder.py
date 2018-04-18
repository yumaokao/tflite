import numpy as np


def _get_bbox_center_size(bbox):
  # [ymin, xmin, ymax, xmax] -> [ycen, xcen, h, w]
  # print(bbox)
  h = bbox[:, 2] - bbox[:, 0]
  w = bbox[:, 3] - bbox[:, 1]
  y = bbox[:, 0] + (h / 2.0)
  x = bbox[:, 1] + (w / 2.0)
  center_size = np.stack([y, x, h, w], axis=1)
  return center_size


def _get_bbox_min_max(bbox):
  # [ycen, xcen, h, w] -> [ymin, xmin, ymax, xmax]
  # print(bbox)
  ymin = bbox[:, 0] - (bbox[:, 2] / 2.0)
  xmin = bbox[:, 1] - (bbox[:, 3] / 2.0)
  ymax = bbox[:, 0] + (bbox[:, 2] / 2.0)
  xmax = bbox[:, 1] + (bbox[:, 3] / 2.0)
  min_max = np.stack([ymin, xmin, ymax, xmax], axis=1)
  return min_max


def main():
  # TODO: argparse
  
  # detection_boxes_all is an intermedium output, which obtained with following patch
  '''
  diff --git a/research/object_detection/meta_architectures/ssd_meta_arch.py b/research/object_detection/meta_architectures/ssd_meta_arch.py
  index ad3b80c..e05712c 100644
  --- a/research/object_detection/meta_architectures/ssd_meta_arch.py
  +++ b/research/object_detection/meta_architectures/ssd_meta_arch.py
  @@ -453,6 +453,8 @@ class SSDMetaArch(model.DetectionModel):
				  preprocessed_images, true_image_shapes),
			  additional_fields=additional_fields)
		 detection_dict = {
  +          # YMK DEBUG
  +          'detection_boxes_all': detection_boxes,
			 fields.DetectionResultFields.detection_boxes: nmsed_boxes,
			 fields.DetectionResultFields.detection_scores: nmsed_scores,
			 fields.DetectionResultFields.detection_classes: nmsed_classes,
  '''

  # Make sure preprocessed_inputs is [N, 300, 300, 3] with range (-1.0, 1.0)
  original_image = np.load('npys/original_image.npy')
  preprocessed_inputs = np.load('npys/preprocessed_inputs.npy')

  detection_boxes_all = np.load('npys/detection_boxes_all.npy')
  anchors = np.load('npys/anchors.npy')
  box_encodings = np.load('npys/box_encodings.npy')
  class_predictions_with_background = np.load('npys/class_predictions_with_background.npy')

  # decode with Faster RCNN box coder
  """Faster RCNN box coder.

  Faster RCNN box coder follows the coding schema described below:
	ty = (y - ya) / ha
	tx = (x - xa) / wa
	th = log(h / ha)
	tw = log(w / wa)
	where x, y, w, h denote the box's center coordinates, width and height
	respectively. Similarly, xa, ya, wa, ha denote the anchor's center
	coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
	center, width and height respectively.

	See http://arxiv.org/abs/1506.01497 for details.

	def _decode(self, rel_codes, anchors):
	  ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

	  ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
	  if self._scale_factors:
		ty /= self._scale_factors[0]
		tx /= self._scale_factors[1]
		th /= self._scale_factors[2]
		tw /= self._scale_factors[3]
	  w = tf.exp(tw) * wa
	  h = tf.exp(th) * ha
	  ycenter = ty * ha + ycenter_a
	  xcenter = tx * wa + xcenter_a
	  ymin = ycenter - h / 2.
	  xmin = xcenter - w / 2.
	  ymax = ycenter + h / 2.
	  xmax = xcenter + w / 2.
	  return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
  """

  # anchors is in [ymin, xmin, ymax, xmax]
  #  => anchors_center_size [ycen_a, xcen_a, ha, wa]
  anchors_center_size = _get_bbox_center_size(anchors)

  # TODO: scale_factor from argparse
  scale_factors = np.array([10.0, 10.0, 5.0, 5.0])
  # box_encodings is in [ycen, xcen, height, width]
  box_encodings = np.squeeze(box_encodings)
  box_encodings = box_encodings / scale_factors

  w = np.exp(box_encodings[:, 3]) * anchors_center_size[:, 3]
  h = np.exp(box_encodings[:, 2]) * anchors_center_size[:, 2]
  xcen = box_encodings[:, 1] * anchors_center_size[:, 3] + anchors_center_size[:, 1]
  ycen = box_encodings[:, 0] * anchors_center_size[:, 2] + anchors_center_size[:, 0]
  box_decoded_cen_size = np.stack([ycen, xcen, h, w], axis=1)
  box_decoded = _get_bbox_min_max(box_decoded_cen_size)

  # box_decoded should == detection_boxes_all 
  error_diff = box_decoded - np.squeeze(detection_boxes_all)
  print('error diff sum of box decoder: {}'.format(error_diff.sum()))
  np.save('box_decoded.npy', box_decoded)


if __name__ == '__main__':
  main()
