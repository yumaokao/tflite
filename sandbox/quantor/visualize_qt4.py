import sys
import os
import time
import subprocess
import re
import itertools
import numpy as np
import threading
import inspect
import json
from datetime import datetime
from scipy.stats import gaussian_kde

import tensorflow as tf
from PyQt4.QtCore import Qt, QThread, QString, QStringList, SIGNAL, SLOT
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def logThreadName(instance):
  prev_frame = inspect.currentframe().f_back
  #print('[{}.{}()] executed by {}'.format(instance.__class__.__name__, prev_frame.f_code.co_name, threading.current_thread().name))

#########################
##  Utility functions  ##
#########################

def getHBoxLayout(widget_list, stretch_at_front=False, stretch_at_end=False, margin=None):
  layout = QHBoxLayout()
  for widget in widget_list:
    layout.addWidget(widget)
  if margin is not None:
    layout.setMargin(margin)

  if stretch_at_front:
    layout.insertStretch(0, 1)
  if stretch_at_end:
    layout.insertStretch(-1, 1)

  widget = QWidget()
  widget.setLayout(layout)
  return widget

def getVBoxLayout(widget_list, stretch_at_front=False, stretch_at_end=False, margin=None):
  layout = QVBoxLayout()
  for widget in widget_list:
    layout.addWidget(widget)
  if margin is not None:
    layout.setMargin(margin)

  if stretch_at_front:
    layout.insertStretch(0, 1)
  if stretch_at_end:
    layout.insertStretch(-1, 1)

  widget = QWidget()
  widget.setLayout(layout)
  return widget

def createButtonWithText(text, set_fixed_width=False, set_min_width=False):
  btn = QPushButton(text)
  width = btn.fontMetrics().boundingRect(btn.text()).width() + 16
  if set_fixed_width:
    btn.setFixedWidth(width)
  if set_min_width:
    btn.setMinimumWidth(width)
  return btn

def saveFileDialogAndReturn(parent, filter_str=None):
  dialog = QFileDialog(parent, 'Save as', os.getcwd())
  dialog.setAcceptMode(QFileDialog.AcceptSave)
  if filter_str is not None:
    dialog.setNameFilter(QString(filter_str))
  if dialog.exec_():
    return dialog.selectedFiles()[0]

def openFileDialogAndReturn(parent, filter_list=None):
  dialog = QFileDialog(parent, 'Open file', os.getcwd())
  if filter_list is not None:
    dialog.setNameFilters(QStringList(filter_list))
  if dialog.exec_():
    return dialog.selectedFiles()[0]

def openFileDialogAndDisplay(parent, display_widget, need_dir=False, filter_list=None):
  dialog = QFileDialog(parent, 'Open file', os.getcwd())
  if need_dir:
    dialog.setFileMode(QFileDialog.Directory)
  if filter_list is not None:
    dialog.setNameFilters(QStringList(filter_list))
  if dialog.exec_():
    display_widget.setText(dialog.selectedFiles()[0])

######################
##  Member classes  ##
######################

class Executor(QThread):
  def __init__(self, parent=None):
    super(Executor, self).__init__(parent)

  def setParameter(self, param):
    self.param = param

  def run(self):
    logThreadName(self)

    def processFrozenPb(model_fn, input_fn, input_node, output_node):
      graph_def = tf.GraphDef()
      input_data = np.load(input_fn)
      with tf.gfile.GFile(model_fn, "rb") as f:
        graph_def.ParseFromString(f.read())
      with tf.Graph().as_default() as cur_graph:
        tf.import_graph_def(graph_def, name='')
      with tf.Session(graph=cur_graph) as sess:
        graph = sess.graph
        x = graph.get_tensor_by_name('{}:0'.format(input_node))
        y = graph.get_tensor_by_name('{}:0'.format(output_node))
        output_data = sess.run(y, feed_dict={x: input_data})
      return output_data

    def processTfliteModel(model_fn, input_fn, input_node, output_node):
      def getTfliteNodeIndexAndQuantizationInfo(model_fn, node_name):
        cmd = [self.param['tensorflow_dir'] + '/bazel-bin/tensorflow/contrib/lite/utils/dump_tflite', model_fn]
        out = subprocess.check_output(cmd)
        for line in out.splitlines():
          if 'name ' + node_name in line:
            result = re.search('(?P<index>[0-9]+): name ' + node_name + '.*quantization \((?P<scale>[0-9\.]+) (?P<zero>[0-9\.]+)\)', line)
            return int(result.group('index')), float(result.group('scale')), int(result.group('zero'))
        raise ValueError('Quantization of the output node is not embedded inside the TFLite model')

      output_index, scale, zero_point = getTfliteNodeIndexAndQuantizationInfo(model_fn, output_node)
      tmp_output_fn = input_fn + '.out.tmp'
      cmd = [self.param['tensorflow_dir'] + '/bazel-bin/tensorflow/contrib/lite/utils/run_tflite',
          '--tflite_file={}'.format(model_fn),
          '--batch_xs={}'.format(input_fn),
          '--batch_ys={}'.format(tmp_output_fn),
          '--output_tensor_idx={}'.format(output_index),
          '--inference_type=' + self.param['type']]
      subprocess.check_output(cmd)
      output_data = np.load(tmp_output_fn)
      subprocess.check_output(['rm', tmp_output_fn])
      return (output_data.astype(float) - zero_point) * scale

    model_ext = os.path.splitext(self.param['model'])[1]
    if model_ext == '.pb':
      result = processFrozenPb(self.param['model'], self.param['data'], self.param['input_node'], self.param['output_node'])
      self.emit(SIGNAL('Result'), result)
    elif model_ext == '.tflite' or model_ext == '.lite':
      result = processTfliteModel(self.param['model'], self.param['data'], self.param['input_node'], self.param['output_node'])
      self.emit(SIGNAL('Result'), result)


class ModelExecutor:
  def __del__(self):
    self.executor.terminate()
    self.executor.wait()

  def __init__(self, name, errorReporter, updateCallback, fixed_width=None, fixed_height=None):

    self.errorReporter = errorReporter
    self.updateCallback = updateCallback

    # Worker thread
    self.instance = QApplication.instance()
    self.executor = Executor(parent=self.instance)
    self.executor.started.connect(self.executorStarted)
    self.executor.finished.connect(self.executorFinished)
    self.instance.connect(self.executor, SIGNAL('Result'), self.setNumpyResult)
    self.tensorflow_dir = None
    self.numpy_result = None

    self.group_box = QGroupBox(name)
    if fixed_width is not None:
      self.group_box.setFixedWidth(fixed_width)
    if fixed_height is not None:
      self.group_box.setFixedHeight(fixed_height)

    # Create Qt Widgets
    self.file = QLabel('Model file')
    self.file_edit = QLineEdit()
    self.file_edit.setReadOnly(True)
    self.file_btn = createButtonWithText('...', set_fixed_width=True)
    self.file_btn.clicked.connect(lambda : openFileDialogAndDisplay(self.file_btn, self.file_edit, filter_list=['TensorFlow model (*.pb)', 'TensorFlow Lite model (*.tflite *.lite)']))
    self.data = QLabel('Data file')
    self.data_edit = QLineEdit()
    self.data_edit.setReadOnly(True)
    self.data_btn = createButtonWithText('...', set_fixed_width=True)
    self.data_btn.clicked.connect(lambda : openFileDialogAndDisplay(self.data_btn, self.data_edit,filter_list=['NumPy array file (*.npy)']))
    self.type_btn_group = QButtonGroup()
    self.type_btn_group.setExclusive(True)
    self.float_btn = QRadioButton('float')
    self.float_btn.setChecked(True)
    self.uint8_btn = QRadioButton('uint8')
    self.type_btn_group.addButton(self.uint8_btn)
    self.type_btn_group.addButton(self.float_btn)
    self.input_edit = QLineEdit()
    self.input_edit.setPlaceholderText('Input node name')
    self.output_edit = QLineEdit()
    self.output_edit.setPlaceholderText('Output node name')
    self.run_btn = createButtonWithText('Run', set_min_width=True)
    self.run_btn.clicked.connect(self.run)
    self.save_btn = createButtonWithText('Save', set_min_width=True)
    self.save_btn.clicked.connect(lambda : self.saveNumpyResult(self.save_btn))

    # Assign widget placement
    self.layout = QGridLayout()
    self.layout.addWidget(self.file, 0, 0)
    self.layout.addWidget(getHBoxLayout([self.file_edit, self.file_btn], margin=0), 1, 0)
    self.layout.addWidget(self.data, 2, 0)
    self.layout.addWidget(getHBoxLayout([self.data_edit, self.data_btn], margin=0), 3, 0)
    self.layout.addWidget(getHBoxLayout([self.float_btn, self.uint8_btn], margin=0), 4, 0)
    self.layout.addWidget(self.input_edit, 5, 0)
    self.layout.addWidget(self.output_edit, 6, 0)
    self.layout.addWidget(getHBoxLayout([self.run_btn, self.save_btn], margin=0), 7, 0)
    self.layout.setRowStretch(8, 1)
    self.group_box.setLayout(self.layout)

  def executorStarted(self):
    logThreadName(self)
    self.run_btn.clearFocus()
    self.run_btn.setEnabled(False)
    self.save_btn.setEnabled(False)

  def executorFinished(self):
    logThreadName(self)
    self.run_btn.setEnabled(True)
    self.save_btn.setEnabled(True)

  def setNumpyResult(self, numpy_result):
    logThreadName(self)
    self.numpy_result = numpy_result
    self.updateCallback()

  def getNumpyResult(self):
    return self.numpy_result

  def saveNumpyResult(self, parent):
    if self.numpy_result is not None:
      fn = saveFileDialogAndReturn(parent, filter_str='NumPy array file (*.npy)')
      np.save(str(fn), self.numpy_result)
    else:
      self.errorReporter('No result to save!', timeout=1000)

  def setTensorflowDir(self, dirname):
    self.tensorflow_dir = dirname

  def getGroupBoxWidget(self):
    return self.group_box

  def importConfig(self, config):
    self.file_edit.setText(config['model'])
    self.data_edit.setText(config['data'])
    self.input_edit.setText(config['input_node'])
    self.output_edit.setText(config['output_node'])
    if config['type'] == 'float':
      self.float_btn.setChecked(True)
    else:
      self.uint8_btn.setChecked(True)

  def exportConfig(self):
    config = {}
    config['model'] = str(self.file_edit.text());
    config['data'] = str(self.data_edit.text());
    config['input_node'] = str(self.input_edit.text());
    config['output_node'] = str(self.output_edit.text());
    config['type'] = 'float' if self.float_btn.isChecked() else 'uint8'
    return config

  def run(self):

    if self.tensorflow_dir is None:
      self.errorReporter('Please specify the tensorflow directory.', timeout=1000)
      return

    worker_param = {}
    worker_param['model'] = str(self.file_edit.text());
    worker_param['data'] = str(self.data_edit.text());
    worker_param['input_node'] = str(self.input_edit.text());
    worker_param['output_node'] = str(self.output_edit.text());
    worker_param['type'] = 'float' if self.float_btn.isChecked() else 'uint8'
    worker_param['tensorflow_dir'] = self.tensorflow_dir

    if (worker_param['model'] == '') or (worker_param['data'] == '') or (worker_param['input_node'] == '') or (worker_param['output_node'] == ''):
      self.errorReporter('Please fill in all the information.', timeout=1000)
      return

    self.executor.setParameter(worker_param)
    self.executor.start()


class Visualizer(QMainWindow):
  def __init__(self):
    super(Visualizer, self).__init__()
    self.instance = QApplication.instance()
    self.initUI()

  def initUI(self):

    def updateTensorflowDir(dirname):
      self.model_a.setTensorflowDir(str(dirname))
      self.model_b.setTensorflowDir(str(dirname))

    self.model_a = ModelExecutor('Model A', fixed_height=250, errorReporter=self.errorReporter, updateCallback=lambda : self.updateCallback(0))
    self.model_b = ModelExecutor('Model B', fixed_height=250, errorReporter=self.errorReporter, updateCallback=lambda : self.updateCallback(1))
    self.figure = Figure()
    self.canvas = FigureCanvas(self.figure)
    self.canvas_toolbar = NavigationToolbar(self.canvas, self)
    self.status_bar = QStatusBar()
    self.tf_dir = QLabel('Path to tensorflow directory')
    self.tf_dir_edit = QLineEdit()
    self.tf_dir_edit.setReadOnly(True)
    self.tf_dir_edit.textChanged.connect(updateTensorflowDir)
    self.tf_dir_btn = createButtonWithText('...', set_fixed_width=True)
    self.tf_dir_btn.clicked.connect(lambda : openFileDialogAndDisplay(self.tf_dir_btn, self.tf_dir_edit, need_dir=True))
    self.display_mode = QLabel('Display mode')
    self.display_mode_box = QComboBox()
    self.display_mode_box.addItem('Histogram(2D)')
    self.display_mode_box.addItem('Histogram(3D)')
    self.draw_btn = createButtonWithText('Draw', set_min_width=True)
    self.draw_btn.clicked.connect(lambda x : self.drawFigure())

    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(getVBoxLayout([self.canvas_toolbar, self.canvas]));
    splitter.addWidget(getVBoxLayout(
                          [self.tf_dir, getHBoxLayout([self.tf_dir_edit, self.tf_dir_btn], margin=0),
                          self.model_a.getGroupBoxWidget(), self.model_b.getGroupBoxWidget(),
                          self.display_mode, self.display_mode_box, self.draw_btn], stretch_at_end=True));

    self.setCentralWidget(splitter)
    self.setStatusBar(self.status_bar)

    # Menu bar
    self.import_action = QAction('Import', self)
    self.export_action = QAction('Export', self)
    self.import_action.triggered.connect(self.importConfig)
    self.export_action.triggered.connect(self.exportConfig)
    self.file_menu = self.menuBar().addMenu('File')
    self.file_menu.addAction(self.import_action)
    self.file_menu.addAction(self.export_action)

    self.canvas.setFocus()
    self.show()

  def importConfig(self):
    fn = openFileDialogAndReturn(self, filter_list=['JSON file (*.json)'])
    with open(str(fn), 'r') as f:
      config = json.load(f)
    self.tf_dir_edit.setText(config['tensorflow_dir'])
    self.display_mode_box.setCurrentIndex(self.display_mode_box.findText(config['display_mode']))
    self.model_a.importConfig(config['model_a'])
    self.model_b.importConfig(config['model_b'])

  def exportConfig(self):
    config = {}
    config['tensorflow_dir'] = str(self.tf_dir_edit.text())
    config['display_mode'] = str(self.display_mode_box.currentText())
    config['model_a'] = self.model_a.exportConfig()
    config['model_b'] = self.model_b.exportConfig()
    fn = saveFileDialogAndReturn(self, filter_str='JSON file (*.json)')
    with open(str(fn), 'w') as f:
      json.dump(config, f)

  def drawFigure(self):
    logThreadName(self)
    result_a = self.model_a.getNumpyResult()
    result_b = self.model_b.getNumpyResult()
    data_list = []
    fn_list = []
    if result_a is not None:
      data = result_a.flatten()
      data_list.append(data)
      fn_list.append('Model A')
    if result_b is not None:
      data = result_b.flatten()
      data_list.append(data)
      fn_list.append('Model B')

    if len(data_list) == 0:
      self.errorReporter('No result to draw!')
      return

    def getGaussianKdeFromData(data, xs):
      tmp = data.flatten()
      density = gaussian_kde(tmp)
      density.covariance_factor = lambda : .05
      density._compute_covariance()
      return xs, density(xs)

    def getHistogramFromData(data, hist_bins):
      hist, bins = np.histogram(data, bins=hist_bins, density=True)
      bar_xs = (bins[:-1] + bins[1:])/2
      bar_width = np.mean(bins[1:] - bins[:-1]) * 0.8
      return bar_xs, bar_width, hist

    minmax_list = [(np.min(data), np.max(data)) for data in data_list]
    min_range = min([(max_val - min_val) for min_val, max_val in minmax_list])
    all_min = min([min_val for min_val, _ in minmax_list])
    all_max = min([max_val for _, max_val in minmax_list])
    hist_step = min_range / 50
    hist_bins = np.arange(all_min, all_max + hist_step, hist_step)
    hist_list = [getHistogramFromData(data, hist_bins) for data in data_list]
    kde_step = min_range / 100
    kde_xs = np.arange(all_min, all_max + kde_step, kde_step)
    kde_list = [getGaussianKdeFromData(data, kde_xs) for data in data_list]

    self.figure.clf()
    if self.display_mode_box.currentText() == 'Histogram(2D)':
      ax_list = self.figure.subplots(len(fn_list), sharex='col', sharey='col')
      if len(fn_list) == 1:
        ax_list = [ax_list] # make it iterable

      for fn, ax, hist, kde in zip(fn_list, ax_list, hist_list, kde_list):
        bar_x, bar_width, bar_y = hist
        kde_x, kde_y = kde
        ax.bar(bar_x, bar_y, width=bar_width, edgecolor='gray', facecolor='green', alpha=0.5)
        ax.plot(kde_x, kde_y, color='black', linewidth=2)
        ax.title.set_text(fn)
        ax.set_ylabel('Density')
        if fn == fn_list[-1]:
          ax.set_xlabel('Value')

    elif self.display_mode_box.currentText() == 'Histogram(3D)':
      zpos_iter = itertools.count()
      color_iter = itertools.cycle(['red', 'orange', 'yellow', 'green', 'blue', 'megenta', 'purple', 'black'])
      ax = self.figure.add_subplot(111, projection='3d')
      for fn, hist, kde in zip(fn_list, hist_list, kde_list):
        z = next(zpos_iter)
        bar_x, bar_width, bar_y = hist
        kde_x, kde_y = kde
        ax.bar(bar_x, bar_y, zs=z, zdir='y', width=bar_width, edgecolor='gray', facecolor=next(color_iter), alpha=0.9, label=fn)
        ax.plot(kde_x, [z] * len(kde_x), kde_y, color='black', linewidth=2)

      ax.set_yticklabels([])
      ax.set_xlabel('Value')
      ax.set_zlabel('Density')
      ax.set_ylim(-1, 2)
      ax.legend(loc='best')

    self.canvas.draw()

  def updateCallback(self, update_index):
    # TODO: Use light widget to notify users that the result is ready
    #self.drawFigure()
    pass

  def errorReporter(self, msg, timeout=0):
    time_msg = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' > '
    time_msg += msg
    self.status_bar.showMessage(time_msg, timeout)
    return


def main(_):
  app = QApplication(sys.argv)
  app.setStyle('Cleanlooks')
  ex = Visualizer()
  sys.exit(app.exec_())

if __name__ == '__main__':
  tf.app.run()
