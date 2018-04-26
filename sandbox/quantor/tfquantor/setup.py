#!/usr/bin/env python2
# vim:fileencoding=UTF-8:ts=2:sw=2:sta:et:sts=2:ai

from distutils.core import setup

setup(
  name='tfquantor',
  version='1.8.1',
  author='yumaokao',
  author_email='yumao.kao@mediatek.com',
  description='Quantization module for tensorflow with FakeQuant*',
  long_description="Quantization module for tensorflow and tflite with "
                   "FakeQuant* comes from tensorflow/contrib/quantize/ . "
                   "Please also refer to https://arxiv.org/pdf/1712.05877",
  license='LICENSE.txt',
  url='http://www.github.com',

  packages=['tfquantor', 'tfquantor.quantize', 'tfquantor.quantize.extra',
            'tfquantor.eval', 'tfquantor.tools'],
  scripts=['tfquantor/eval/bin/eval_frozen',
           'tfquantor/eval/bin/eval_tflite',
           'tfquantor/eval/bin/quantor_frozen',
           'tfquantor/tools/bin/compare_toco',
           'tfquantor/tools/bin/direct_quantor',
           'tfquantor/tools/bin/visualizer_qt4',
           'tfquantor/tools/bin/visualizer'],
  entry_points={
    'console_scripts':[
      'save_summaries=tfquantor.tools.save_summaries:main',
      'drop_dropouts=tfquantor.tools.drop_dropouts:main',
      'rename_ops=tfquantor.tools.rename_ops:main',
      'npyhisto=tfquantor.tools.npyhisto:main',
    ],
  },
  install_requires=[
    'tensorflow>=1.5.0',
    'urwid',
  ],
)
