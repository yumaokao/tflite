#!/usr/bin/env python2
# vim:fileencoding=UTF-8:ts=2:sw=2:sta:et:sts=2:ai

from distutils.core import setup

setup(
  name='tfquantor',
  version='0.1.0',
  author='yumaokao',
  author_email='yumao.kao@mediatek.com',
  description='Quantization module for tensorflow with FakeQuant*',
  long_description="Quantization module for tensorflow and tflite with FakeQuant*, "
                   "comes from tensorflow/contrib/quantize/ and "
                   "https://arxiv.org/pdf/1712.05877",
  license='LICENSE.txt',
  url='http://www.github.com',

  packages=['tfquantor'],
  install_requires=[
      'tensorflow>=1.5.0rc0',
  ],
)
