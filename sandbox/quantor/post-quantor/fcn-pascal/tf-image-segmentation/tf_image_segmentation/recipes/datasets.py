#!/usr/bin/env python
# coding=utf-8
"""
This sets up the location where datasets are stored.
"""
from __future__ import division, print_function, unicode_literals
from sacred import Experiment, Ingredient
from os.path import expanduser

# ============== Ingredient 0: settings =================
s = Ingredient("settings")


@s.config
def cfg1():
    verbose = True


# ============== Ingredient 1: dataset.paths =================
data_paths = Ingredient("dataset.paths", ingredients=[s])


@data_paths.config
def cfg2(settings):
    v = not settings['verbose']
    base = expanduser("~") + "/datasets"