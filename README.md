# IFCB Jupyter Viewer

## Introduction

This Python package contains a graphical tool for viewing automatically classified IFCB images.
It can be used to inspect and evaluate the accuracy of the classifications, as well as
create new labeled image collections with the help of the classifications.

The tool is meant to be imported and used inside a Jupyter Notebook, as it relies on ipywidgets for the graphical elements.
It is built with certain assumptions about how the classifications are provided (i.e., class probabilities and thresholds),
since it is custom built for IFCB data processing at SYKE.

## Installation

This package along with its ![requirements](requirements.txt) can be installed with: `pip install .`

## Usage

The example Jupyter Notebook `demo/viewer.ipynb` demonstrates how the viewer is imported, initialized and used.
To find out what other usage options are available, see the inline documentation for `JupyterViewer` and its `open` method.
