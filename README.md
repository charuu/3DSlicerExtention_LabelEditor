# README #
# 3DSlicerExtention_LabelEditor

### What is this repository for? ###

* A customized 3D Slicer extension using a trained deep learning model to assist in automated segmentation of new images while allowing for manual corrections using an editor
* Version 1.1

### How do I get set up? ###

* For using the plugin offline it can be placed under the ../bin/Python folder under Slicer's home installation
* It requires installation of tensorflow, tensorflow-ops and scikit-image in 3D slicer environment 
* Commands for installing libraries in python-interactor - 
  * pip_install('tensorflow')
  * pip_install('tensorflow-ops')
  * pip_install('scikit-image')
* Create a folder structure under a base directory with folder names - image, mask and model
* Image and mask folder contains the types of teeth to be segmented. Each teeth type folder contain image and label files in NIfTI format
* Model folder contains the trained model, for example - 'model.h5'

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner: Charu Jain
