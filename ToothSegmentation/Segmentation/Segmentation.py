import os

import ctk
import numpy as np
import qt
import slicer
import tensorflow as tf
import vtk as vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from skimage.transform import  resize
from skimage import exposure

from tensorflow.python.framework import ops
from tensorflow.python.keras.losses import Loss
from tensorflow.python.ops import gen_math_ops
#
# Segmentation
#
class Dice(Loss):
  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = gen_math_ops.cast(y_true, y_pred.dtype)
    y_true = tf.keras.backend.clip(y_true, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    numerator = (2. * intersection + 1)
    denominator = (tf.keras.backend.sum(tf.keras.backend.square(y_true), -1) + tf.keras.backend.sum(
      tf.keras.backend.square(y_pred), -1) + 1)
    return 1 - (numerator / denominator)


class Segmentation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Segmentation"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Segmentation">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.



#
# SegmentationWidget
#

class SegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Load image"
    self.layout.addWidget(parametersCollapsibleButton)

    deepLearningmodelCollapsibleButton = ctk.ctkCollapsibleButton()
    deepLearningmodelCollapsibleButton.text = "Load mask"
    self.layout.addWidget(deepLearningmodelCollapsibleButton)

    segmentEditorCollapsibleButton = ctk.ctkCollapsibleButton()
    segmentEditorCollapsibleButton.text = "Segment Editor"
    self.layout.addWidget(segmentEditorCollapsibleButton)

    # Layout within the dummy collapsible button
    deepLearningFormLayout = qt.QFormLayout(deepLearningmodelCollapsibleButton)
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
    segmentEditorFormLayout = qt.QFormLayout(segmentEditorCollapsibleButton)


    self.segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    self.segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorFormLayout.addWidget(self.segmentEditorWidget)

    self.clearButton = qt.QPushButton("Clear all")
    self.clearButton.toolTip = "Clear."
    self.clearButton.enabled = True
    parametersFormLayout.addRow(self.clearButton)
    
    self.moduleDir = os.path.dirname(slicer.modules.segmentations.path)
    self.saveDirectoryFilePathSelector = ctk.ctkDirectoryButton()
    self.saveDirectoryFilePath = os.path.join(self.moduleDir, os.pardir)
    self.saveDirectoryFilePathSelector.directory = self.saveDirectoryFilePath
    segmentEditorFormLayout.addRow(self.saveDirectoryFilePathSelector)

    self.saveButton = qt.QPushButton("Save")
    self.saveButton.toolTip = "Run the algorithm."
    self.saveButton.enabled = True
    segmentEditorFormLayout.addRow(self.saveButton)

    self.loadButton = qt.QPushButton("Load existing mask")
    self.loadButton.toolTip = "Run the algorithm."
    self.loadButton.enabled = True
    deepLearningFormLayout.addRow(self.loadButton)

    self.predictButton = qt.QPushButton("Predict new mask")
    self.predictButton.toolTip = "Run the algorithm."
    self.predictButton.enabled = True
    deepLearningFormLayout.addRow(self.predictButton)

    self.modelDirectoryFilePathSelector = ctk.ctkDirectoryButton()
    self.modelDirectoryFilePath = os.path.join(self.moduleDir,os.pardir)
    self.modelDirectoryFilePathSelector.directory  = self.modelDirectoryFilePath
    parametersFormLayout.addRow(self.modelDirectoryFilePathSelector)

    self.objectTypeSelector = qt.QComboBox()
    self.objectTypeSelector.addItems(["Select tooth type"])
    objects = os.listdir(self.modelDirectoryFilePath)
    objects = [x for x in objects if not '.' in x]
    self.objectTypeSelector.addItems(objects)
    self.objectType = "Select tooth type"
    parametersFormLayout.addRow(self.objectTypeSelector)

    self.objectSelector = qt.QComboBox()
    self.objectSelector.addItems(["Select patient data"])
    self.objectName = "Select patient data"
    parametersFormLayout.addRow(self.objectSelector)

    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.predictButton.connect('clicked(bool)', self.onPredictButton)
    self.clearButton.connect('clicked(bool)', self.onClearButton)
    self.saveButton.connect('clicked(bool)', self.onSaveButton)
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.loadButton.connect('clicked(bool)', self.onLoadButton)

    self.modelDirectoryFilePathSelector.connect('directorySelected(QString)', self.onDirectorySelected)
    self.saveDirectoryFilePathSelector.connect('directorySelected(QString)', self.onSaveDirectorySelected)
    self.objectSelector.connect('currentIndexChanged(int)', self.onObjectSelected)
    self.objectTypeSelector.connect('currentIndexChanged(int)', self.onObjectTypeSelected)
    # Add vertical spacer
    self.layout.addStretch(1)


  def cleanup(self):
    pass

  def onDirectorySelected(self):
    self.modelDirectoryFilePath = self.modelDirectoryFilePathSelector.directory
    currentItems = self.objectTypeSelector.count
    for i in range(currentItems,-1,-1):
      self.objectTypeSelector.removeItem(i)
    networks = os.listdir(self.modelDirectoryFilePath)
    networks = [x for x in networks if not '.' in x and x[0] != '_']
    networks = ["Select tooth type"] + networks
    self.objectTypeSelector.addItems(networks)


  def onObjectTypeSelected(self):
    #self.modelDirectoryFilePath = self.modelDirectoryFilePathSelector.directory
    self.objectType = self.objectTypeSelector.currentText
    currentItems = self.objectSelector.count
    for i in range(currentItems, -1, -1):
      self.objectSelector.removeItem(i)
    self.objectSelector.addItem("Select patient data")
    if self.objectType != "Select tooth type":
      objects = os.listdir(os.path.join(self.modelDirectoryFilePath, self.objectType))
      objects = [x for x in objects]
      self.objectSelector.addItems(objects)

  def onObjectSelected(self):
    self.objectName = self.objectSelector.currentText

  def onSelect(self):
    # Start monitoring
    self.applyButton.enabled = self.objectSelector.currentNode()

  def onSaveDirectorySelected(self):
    self.saveDirectoryFilePath = self.saveDirectoryFilePathSelector.directory


  def onSaveButton(self):
    print("Save")
    outputPath=self.saveDirectoryFilePathSelector.directory
    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    segmentationNode = self.segmentEditorWidget.segmentationNode()
    labelmapVolumeNode.SetOrigin(self.image.GetOrigin())
    labelmapVolumeNode.SetSpacing(self.image.GetSpacing())
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentationNode, labelmapVolumeNode)
    filepath = os.path.join(self.saveDirectoryFilePathSelector.directory,self.objectTypeSelector.currentText)
    slicer.util.saveNode(labelmapVolumeNode, filepath)
  #  modelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode')
    filepath2 = (self.saveDirectoryFilePathSelector.directory)

    slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(filepath2,segmentationNode,"STL")
  #  slicer.util.saveNode(modelNode, filepath2)

  def onLoadButton(self):
    #self.objectTypeSelector.currentText,
    path=(os.path.join(self.modelDirectoryFilePathSelector.directory,self.objectTypeSelector.currentText,self.objectSelector.currentText)).replace('image','mask')
    labelmapVolumeNode =slicer.util.loadLabelVolume(path)
    masterVolumeNode = self.image

    segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)

    segmentationNode.CreateClosedSurfaceRepresentation()
    slicer.mrmlScene.AddNode(segmentEditorNode)
    segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
    self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    self.segmentEditorWidget.setSegmentationNode(segmentationNode)
    self.segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)

  def onPredictButton(self):
    path= os.path.join(self.modelDirectoryFilePathSelector.directory, 'model/')
    model1 ='iteration5.h5'
    self.loadKerasModel(os.path.join(path,model1))

  def onClearButton(self):
    slicer.mrmlScene.RemoveNode(self.image)
    slicer.mrmlScene.RemoveNode(self.labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(self.segmentationNode)

  def onApplyButton(self):
    path = os.path.join(self.modelDirectoryFilePathSelector.directory, self.objectTypeSelector.currentText,self.objectSelector.currentText)
    self.image = slicer.util.loadVolume(path)


  def loadKerasModel(self, modelFilePath):
    """
    Tries to load Keras model for classifiation
    :param modelFilePath: full path to saved model file
    :return: True on success, False on error
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    #Load model
    self.model = tf.keras.models.load_model(modelFilePath,custom_objects={'Dice': Dice})
    #pre-processing

    x = np.swapaxes((slicer.util.arrayFromVolume(self.image)),0,2)
    #x=np.array([x[2],x[1],x[0]])
    print(np.min(x),' ',np.max(x))
    x[x < -1000] = -1000
    x[x > 4000] = 4000
    x=((x + 1000) / 5000)
    input_array = exposure.equalize_hist(x)
    print(input_array.shape)
    resize_image =  resize(input_array, (40, 40, 40), anti_aliasing=True)
    img_np = np.rollaxis(np.array(np.array([np.array([np.array(resize_image)])])), 0, 5)

    #predict
    pred1 = ((self.model.predict(img_np)).astype(float))
    new_pred = np.swapaxes(np.round(resize(pred1[0][:, :, :, 0], input_array.shape, anti_aliasing=True)),0,2)

    self.labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
    volumeNode.CreateDefaultDisplayNodes()
    slicer.util.updateVolumeFromArray(volumeNode, new_pred)
    volumeNode.SetOrigin(self.image.GetOrigin())
    volumeNode.SetSpacing(self.image.GetSpacing())
    slicer.vtkSlicerVolumesLogic().CreateLabelVolumeFromVolume(slicer.mrmlScene, self.labelmapVolumeNode, volumeNode)

    masterVolumeNode = self.image
    slicer.app.processEvents()

    self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()

    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(self.labelmapVolumeNode,
                                                                          self.segmentationNode)
    slicer.mrmlScene.AddNode(segmentEditorNode)
    #self.segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
    self.segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(masterVolumeNode)
    self.segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    self.segmentEditorWidget.setSegmentationNode(self.segmentationNode)
    self.segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)
    self.segmentationNode.CreateClosedSurfaceRepresentation()

class SegmentationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """

  def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """


# SegTest
#

class SegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
      """ Do whatever is needed to reset the state - typically a scene clear will be enough.
      """
      slicer.mrmlScene.Clear(0)

    def runTest(self):
      """Run as few or as many tests as needed here.
      """
      self.setUp()
      self.test_LineIntensityProfile1()

    def test_LineIntensityProfile1(self):
      """ Ideally you should have several levels of tests.  At the lowest level
      tests sould exercise the functionality of the logic with different inputs
      (both valid and invalid).  At higher levels your tests should emulate the
      way the user would interact with your code and confirm that it still works
      the way you intended.
      One of the most important features of the tests is that it should alert other
      developers when their changes will have an impact on the behavior of your
      module.  For example, if a developer removes a feature that you depend on,
      your test should break so they know that the feature is needed.
      """

      self.delayDisplay("Starting the test")

      #labelmapVolumeNode = slicer.util.loadLabelVolume('/export/skulls/projects/teeth/data/u-net-data/train/mask/canine/1.2.80.nii')
      #moduleWidget = slicer.modules.SegmentationWidget
     # moduleWidget.inputSelector.setCurrentNode(labelmapVolumeNode)

      #volumeNode2 = slicer.util.loadVolume('/export/skulls/projects/teeth/data/u-net-data/train/image/canine/1.2.80.nii')
     # moduleWidget2 = slicer.modules.SegmentationWidget
     # moduleWidget2.inputSelector2.setCurrentNode(volumeNode2)

      #labelmapVolumeNode = slicer.util.getNode('label')

      logic = SegmentationLogic()
    #  self.assertTrue(logic.hasImageData(volumeNode))
      self.delayDisplay('Test passed!')
