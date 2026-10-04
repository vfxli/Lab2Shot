"""Lab2Shot's one Maya command: lab2shotUndoStep, a result version imported as one undo step (lab2shot_maya/undo.py).
Plain Python (Maya API 2.0), nothing compiled; loaded by the plugin when it first imports a result."""

import maya.api.OpenMaya as om


def maya_useNewAPI():
    pass


def initializePlugin(plugin):
    from lab2shot_maya import undo

    om.MFnPlugin(plugin, "Lab2Shot", "1.0").registerCommand(undo.COMMAND, undo.Step.create)


def uninitializePlugin(plugin):
    from lab2shot_maya import undo

    om.MFnPlugin(plugin).deregisterCommand(undo.COMMAND)
