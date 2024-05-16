# Append the path of the parent directory to the system path
# to import the modules from the parent directory
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(ROOT_DIR)
# import se modules
from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed
from examples.cbt._code.pipeline import *


class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
    
    def onInit(self):
        self.pipeline = CBTSpatialTreePipeline()
        self.pipeline.setStandardSize(se.ivec3(800,800,1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        
        self.scene = se.gfx.Context.create_scene("New Scene")
        sed.EditorBase.bindScene(self.scene)
        self.scene.get().updateTransform()
        self.scene.get().updateGPUScene()
    
    def onUpdate(self):
        self.ctx.device.waitIdle()
        self.scene.get().updateGPUScene()
        self.pipeline.onUpdate(self.ctx)
    
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        self.pipeline.bindScene(self.scene)
        self.pipeline.execute(cmdEncoder)
    
    def onDrawGui(self):
        pass
    
    def onClose(self):
        del self.pipeline
        del self.scene
        super().onClose()


editor = EditorApp()
editor.run()
print("Script End Successfully!")