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
from examples.nee._code.pipeline import *
import matplotlib.pyplot as plt


class NEEDemoApp(EditorApplication):
    def __init__(self):
        super().__init__()
    
    def onInit(self):
        # self.pipeline = NEEPipeline()
        self.pipeline = NEEStratifiedPipeline()
        self.pipeline.setStandardSize(se.ivec3(1024,1024,1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        
        self.scene = se.gfx.Context.load_scene_gltf("examples/nee/_data/scene.gltf")
        sed.EditorBase.bindScene(self.scene)
        self.scene.get().updateTransform()
        self.scene.get().updateGPUScene()
    
    def onUpdate(self):
        self.ctx.device.waitIdle()
        self.scene.get().updateGPUScene()
        
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        self.pipeline.bindScene(self.scene)
        self.pipeline.execute(cmdEncoder)
    
    def onDrawGui(self):
        pass
    
    def onClose(self):
        del self.pipeline
        del self.scene
        super().onClose()


editor = NEEDemoApp()
editor.run()
print("Script End Successfully!")