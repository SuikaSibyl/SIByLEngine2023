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
from examples.rasterdiff._code.pipeline import *

class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
    
    def onInit(self):
        # create and bind the scene
        # self.scene = se.gfx.Context.load_scene_gltf("D:/Art/Objects/bunny/scene-new.gltf")
        # self.scene = se.gfx.Context.load_scene_gltf("examples/rasterdiff/_data/sphere.gltf")
        self.scene = se.gfx.Context.load_scene_gltf("examples/rasterdiff/_data/plane.gltf")
        sed.EditorBase.bindScene(self.scene)
        self.scene.get().updateTransform()
        self.scene.get().updateGPUScene()
        # get the size of the position buffer
        pos_buffer_size = self.scene.get().getGPUScene().\
            getPositionBuffer().get().getDevice().size()
        index_buffer_size = self.scene.get().getGPUScene().\
            getIndexBuffer().get().getDevice().size()
        # create and bind the pipeline
        self.pipeline = DiffRasterPipeline(pos_buffer_size, index_buffer_size)
        self.pipeline.setStandardSize(se.ivec3(800,800,1))
        self.pipeline.build()
        sed.EditorBase.bindPipeline(self.pipeline.pipeline())
        # # only for produce the gt
        # self.frame_index = 0
    
    def onUpdate(self):
        # update the scene
        self.ctx.device.waitIdle()
        self.scene.get().updateGPUScene()
        self.ctx.device.waitIdle()
        # update camera index
        camera_index = self.pipeline.setting.camera_idx.get()
        # camera_index = camera_index + 1
        # if camera_index >= 14:
        #     camera_index = 0
        self.pipeline.setting.camera_idx.set(camera_index)
        # only for produce the gt
        # if self.frame_index > 0 and self.frame_index <= 16:
        #     se.gfx.Context.captureImage(
        #         self.pipeline.getOutput(),
        #         f"S:/SIByL2024/Sandbox/examples/rasterdiff/_images/{self.frame_index-1}.exr")
    
    def onCommandRecord(self, cmdEncoder: se.rhi.CommandEncoder):
        # if self.frame_index <= 16:
        #     sed.EditorBase.getViewportWidget().setCameraIndex(self.frame_index)
        #     self.frame_index += 1
        self.pipeline.bindScene(self.scene)
        self.pipeline.execute(cmdEncoder)
    
    def onDrawGui(self):
        pass
    
    def onClose(self):
        del self.pipeline
        del self.scene
        super().onClose()

# debug.reexecute_if_unbuffered()
editor = EditorApp()
editor.run()
print("Script End Successfully!")