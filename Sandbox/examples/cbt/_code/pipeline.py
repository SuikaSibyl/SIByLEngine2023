import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed

class CBTIndirectParamPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "cbt/_shader/cbt-indirect.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInputOutput("CBTree")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Indirect")\
            .isBuffer().withSize(32 * 4).withUsages(
                int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        cbtree = rdrDat.getBuffer("CBTree")
        indir = rdrDat.getBuffer("Indirect")

        self.updateBindings(rdrCtx, [
            ["cbt_heap", cbtree.get().getBindingResource()],
            ["indirect_buffer", indir.get().getBindingResource()],
        ])
        
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(1, 1, 1)
        encoder.end()


class CBTTestPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "cbt/_shader/cbt-test.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.index = 0
        self.position_x = se.Float32(0.5)
        self.position_y = se.Float32(0.5)
        self.position_z = se.Float32(0.5)
        self.radius = se.Float32(0.3)
        self.stop_split = se.Int32(0)
    
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInput("CBTree")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("DebugBuffer")\
            .isBuffer().withSize(32 * 4 * 8).withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("Indirect")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.INDIRECT))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.INDIRECT_COMMAND_READ_BIT))
                .addStage(se.rhi.PipelineStageBit.DRAW_INDIRECT_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        cbtree = rdrDat.getBuffer("CBTree")
        dbgbuf = rdrDat.getBuffer("DebugBuffer")
        indir = rdrDat.getBuffer("Indirect")

        self.updateBindings(rdrCtx, [
            ["cbt_heap", cbtree.get().getBindingResource()],
            ["debug_buffer", dbgbuf.get().getBindingResource()],
        ])

        class PushConstant(ctypes.Structure):
          _fields_ = [("position_x", ctypes.c_float),
                    ("position_y", ctypes.c_float),
                    ("position_z", ctypes.c_float),
                    ("radius", ctypes.c_float),
                    ("stop_split", ctypes.c_int32)]
        pConst = PushConstant(
            position_x = self.position_x.get(),
            position_y = self.position_y.get(),
            position_z = self.position_z.get(),
            radius = self.radius.get(),
            stop_split = self.stop_split.get()
        )
        self.index += 1
        
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroupsIndirect(indir.get().getDevice(), 16)
        encoder.end()

    def renderUI(self):
        sed.ImGui.DragFloat("Position X", self.position_x, 0.01, 0.0, 1.0)
        sed.ImGui.DragFloat("Position Y", self.position_y, 0.01, 0.0, 1.0)
        sed.ImGui.DragFloat("Position Z", self.position_z, 0.01, 0.0, 1.0)
        sed.ImGui.DragFloat("Radius", self.radius, 0.01, 0.0, 1.0)
        sed.ImGui.DragInt("Stop Split", self.stop_split, 1, 0, 15)


class CBTSpatialTreeGraph(core.rdg.Graph):
    def __init__(self):
        core.rdg.Graph.__init__(self)
        max_depth = 15
        self.create_pass = se.passes.cbt.CreateCBTPass(max_depth, 3)
        self.indirect_pass = CBTIndirectParamPass()
        self.addPass(self.create_pass, "Create Pass")
        self.addPass(self.indirect_pass, "Indirect Pass")
        self.addEdge("Create Pass", "CBTree", "Indirect Pass", "CBTree")
        
        self.test_pass = CBTTestPass()
        self.addPass(self.test_pass, "Test Pass")
        self.addEdge("Indirect Pass", "CBTree", "Test Pass", "CBTree")
        self.addEdge("Indirect Pass", "Indirect", "Test Pass", "Indirect")

        self.prefix_0_pass = se.passes.cbt.SumReductionFusedPass(max_depth)
        self.prefix_1_pass = se.passes.cbt.SumReductionOneLayerPass(max_depth)
        self.addPass(self.prefix_0_pass, "Prefix 0 Pass")
        self.addPass(self.prefix_1_pass, "Prefix 1 Pass")
        self.addEdge("Test Pass", "CBTree", "Prefix 0 Pass", "CBTree")
        self.addEdge("Prefix 0 Pass", "CBTree", "Prefix 1 Pass", "CBTree")


        self.editor_init_pass = core.passes.EditorInitPass()
        self.sbt_visualize_pass = se.passes.cbt.CBTSpatialTreeVisualizePass(0)
        self.addPass(self.editor_init_pass, "Editor Init")
        self.addPass(self.sbt_visualize_pass, "SBT Visualize")
        self.addEdge("Indirect Pass", "Indirect", "SBT Visualize", "Indirect")
        self.addEdge("Editor Init", "Depth", "SBT Visualize", "Depth")
        self.addEdge("Prefix 1 Pass", "CBTree", "SBT Visualize", "CBTree")

        self.markOutput("SBT Visualize", "Color")


class CBTSpatialTreePipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.graph = CBTSpatialTreeGraph()
        self.setGraph(self.graph)
    
    def onUpdate(self, ctx:SEContext):
        pass
    
    def renderUI(self):
        pass
