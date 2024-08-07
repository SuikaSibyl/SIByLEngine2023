from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed

class NEEPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/direct.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())

        self.pos_x = se.Float32(0.0)
        self.pos_y = se.Float32(0.0)
        self.pos_z = se.Float32(0.0)
        self.rotate_x = se.Float32(0.0)
        self.rotate_y = se.Float32(0.0)
        self.rotate_z = se.Float32(0.0)
        self.width = se.Float32(10.0)
        self.height = se.Float32(10.0)
        self.render_mode = se.Int32(0)

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("PMFIdeal")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("PMFActual")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color = rdrDat.getTexture("Color")
        pmf_ideal = rdrDat.getTexture("PMFIdeal")
        pmf_actual = rdrDat.getTexture("PMFActual")
        scene = rdrDat.getScene()
        
        self.updateBindings(rdrCtx, [
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            ["GPUScene_position", scene.get().getGPUScene().bindingResourcePosition()],
            ["GPUScene_tlas", scene.get().getGPUScene().bindingResourceTLAS()],
            ["u_image", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
            ["u_pmf_ideal", core.rhi.BindingResource(pmf_ideal.get().getUAV(0,0,1))],
            ["u_pmf_actual", core.rhi.BindingResource(pmf_actual.get().getUAV(0,0,1))],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [
            ("position_x", ctypes.c_float),
            ("position_y", ctypes.c_float),
            ("position_z", ctypes.c_float),
            ("random_seed", ctypes.c_int),
            ("rotate_x", ctypes.c_float),
            ("rotate_y", ctypes.c_float),
            ("rotate_z", ctypes.c_float),
            ("camera_index", ctypes.c_int),
            ("width", ctypes.c_float),
            ("height", ctypes.c_float),
            ("render_mode", ctypes.c_int),
        ]
        pConst = PushConstant(
            position_x = self.pos_x.get(),
            position_y = self.pos_y.get(),
            position_z = self.pos_z.get(),
            random_seed=np.random.randint(0, 1000000),
            rotate_x = self.rotate_x.get(),
            rotate_y = self.rotate_y.get(),
            rotate_z = self.rotate_z.get(),
            camera_index=scene.get().getEditorActiveCameraIndex(),
            width = self.width.get(),
            height = self.height.get(),
            render_mode = self.render_mode.get(),)
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()

    def renderUI(self):
        sed.ImGui.DragFloat("Position X", self.pos_x, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Position Y", self.pos_y, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Position Z", self.pos_z, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Rotate X", self.rotate_x, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Rotate Y", self.rotate_y, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Rotate Z", self.rotate_z, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Width", self.width, 0.01, -1000, 1000)
        sed.ImGui.DragFloat("Height", self.height, 0.01, -1000, 1000)
        sed.ImGui.Combo("Render Mode", self.render_mode, [
            "Cosine Hemisphere Sample", 
            "Uniform Area Sample",
            "Uniform Solid Angle Sample",
            "Bilinear Solid Angle Sample",
        ])

class NEEStratifiedPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/stratified.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.render_mode = se.Int32(0)
    
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("GuidingInfo-1")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("GuidingInfo-2")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color = rdrDat.getTexture("Color")
        info1 = rdrDat.getTexture("GuidingInfo-1")
        info2 = rdrDat.getTexture("GuidingInfo-2")
        scene = rdrDat.getScene()
        
        self.updateBindings(rdrCtx, [
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            ["GPUScene_position", scene.get().getGPUScene().bindingResourcePosition()],
            ["GPUScene_tlas", scene.get().getGPUScene().bindingResourceTLAS()],
            ["u_image", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
            # ["u_guiding_info_1", core.rhi.BindingResource(info1.get().getUAV(0,0,1))],
            # ["u_guiding_info_2", core.rhi.BindingResource(info2.get().getUAV(0,0,1))],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [
            ("random_seed", ctypes.c_int),
            ("camera_index", ctypes.c_int),
            ("render_mode", ctypes.c_int),
        ]
        pConst = PushConstant(
            random_seed=np.random.randint(0, 1000000),
            camera_index=scene.get().getEditorActiveCameraIndex(),
            render_mode = self.render_mode.get(),)
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()

    def renderUI(self):
        sed.ImGui.Combo("Render Mode", self.render_mode, [
            "Cosine Hemisphere Sample", 
            "Uniform Area Sample",
            "Uniform Solid Angle Sample",
            "Bilinear Solid Angle Sample",
        ])
        
class ReducePass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/reduce.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInputOutput("PMFTexture")\
            .isTexture().withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInternal("SumTexture")\
            .isTexture().withSize(se.ivec3(8, 8, 1))\
            .withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Summation")\
            .isBuffer().withSize(4 * 16)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        pmf_tex = rdrDat.getTexture("PMFTexture")
        sum_tex = rdrDat.getTexture("SumTexture")
        sum_buffer = rdrDat.getBuffer("Summation")
        
        self.updateBindings(rdrCtx, [
            ["pmf_texture", core.rhi.BindingResource(pmf_tex.get().getUAV(0,0,1))],
            ["sum_texture", core.rhi.BindingResource(sum_tex.get().getUAV(0,0,1))],
            ["summation", sum_buffer.get().getBindingResource()],
        ])
        
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(int(512 / 64), int(512 / 64), 1)
        encoder.end()


class VisualizePass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/visualize.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.visualize_ideal = None
        self.visualize_actual = None
        self.val_max = se.Float32(1.0)
        self.colormap = se.Int32(0)

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInput("PMFTexture_Ideal")\
            .isTexture().withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("PMFTexture_Actual")\
            .isTexture().withFormat(se.rhi.TextureFormat.R32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("Summation_Ideal")\
            .isBuffer().withSize(4 * 16)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addInput("Summation_Actual")\
            .isBuffer().withSize(4 * 16)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("PMFIdeal")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA8_UNORM)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("PMFActual")\
            .isTexture().withSize(se.ivec3(512, 512, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA8_UNORM)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):        
        self.updateBindings(rdrCtx, [
            ["pmf_ideal_texture", core.rhi.BindingResource(
                rdrDat.getTexture("PMFTexture_Ideal").get().getUAV(0,0,1))],
            ["pmf_actual_texture", core.rhi.BindingResource(
                rdrDat.getTexture("PMFTexture_Actual").get().getUAV(0,0,1))],
            ["pmf_ideal_summation", rdrDat.getBuffer("Summation_Ideal").get().getBindingResource()],
            ["pmf_actual_summation", rdrDat.getBuffer("Summation_Actual").get().getBindingResource()],
            ["pmf_ideal_visualize", core.rhi.BindingResource(
                rdrDat.getTexture("PMFIdeal").get().getUAV(0,0,1))],
            ["pmf_actual_visualize", core.rhi.BindingResource(
                rdrDat.getTexture("PMFActual").get().getUAV(0,0,1))],
        ])

        self.visualize_ideal = rdrDat.getTexture("PMFIdeal")
        self.visualize_actual = rdrDat.getTexture("PMFActual")
        
        class PushConstant(ctypes.Structure):
          _fields_ = [
            ("vmax", ctypes.c_float),
            ("colormap", ctypes.c_int),
        ]
        pConst = PushConstant(
            vmax = self.val_max.get(),
            colormap = self.colormap.get(),)
        
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(512 / 32), int(512 / 32), 1)
        encoder.end()

    def renderUI(self):
        sed.ImGui.DragFloat("Max Value", self.val_max, 0.01, 0, 1000)
        sed.ImGui.Combo("Colormap", self.colormap, ["viridis", "magma", "inferno", "plasma"])
        sed.ImGui.ShowTexture(self.visualize_ideal)
        sed.ImGui.ShowTexture(self.visualize_actual)


class NEEGraph(core.rdg.Graph):
    def __init__(self):
        core.rdg.Graph.__init__(self)
        # self.fwd_pass = NeuralGGXPass()
        self.fwd_pass = NEEPass()
        self.accum_pass = se.passes.AccumulatePass(se.ivec3(1024, 1024, 1))
        self.addPass(self.fwd_pass, "Render Pass")
        self.addPass(self.accum_pass, "Accum Pass")
        self.addEdge("Render Pass", "Color", "Accum Pass", "Input")

        self.reduce_0 = ReducePass()
        self.reduce_1 = ReducePass()
        self.addPass(self.reduce_0, "Reduce Pass 0")
        self.addPass(self.reduce_1, "Reduce Pass 1")
        self.addEdge("Render Pass", "PMFIdeal", "Reduce Pass 0", "PMFTexture")
        self.addEdge("Render Pass", "PMFActual", "Reduce Pass 1", "PMFTexture")

        self.visualize = VisualizePass()
        self.addPass(self.visualize, "Visualize Pass")
        self.addEdge("Reduce Pass 0", "Summation", "Visualize Pass", "Summation_Ideal")
        self.addEdge("Reduce Pass 0", "PMFTexture", "Visualize Pass", "PMFTexture_Ideal")
        self.addEdge("Reduce Pass 1", "Summation", "Visualize Pass", "Summation_Actual")
        self.addEdge("Reduce Pass 1", "PMFTexture", "Visualize Pass", "PMFTexture_Actual")

        self.markOutput("Accum Pass", "Output")

class NEEStratifiedGraph(core.rdg.Graph):
    def __init__(self):
        core.rdg.Graph.__init__(self)
        # self.fwd_pass = NeuralGGXPass()
        self.fwd_pass = NEEStratifiedPass()
        self.accum_pass = se.passes.AccumulatePass(se.ivec3(1024, 1024, 1))
        self.addPass(self.fwd_pass, "Render Pass")
        self.addPass(self.accum_pass, "Accum Pass")
        self.addEdge("Render Pass", "Color", "Accum Pass", "Input")
        self.markOutput("Accum Pass", "Output")


class NEEPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.graph = NEEGraph()
        self.setGraph(self.graph)
    
    def onUpdate(self, ctx:SEContext):
        pass
    
    def renderUI(self):
        pass

class NEEStratifiedPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.graph = NEEStratifiedGraph()
        self.setGraph(self.graph)
    
    def onUpdate(self, ctx:SEContext):
        pass
    
    def renderUI(self):
        pass