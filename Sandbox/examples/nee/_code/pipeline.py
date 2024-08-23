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
        self.nee_sampling_mode = se.Int32(0)

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
            ("nee_sampling_mode", ctypes.c_int),
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
            nee_sampling_mode = self.nee_sampling_mode.get(),)
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
        sed.ImGui.Combo("Render Mode", self.nee_sampling_mode, [
            "Cosine Hemisphere Sample", 
            "Uniform Area Sample",
            "Uniform Solid Angle Sample",
            "Bilinear Solid Angle Sample",
        ])

class MultivariateCommon:
    def __init__(self):
        self.multivariable_estimator_mode = se.Int32(0)
        self.accum_frame = 0
        self.accum_target = se.Int32(10)
        self.accum_reset = False

class NEEStratifiedPass(core.rdg.ComputePass):
    def __init__(self, common:MultivariateCommon):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/stratified.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.nee_sampling_mode = se.Int32(0)
        self.sampler = se.gfx.Context.create_sampler_desc(se.rhi.EnumAddressMode.CLAMP_TO_EDGE, 
            se.rhi.EnumFilterMode.LINEAR, se.rhi.EnumMipmapFilterMode.NEAREST)
        self.lut1 = se.gfx.Context.create_texture_file("../Engine/binary/resources/textures/lut_ltc1.exr")
        self.lut2 = se.gfx.Context.create_texture_file("../Engine/binary/resources/textures/lut_ltc2.exr")
        self.common = common

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("ColorCV")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color = rdrDat.getTexture("Color")
        color_cv = rdrDat.getTexture("ColorCV")
        scene = rdrDat.getScene()
        
        self.updateBindings(rdrCtx, [
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            ["GPUScene_position", scene.get().getGPUScene().bindingResourcePosition()],
            ["GPUScene_tlas", scene.get().getGPUScene().bindingResourceTLAS()],
            ["u_image", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
            ["u_image_cv", core.rhi.BindingResource(color_cv.get().getUAV(0,0,1))],
            ["lut_ltc1", core.rhi.BindingResource(self.lut1.get().getSRV(0,1,0,1), self.sampler.get())],
            ["lut_ltc2", core.rhi.BindingResource(self.lut2.get().getSRV(0,1,0,1), self.sampler.get())],
        ])
        
        class PushConstant(ctypes.Structure):
          _fields_ = [
            ("random_seed", ctypes.c_int),
            ("camera_index", ctypes.c_int),
            ("nee_sampling_mode", ctypes.c_int),
            ("estimator_mode", ctypes.c_int),
            ("accum_frame", ctypes.c_int),
        ]
        should_accum = 0
        if self.common.accum_target.get() == 0:
            should_accum = 0
        elif self.common.accum_frame == 0:
            should_accum = 0
        elif self.common.accum_frame == self.common.accum_target.get():
            should_accum = 2
        else:
            should_accum = 1

        if (should_accum == 0 or should_accum == 1) and self.common.accum_target.get() != 0:
            self.common.accum_frame += 1

        pConst = PushConstant(
            random_seed=np.random.randint(0, 1000000),
            camera_index=scene.get().getEditorActiveCameraIndex(),
            nee_sampling_mode = self.nee_sampling_mode.get(),
            estimator_mode = self.common.multivariable_estimator_mode.get(),
            accum_frame = should_accum,)
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 32), int(1024 / 4), 1)
        encoder.end()

    def renderUI(self):
        sed.ImGui.Combo("NEE Mode", self.nee_sampling_mode, [
            "Uniform Area Sample",
            "Uniform Solid Angle Sample",
            "Bilinear Solid Angle Sample",
        ])
        
class RGBVisualizePass(core.rdg.ComputePass):
    def __init__(self, common:MultivariateCommon):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "nee/_shader/visualize-multivar.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.common = common

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInput("Color")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("ColorCV")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Output")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):   
        color = rdrDat.getTexture("Color")
        color_cv = rdrDat.getTexture("ColorCV")
        output = rdrDat.getTexture("Output")
     
        self.updateBindings(rdrCtx, [
            ["u_image", core.rhi.BindingResource(color.get().getUAV(0,0,1))],
            ["u_image_cv", core.rhi.BindingResource(color_cv.get().getUAV(0,0,1))],
            ["u_output", core.rhi.BindingResource(output.get().getUAV(0,0,1))],
        ])

        class PushConstant(ctypes.Structure):
          _fields_ = [
            ("estimator_type", ctypes.c_int),
            ("accum_frame", ctypes.c_int),
        ]
        pConst = PushConstant(
            estimator_type = self.common.multivariable_estimator_mode.get(),
            accum_frame = self.common.accum_frame,
        )
        
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(int(1024 / 16), int(1024 / 16), 1)
        encoder.end()

    def renderUI(self):
        pass

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

class NEEMultivariateGraph(core.rdg.Graph):
    def __init__(self, common:MultivariateCommon):
        core.rdg.Graph.__init__(self)
        # self.fwd_pass = NeuralGGXPass()
        self.fwd_pass = NEEStratifiedPass(common)
        self.vis_pass = RGBVisualizePass(common)
        
        self.addPass(self.fwd_pass, "Render Pass")
        self.addPass(self.vis_pass, "Visualize Pass")
        self.addEdge("Render Pass", "Color", "Visualize Pass", "Color")
        self.addEdge("Render Pass", "ColorCV", "Visualize Pass", "ColorCV")

        self.markOutput("Visualize Pass", "Output")


class NEEPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.graph = NEEGraph()
        self.setGraph(self.graph)
    
    def onUpdate(self, ctx:SEContext):
        pass
    
    def renderUI(self):
        pass


class NEEMultivariatePipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.common = MultivariateCommon()
        self.graph = NEEMultivariateGraph(self.common)
        self.setGraph(self.graph)
    
    def onUpdate(self, ctx:SEContext):
        pass
    
    def renderUI(self):
        sed.ImGui.Combo("Estimator Mode", 
            self.common.multivariable_estimator_mode, [
            "BRDF Sampling",
            "LTC Solution",
            "Luminance Importance Sampling",
            "Optimal Importance Sampling",
            "Stratified-Channel Sampling",
            "Stratified-Light Sampling",
            "Residual Control Variates",
            "Ratio Control Variates",
        ])
        sed.ImGui.DragInt("Accum Target", self.common.accum_target, 1, 1, 1000)
        sed.ImGui.Text(f"Accum Frame: {self.common.accum_frame}")
        sed.ImGui.SameLine()
        if sed.ImGui.Button("Reset Accum"):
            self.common.accum_reset = True
            self.common.accum_frame = 0