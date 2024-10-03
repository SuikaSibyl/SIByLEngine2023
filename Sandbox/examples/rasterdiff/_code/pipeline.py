# import se modules
from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed

class TrainSetting:
    def __init__(self, vertex_buffer_size:int, index_buffer_size:int):
        self.vertex_buffer_size = vertex_buffer_size
        self.index_buffer_size = index_buffer_size
        self.epsilon_vert = se.Float32(0.01)
        self.epsilon_albedo = se.Float32(0.01)
        self.camera_idx = se.Int32(0)
        self.learning_rate = se.Float32(0.0)
        self.learning_rate2 = se.Float32(0.0)
        self.smooth_rate = se.Float32(0.00)
        self.spp = se.Int32(1)
        self.random_seed = 0
        self.batch_size = 1
        self.lr = 0.01
        self.epoch = 1000
        self.epsilon = 0.01
        # load the ground truth
        gt = torch.load('examples/rasterdiff/_data/gt.pt')
        self.gt = SETensor(core.gfx.Context.getDevice(), [14, 800, 800], False, core.rhi.EnumDataType.INT32)
        self.gt.as_torch().copy_(gt)



class ParamPerturbPass(core.rdg.ComputePass):
    class PushConstant(ctypes.Structure):
        _fields_ = [
        ("position_num", ctypes.c_int),
        ("random_seed", ctypes.c_int),
        ("epsilon_albedo", ctypes.c_float),
        ("epsilon_pos",  ctypes.c_float),
        ("initialize", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
        ("learning_rate2", ctypes.c_float),
        ("smooth_rate", ctypes.c_float),
        ]

    def __init__(self, setting:TrainSetting):
        self.setting = setting
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "rasterdiff/_shader/perturb.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.initialized = False
        self.pConst = ParamPerturbPass.PushConstant()
        self.frame = 0
    
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Albedo")\
            .isTexture().withSize(se.ivec3(1024, 1024, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Albedo+")\
            .isTexture().withSize(se.ivec3(1024, 1024, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Albedo-")\
            .isTexture().withSize(se.ivec3(1024, 1024, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("DAlbedo")\
            .isBuffer().withSize(4 * 1024 * 1024 * 3)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addOutput("Position")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE)
                        |int(se.rhi.EnumBufferUsage.COPY_DST))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("Position+")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("Position-")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("DPosition")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("PositionAverage")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addOutput("PositionAverageCount")\
            .isBuffer().withSize(self.setting.vertex_buffer_size // 3)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        albedo:se.gfx.TextureHandle = rdrDat.getTexture("Albedo")
        position:se.gfx.BufferHandle = rdrDat.getBuffer("Position")
        position_plus:se.gfx.BufferHandle = rdrDat.getBuffer("Position+")
        position_minus:se.gfx.BufferHandle = rdrDat.getBuffer("Position-")
        dposition:se.gfx.BufferHandle = rdrDat.getBuffer("DPosition")
        scene:se.gfx.SceneHandle = rdrDat.getScene()

        self.setting.random_seed = np.random.randint(0, 100000)
        self.pConst.position_num = self.setting.vertex_buffer_size // 12
        self.pConst.random_seed = self.setting.random_seed
        self.pConst.epsilon_albedo = self.setting.epsilon_albedo.get()
        self.pConst.epsilon_pos = self.setting.epsilon_vert.get()
        self.pConst.initialize = 1 if self.initialized == False else 0
        self.pConst.smooth_rate = self.setting.smooth_rate.get()
        if self.frame % self.setting.spp.get() == 0:
            self.pConst.learning_rate = self.setting.learning_rate.get() / self.setting.spp.get()
            self.pConst.learning_rate2 = self.setting.learning_rate2.get() / self.setting.spp.get()
        else:
            self.pConst.learning_rate = 0
            self.pConst.learning_rate2 = 0
        self.frame += 1

        if self.initialized == False:
            self.initialized = True
            position_buffer:se.gfx.BufferHandle = scene.get().getGPUScene().getPositionBuffer()
            core.gfx.Context.getDevice().copyBufferToBuffer(position_buffer.get().getDevice(), 0,
                position.get().getDevice(), 0, self.setting.vertex_buffer_size)
        
        self.updateBindings(rdrCtx, [
            ["Position", position.get().getBindingResource()],
            ["PositionPlus", position_plus.get().getBindingResource()],
            ["PositionMinus", position_minus.get().getBindingResource()],
            ["dPosition", dposition.get().getBindingResource()],
            ["Albedo", core.rhi.BindingResource(albedo.get().getUAV(0,0,1))],
            ["AlbedoPlus", core.rhi.BindingResource(rdrDat.getTexture("Albedo+").get().getUAV(0,0,1))],
            ["AlbedoMinus", core.rhi.BindingResource(rdrDat.getTexture("Albedo-").get().getUAV(0,0,1))],
            ["dAlbedo", rdrDat.getBuffer("DAlbedo").get().getBindingResource()],
            ["PositionAvg", rdrDat.getBuffer("PositionAverage").get().getBindingResource()],
            ["PositionCount", rdrDat.getBuffer("PositionAverageCount").get().getBindingResource()],
        ])
        
        vertex_size = self.setting.vertex_buffer_size // 12
        texel_size = 1024 * 1024
        thread_size = max(vertex_size, texel_size)

        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(self.pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(self.pConst))
        encoder.dispatchWorkgroups(int(thread_size + 511) // 512, 1, 1)
        encoder.end()

    def renderUI(self):
        pass


class LaplacePass(core.rdg.ComputePass):
    class PushConstant(ctypes.Structure):
        _fields_ = [
        ("primitive_num", ctypes.c_int),
        ]

    def __init__(self, setting:TrainSetting):
        core.rdg.ComputePass.__init__(self)
        self.setting = setting
        [self.comp] = core.gfx.Context.load_shader_slang(
            "rasterdiff/_shader/laplace.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.pConst = LaplacePass.PushConstant()
        self.pConst.primitive_num = self.setting.index_buffer_size // 12
    
    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInputOutput("Position")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE)
                        |int(se.rhi.EnumBufferUsage.COPY_DST))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addInputOutput("PositionAverage")\
            .isBuffer().withSize(self.setting.vertex_buffer_size)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        reflector.addInputOutput("PositionAverageCount")\
            .isBuffer().withSize(self.setting.vertex_buffer_size // 3)\
            .withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        position:se.gfx.BufferHandle = rdrDat.getBuffer("Position")
        position_avg:se.gfx.BufferHandle = rdrDat.getBuffer("PositionAverage")
        position_count:se.gfx.BufferHandle = rdrDat.getBuffer("PositionAverageCount")
        scene:se.gfx.SceneHandle = rdrDat.getScene()

        self.updateBindings(rdrCtx, [
            ["Position", position.get().getBindingResource()],
            ["PositionAvg", position_avg.get().getBindingResource()],
            ["PositionCount", position_count.get().getBindingResource()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
        ])
        
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(self.pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(self.pConst))
        encoder.dispatchWorkgroups(int(self.pConst.primitive_num + 511) // 512, 1, 1)
        encoder.end()

    def renderUI(self):
        pass


class RasterFwdPass(core.rdg.RenderPass):
    class PushConstant(ctypes.Structure):
        _fields_ = [
            ("resolutionX", ctypes.c_int),
            ("resolutionY", ctypes.c_int),
            ("geometryIndex", ctypes.c_int),
            ("cameraIndex",  ctypes.c_int),
        ]

    def __init__(self, setting:TrainSetting):
        core.rdg.RenderPass.__init__(self)
        self.setting = setting
        [self.vert, self.frag] = core.gfx.Context.load_shader_slang(
            "rasterdiff/_shader/forward.slang",
            [("vertexMain", core.rhi.EnumShaderStage.VERTEX),
             ("fragmentMain", core.rhi.EnumShaderStage.FRAGMENT)], [], False)
        self.init(self.vert.get(), self.frag.get())
        self.pConst = RasterFwdPass.PushConstant()

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.COLOR_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.ColorAttachment)\
                .setAttachmentLoc(0))
        reflector.addOutput("Index")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RG32_UINT)\
            .withUsages(se.rhi.TextureUsageBit.COLOR_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.ColorAttachment)\
                .setAttachmentLoc(1))
        reflector.addOutput("Depth")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.DEPTH32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.DEPTH_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.DepthStencilAttachment)\
                .enableDepthWrite(True).setAttachmentLoc(0).setDepthCompareFn(se.rhi.CompareFunction.LESS_EQUAL))
        reflector.addInputOutput("Position")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.VERTEX_SHADER_BIT)
                          |int(se.rhi.PipelineStageBit.FRAGMENT_SHADER_BIT)))
        reflector.addInputOutput("Albedo")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.FRAGMENT_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color:se.gfx.TextureHandle = rdrDat.getTexture("Color")
        index:se.gfx.TextureHandle = rdrDat.getTexture("Index")
        depth:se.gfx.TextureHandle = rdrDat.getTexture("Depth")
        albedo:se.gfx.TextureHandle = rdrDat.getTexture("Albedo")
        position:se.gfx.BufferHandle = rdrDat.getBuffer("Position")
        scene:se.gfx.SceneHandle = rdrDat.getScene()

        self.setRenderPassDescriptor(se.rhi.RenderPassDescriptor([
            se.rhi.RenderPassColorAttachment(color.get().getRTV(0,0,1), None, se.vec4(0,0,0,1), se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE),
            se.rhi.RenderPassColorAttachment(index.get().getRTV(0,0,1), None, se.vec4(0,0,0,1), se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE)
        ], se.rhi.RenderPassDepthStencilAttachment(depth.get().getDSV(0,0,1), 1, se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE, False, 0, se.rhi.LoadOp.DONT_CARE, se.rhi.StoreOp.DONT_CARE, False)))

        self.updateBindings(rdrCtx, [
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            # ["GPUScene_position", scene.get().getGPUScene().bindingResourcePosition()],
            ["GPUScene_position", position.get().getBindingResource()],
            ["u_texture",  core.rhi.BindingResource(albedo.get().getSRV(0,1,0,1))],
        ])
        
        self.pConst.resolutionX = color.get().getWidth()
        self.pConst.resolutionY = color.get().getHeight()
        self.pConst.cameraIndex = self.setting.camera_idx.get()

        encoder = self.beginPass(rdrCtx, color.get())
        self.issueDirectDrawcalls(encoder, scene)
        encoder.end()

    # @debug.traced
    def beforeDirectDrawcall(self, encoder:core.rhi.RenderPassEncoder, geometryIdx:int, geometry:core.gfx.GeometryDrawData):
        self.pConst.geometryIndex = geometryIdx
        encoder.pushConstants(get_ptr(self.pConst), int(int(core.rhi.EnumShaderStage.VERTEX) | int(core.rhi.EnumShaderStage.FRAGMENT)), 0, ctypes.sizeof(self.pConst))

    def renderUI(self):
        pass


class RasterViewFwdPass(core.rdg.RenderPass):
    class PushConstant(ctypes.Structure):
        _fields_ = [
            ("resolutionX", ctypes.c_int),
            ("resolutionY", ctypes.c_int),
            ("geometryIndex", ctypes.c_int),
            ("cameraIndex",  ctypes.c_int),
            ("dbg_mode", ctypes.c_int),
        ]

    def __init__(self, setting:TrainSetting):
        core.rdg.RenderPass.__init__(self)
        self.setting = setting
        [self.vert, self.frag] = core.gfx.Context.load_shader_slang(
            "rasterdiff/_shader/inspector.slang",
            [("vertexMain", core.rhi.EnumShaderStage.VERTEX),
             ("fragmentMain", core.rhi.EnumShaderStage.FRAGMENT)], [], False)
        self.init(self.vert.get(), self.frag.get())
        self.pConst = RasterViewFwdPass.PushConstant()
        self.gt = setting.gt
        self.debug_mode = se.Int32(0)

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Color")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.COLOR_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.ColorAttachment)\
                .setAttachmentLoc(0))
        reflector.addOutput("Index")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.RG32_UINT)\
            .withUsages(se.rhi.TextureUsageBit.COLOR_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.ColorAttachment)\
                .setAttachmentLoc(1))
        reflector.addOutput("Depth")\
            .isTexture().withSize(se.vec3(1, 1, 1))\
            .withFormat(se.rhi.TextureFormat.DEPTH32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.DEPTH_ATTACHMENT)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.DepthStencilAttachment)\
                .enableDepthWrite(True).setAttachmentLoc(0).setDepthCompareFn(se.rhi.CompareFunction.LESS_EQUAL))
        reflector.addInputOutput("Position")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.VERTEX_SHADER_BIT)
                          |int(se.rhi.PipelineStageBit.FRAGMENT_SHADER_BIT)))
        reflector.addInputOutput("Albedo")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.FRAGMENT_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        color:se.gfx.TextureHandle = rdrDat.getTexture("Color")
        index:se.gfx.TextureHandle = rdrDat.getTexture("Index")
        depth:se.gfx.TextureHandle = rdrDat.getTexture("Depth")
        albedo:se.gfx.TextureHandle = rdrDat.getTexture("Albedo")
        position:se.gfx.BufferHandle = rdrDat.getBuffer("Position")
        scene:se.gfx.SceneHandle = rdrDat.getScene()
    
        self.setRenderPassDescriptor(se.rhi.RenderPassDescriptor([
            se.rhi.RenderPassColorAttachment(color.get().getRTV(0,0,1), None, se.vec4(0,0,0,1), se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE),
            se.rhi.RenderPassColorAttachment(index.get().getRTV(0,0,1), None, se.vec4(0,0,0,1), se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE)
        ], se.rhi.RenderPassDepthStencilAttachment(depth.get().getDSV(0,0,1), 1, se.rhi.LoadOp.CLEAR, se.rhi.StoreOp.STORE, False, 0, se.rhi.LoadOp.DONT_CARE, se.rhi.StoreOp.DONT_CARE, False)))

        self.updateBindings(rdrCtx, [
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
            ["GPUScene_position", position.get().getBindingResource()],
            ["u_texture",  core.rhi.BindingResource(albedo.get().getSRV(0,1,0,1))],
            ["GroundTruth", core.rhi.BindingResource(core.rhi.BufferBinding(self.gt.prim(), 0, self.gt.prim().size()))],
        ])
        
        self.pConst.resolutionX = color.get().getWidth()
        self.pConst.resolutionY = color.get().getHeight()
        self.pConst.cameraIndex = scene.get().getEditorActiveCameraIndex()
        self.pConst.dbg_mode = self.debug_mode.get()
        
        encoder = self.beginPass(rdrCtx, color.get())
        self.issueDirectDrawcalls(encoder, scene)
        encoder.end()

    # @debug.traced
    def beforeDirectDrawcall(self, encoder:core.rhi.RenderPassEncoder, geometryIdx:int, geometry:core.gfx.GeometryDrawData):
        self.pConst.geometryIndex = geometryIdx
        encoder.pushConstants(get_ptr(self.pConst), int(int(core.rhi.EnumShaderStage.VERTEX) | int(core.rhi.EnumShaderStage.FRAGMENT)), 0, ctypes.sizeof(self.pConst))

    def renderUI(self):
        sed.ImGui.Combo("Vis Mode", self.debug_mode, ["Fwd", "Primitive", "GT"])


class RasterDiffPass(core.rdg.ComputePass):
    class PushConstant(ctypes.Structure):
        _fields_ = [
        ("random_seed", ctypes.c_int),
        ("epsilon_albedo", ctypes.c_float),
        ("epsilon_pos",  ctypes.c_float),
        ("camera_index", ctypes.c_int),
        ]

    def __init__(self, setting:TrainSetting):
        core.rdg.ComputePass.__init__(self)
        self.setting = setting
        [self.comp] = core.gfx.Context.load_shader_slang(
            "rasterdiff/_shader/backward.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.epsilon_albedo = se.Float32(0.01)
        self.initialized = False
        self.pConst = RasterDiffPass.PushConstant()
        self.gt = setting.gt

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addInput("RenderedPlus")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("RenderedMinus")\
            .isTexture().withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("IndexPlus")\
            .isTexture().withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInput("IndexMinus")\
            .isTexture().withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInputOutput("DAlbedo")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        reflector.addInputOutput("DPosition")\
            .isBuffer().withUsages(int(se.rhi.EnumBufferUsage.STORAGE))\
            .consume(se.rdg.BufferInfo.ConsumeEntry()\
                .setAccess(int(se.rhi.AccessFlagBits.SHADER_READ_BIT)
                           |int(se.rhi.AccessFlagBits.SHADER_WRITE_BIT))
                .addStage(int(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT)))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        renderPlus:se.gfx.TextureHandle = rdrDat.getTexture("RenderedPlus")
        renderMinus:se.gfx.TextureHandle = rdrDat.getTexture("RenderedMinus")
        indexPlus:se.gfx.TextureHandle = rdrDat.getTexture("IndexPlus")
        indexMinus:se.gfx.TextureHandle = rdrDat.getTexture("IndexMinus")
        scene:se.gfx.SceneHandle = rdrDat.getScene()

        self.pConst.random_seed = self.setting.random_seed
        self.pConst.epsilon_albedo = self.epsilon_albedo.get()
        self.pConst.epsilon_pos = self.setting.epsilon_vert.get()
        self.pConst.camera_index = self.setting.camera_idx.get()
        
        self.updateBindings(rdrCtx, [
            ["RenderedPlus", core.rhi.BindingResource(renderPlus.get().getUAV(0,0,1))],
            ["RenderedMinus", core.rhi.BindingResource(renderMinus.get().getUAV(0,0,1))],
            ["IndexPlus", core.rhi.BindingResource(indexPlus.get().getUAV(0,0,1))],
            ["IndexMinus", core.rhi.BindingResource(indexMinus.get().getUAV(0,0,1))],
            ["GroundTruth", core.rhi.BindingResource(core.rhi.BufferBinding(self.gt.prim(), 0, self.gt.prim().size()))],
            ["dAlbedo", rdrDat.getBuffer("DAlbedo").get().getBindingResource()],
            ["dPosition", rdrDat.getBuffer("DPosition").get().getBindingResource()],
            ["GPUScene_camera", scene.get().getGPUScene().bindingResourceCamera()],
            ["GPUScene_index", scene.get().getGPUScene().bindingResourceIndex()],
            ["GPUScene_vertex", scene.get().getGPUScene().bindingResourceVertex()],
            ["GPUScene_geometry", scene.get().getGPUScene().bindingResourceGeometry()],
        ])

        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(get_ptr(self.pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(self.pConst))
        encoder.dispatchWorkgroups(int(800 // 16), int(800 // 16), 1)
        encoder.end()


class DiffRasterGraph(core.rdg.Graph):
    def __init__(self, setting:TrainSetting):
        core.rdg.Graph.__init__(self)
        self.perturb_pass = ParamPerturbPass(setting)
        self.fwd_pls_pass = RasterFwdPass(setting)
        self.fwd_mns_pass = RasterFwdPass(setting)
        self.addPass(self.perturb_pass, "Perturb Pass")
        self.addPass(self.fwd_pls_pass, "Forward+ Pass")
        self.addPass(self.fwd_mns_pass, "Forward- Pass")
        self.addEdge("Perturb Pass", "Position+", "Forward+ Pass", "Position")
        self.addEdge("Perturb Pass", "Albedo+", "Forward+ Pass", "Albedo")
        self.addEdge("Perturb Pass", "Position-", "Forward- Pass", "Position")
        self.addEdge("Perturb Pass", "Albedo-", "Forward- Pass", "Albedo")
        self.bwd_pass = RasterDiffPass(setting)
        self.addPass(self.bwd_pass, "Backward Pass")
        self.addEdge("Perturb Pass", "DAlbedo", "Backward Pass", "DAlbedo")
        self.addEdge("Perturb Pass", "DPosition", "Backward Pass", "DPosition")
        self.addEdge("Forward+ Pass", "Color", "Backward Pass", "RenderedPlus")
        self.addEdge("Forward+ Pass", "Index", "Backward Pass", "IndexPlus")
        self.addEdge("Forward- Pass", "Color", "Backward Pass", "RenderedMinus")
        self.addEdge("Forward- Pass", "Index", "Backward Pass", "IndexMinus")
        self.view_pass = RasterViewFwdPass(setting)
        self.addPass(self.view_pass, "View Pass")
        self.addEdge("Perturb Pass", "Position", "View Pass", "Position")
        self.addEdge("Perturb Pass", "Albedo", "View Pass", "Albedo")
        self.laplace_pass = LaplacePass(setting)
        self.addPass(self.laplace_pass, "Laplace Pass")
        self.addEdge("Perturb Pass", "Position", "Laplace Pass", "Position")
        self.addEdge("Perturb Pass", "PositionAverage", "Laplace Pass", "PositionAverage")
        self.addEdge("Perturb Pass", "PositionAverageCount", "Laplace Pass", "PositionAverageCount")
    

        self.markOutput("View Pass", "Color")

class DiffRasterPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self, vertex_buffer_size:int, index_buffer_size:int):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.setting = TrainSetting(vertex_buffer_size, index_buffer_size)
        self.graph = DiffRasterGraph(self.setting)
        self.setGraph(self.graph)

    def renderUI(self):
        sed.ImGui.DragInt("Camera Index", self.setting.camera_idx, 1, 0, 14)
        sed.ImGui.DragFloat("LR", self.setting.learning_rate, 0.001, 0, 1)
        sed.ImGui.DragFloat("LR2", self.setting.learning_rate2, 0.001, 0, 1)
        sed.ImGui.DragInt("SPP", self.setting.spp, 1, 1, 1000)
        sed.ImGui.DragFloat("smooth_rate", self.setting.smooth_rate, 0.01, 0, 1)
        sed.ImGui.DragFloat("epsilon_vert", self.setting.epsilon_vert, 0.001, 0, 1)
