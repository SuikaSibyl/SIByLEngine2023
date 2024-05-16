# import se modules
from se.common import *
from se.editor import *
import se.pycore as se
import se.pyeditor as sed

class EPFLTestPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "epflmat/_shaders/epfl-test.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.brdf = se.bxdfs.EPFLBrdf("C:/Users/suika/Downloads/cc_ibiza_sunset_rgb.bsdf")
        se.bxdfs.EPFLBrdf.updateGPUResource()
    
    def __del__(self):
        del self.brdf

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        reflector.addOutput("Output")\
            .isTexture().withSize(se.ivec3(16, 16, 1))\
            .withFormat(se.rhi.TextureFormat.RGBA32_FLOAT)\
            .withUsages(se.rhi.TextureUsageBit.STORAGE_BINDING)\
            .consume(se.rdg.TextureInfo.ConsumeEntry(se.rdg.TextureInfo.ConsumeType.StorageBinding)\
                    .addStage(se.rhi.PipelineStageBit.COMPUTE_SHADER_BIT))
        return reflector
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        output = rdrDat.getTexture("Output")
        self.updateBindings(rdrCtx, [
            ["output", core.rhi.BindingResource(output.get().getUAV(0,0,1))],
            ["epfl_data_tensors", se.bxdfs.EPFLBrdf.bindingResourceBuffer()],
            ["epfl_materials", se.bxdfs.EPFLBrdf.bindingResourceBRDFs()],
        ])

        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(1, 1, 1)
        encoder.end()

    def renderUI(self):
        pass


class EPFLTestGraph(core.rdg.Graph):
    def __init__(self):
        core.rdg.Graph.__init__(self)
        self.test_pass = EPFLTestPass()
        self.addPass(self.test_pass, "TestPass")
        self.markOutput("TestPass", "Output")

class EPFLTestPipeline(core.rdg.SingleGraphPipeline):
    def __init__(self):
        core.rdg.SingleGraphPipeline.__init__(self)
        self.graph = EPFLTestGraph()
        self.setGraph(self.graph)