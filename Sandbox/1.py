import torch
import ctypes
import numpy as np
import torch.nn as nn
import torch.optim as optim
import se.common
import se.pycore as core
from se.common import *

class ReparamFwdPass(core.rdg.ComputePass):
    def __init__(self, image_tensor:SETensor, transform_tensor:SETensor):
        core.rdg.ComputePass.__init__(self)  # Without this, a TypeError is raised.
        [self.comp] = core.gfx.Context.load_shader_slang(
            "S:/SIByL2024/Sandbox/shaders/reparam/forward.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], True)
        self.init(self.comp.get())
        self.image_tensor = image_tensor
        self.transform_tensor = transform_tensor
    
    def reflect(self) -> core.rdg.PassReflection:
        return core.rdg.PassReflection()
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        self.updateBindings(rdrCtx, [
            ["u_image", core.rhi.BindingResource(core.rhi.BufferBinding(self.image_tensor.prim(), 0, self.image_tensor.prim().size()))],
            ["u_trans", core.rhi.BindingResource(core.rhi.BufferBinding(self.transform_tensor.prim(), 0, self.transform_tensor.prim().size()))],
        ])
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(32, 32, 1)
        encoder.end()

class ReparamBwdPass(core.rdg.ComputePass):
    def __init__(self, image_tensor:SETensor, transform_tensor:SETensor):
        core.rdg.ComputePass.__init__(self)  # Without this, a TypeError is raised.
        [self.comp] = core.gfx.Context.load_shader_slang(
            "S:/SIByL2024/Sandbox/shaders/reparam/backward.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], True)
        self.init(self.comp.get())
        self.image_tensor = image_tensor
        self.transform_tensor = transform_tensor
        self.seed = 0
    
    def reflect(self) -> core.rdg.PassReflection:
        return core.rdg.PassReflection()
    
    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        self.updateBindings(rdrCtx, [
            ["u_gradient", core.rhi.BindingResource(core.rhi.BufferBinding(self.image_tensor.grad(), 0, self.image_tensor.prim().size()))],
            ["u_trans", core.rhi.BindingResource(core.rhi.BufferBinding(self.transform_tensor.prim(), 0, self.transform_tensor.prim().size()))],
            ["u_trans_gradient", core.rhi.BindingResource(core.rhi.BufferBinding(self.transform_tensor.grad(), 0, self.transform_tensor.grad().size()))],
        ])
        class PushConstant(ctypes.Structure):
          _fields_ = [("random_seed", ctypes.c_uint32)]
        pConst = PushConstant(random_seed=self.seed)
        # execute the pass with the cmd encoder
        encoder = self.beginPass(rdrCtx)
        encoder.pushConstants(se.common.get_ptr(pConst), int(core.rhi.EnumShaderStage.COMPUTE), 0, ctypes.sizeof(pConst))
        encoder.dispatchWorkgroups(32, 32, 1)
        encoder.end()

class ReparamTriModule(nn.Module):
  def __init__(self, ctx:SEContext) -> None:
    super(ReparamTriModule, self).__init__()
    self.image = SETensor(ctx.getDevice(), [512, 512], True)
    self.translation = SETensor(SECtx.getDevice(), [3], True)
    # initialize translation [gt translation: (0.1, -0.1, 0.1)]
    with torch.no_grad():
      self.translation.as_torch()[0] = -1.0
      self.translation.as_torch()[1] = -0.3
      self.translation.as_torch()[2] = -0.3
    # create the forward and backward passes
    self.fwdPass = ReparamFwdPass(self.image, self.translation)
    self.bwdPass = ReparamBwdPass(self.image, self.translation)

  def forward(self, ctx:SEContext):
    ctx.executeOnePass(self.fwdPass)
    return self.image.as_torch()

  def backward(self, ctx:SEContext):
    self.bwdPass.seed = np.random.randint(0, 1000)
    ctx.executeOnePass(self.bwdPass)


canvas = SEPltCanvas(1, 2, (10, 5))
canvas.imagePlot = SEActiveImage(canvas.axs[0], 512, 512, 'Image', -0.04, +0.04, 'viridis')
canvas.lossPlot = SECurvePlot(canvas.axs[1], 'Loss')

SECtx = SEContext()
reparam_module = ReparamTriModule(SECtx)
optimizer = optim.Adam([reparam_module.translation.as_torch()], lr=0.005)

gt_np = np.load("gt.npy")
gt_torch = torch.from_numpy(gt_np).cuda()

for epoch in range(1000):
    # forward - render
    optimizer.zero_grad(set_to_none=False)
    output = reparam_module.forward(SECtx)
    loss = torch.sum((output - gt_torch)**2)
    # backward - loss
    output.grad.zero_()
    loss.backward()
    reparam_module.backward(SECtx)
    # gradient descent
    optimizer.step()

    # plot the loss
    canvas.lossPlot.push(epoch, loss.item())
    # visualize the result
    if epoch % 10 == 0:
      canvas.imagePlot.update(output.grad.detach().cpu().numpy())
      canvas.lossPlot.invalidate()
      canvas.flush()
    # if the plt canvas is closed, break the loop
    if canvas.has_been_closed():
      break

del reparam_module
del SECtx