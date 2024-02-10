import torch
import numpy as np
import se.pycore as core
import se.pyapp as app
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.optim as optim
import torch.nn as nn
from functools import reduce
import operator

class SEContext:
  def __init__(self):
    context_extensions = int(core.rhi.EnumContextExtension.RayTracing) \
      | int(core.rhi.EnumContextExtension.RayTracing) \
      | int(core.rhi.EnumContextExtension.BindlessIndexing) \
      | int(core.rhi.EnumContextExtension.FragmentBarycentric) \
      | int(core.rhi.EnumContextExtension.ConservativeRasterization) \
      | int(core.rhi.EnumContextExtension.CooperativeMatrix) \
      | int(core.rhi.EnumContextExtension.CudaInteroperability) \
      | int(core.rhi.EnumContextExtension.AtomicFloat)
    self.context = core.rhi.Context.create(core.rhi.Context.EnumBackend.Vulkan)
    self.context.init(None, context_extensions)
    self.adapater = self.context.requestAdapter(core.rhi.RequestAdapterOptions())
    self.device = self.adapater.requestDevice()
    core.rhi.CUDAContext.initialize(self.device)
    core.gfx.Context.initialize(self.device)

  def __del__(self):
    core.gfx.Context.finalize()
    self.device = None
    self.adapater = None
    self.context = None
    core.gfx.Context.finalize()


  def getDevice(self):
    return self.device

class SETensor:
  def __init__(self, device, shape, requires_grad=False):
    # multiply all the dimensions of the shape
    self.size = reduce(operator.mul, shape, 1)
    # create the primal buffer
    self.prim_se = device.createBuffer(core.rhi.BufferDescriptor(self.size * 4, 
      int(core.rhi.EnumBufferUsage.STORAGE),
      core.rhi.EnumBufferShareMode.EXCLUSIVE,
      int(core.rhi.EnumMemoryProperty.DEVICE_LOCAL_BIT)))
    self.prim_cu = core.rhi.CUDAContext.toCUDABuffer(self.prim_se)
    self.prim_torch = core.rhi.toTensor(self.prim_cu, shape)
    # create the gradient buffer
    if requires_grad:
      self.grad_se = device.createBuffer(core.rhi.BufferDescriptor(self.size * 4, 
        int(core.rhi.EnumBufferUsage.STORAGE),
        core.rhi.EnumBufferShareMode.EXCLUSIVE,
        int(core.rhi.EnumMemoryProperty.DEVICE_LOCAL_BIT)))
      self.grad_cu = core.rhi.CUDAContext.toCUDABuffer(self.grad_se)
      self.grad_torch = core.rhi.toTensor(self.grad_cu, shape)
      self.prim_torch.requires_grad = True
      self.prim_torch.grad = self.grad_torch
    else:
      self.grad_se = None
      self.grad_cu = None
      self.grad_torch = None
  
  def __del__(self):
    self.prim_torch = None
    self.grad_torch = None
    self.prim_cu = None
    self.grad_cu = None
    self.prim_se = None
    self.grad_se = None

  def as_torch(self):
    return self.prim_torch
  def prim(self):
    return self.prim_se
  def grad(self):
    return self.grad_se

# Enable interactive mode
plt.ion()

# Create a figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
x_data, y_data = [], []

# Initial render data
image_data = np.random.rand(512,512) # Dummy image data
img_data = ax1.imshow(image_data, aspect='auto', vmin=-0.001, vmax=0.001)
ax1.set_title('Error')

# Initial plot
line, = ax2.plot(x_data, y_data)
ax2.set_xlim(0, 10)  # Initial x-axis limits
# ax2.set_ylim(-1, 1)  # Assuming y-data ranges within -1 to 1
ax2.set_title('Loss')

# Automatically adjust subplot params so that the subplot(s) fits into the figure area
plt.tight_layout()

gt_np = np.load("gt.npy")
gt_torch = torch.from_numpy(gt_np).cuda()

SECtx = SEContext()

image_tensor = SETensor(SECtx.getDevice(), [512, 512], True)
trans_tensor = SETensor(SECtx.getDevice(), [3], True)

#(0.1, -0.1, 0.1)
with torch.no_grad():
  trans_tensor.prim_torch[0] = -1.0
  trans_tensor.prim_torch[1] = -0.3
  trans_tensor.prim_torch[2] = -0.3

core.rhi.CUDAContext.synchronize()

fwdPass = app.FwdPass()
fwdPass.image_buffer = image_tensor.prim()
fwdPass.tran_buffer = trans_tensor.prim()

bwdPass = app.BwdPass()
bwdPass.grad_buffer = image_tensor.grad()
bwdPass.tran_buffer = trans_tensor.prim()
bwdPass.trangrad_buffer = trans_tensor.grad()

optimizer = optim.Adam([trans_tensor.as_torch()], lr=0.005)

rdrCtx = core.rdg.RenderContext()
rdrDat = core.rdg.RenderData()
rdrCtx.flightIdx = 0

cmdEncDesc = core.rhi.CommandEncoderDescriptor()
queue = SECtx.getDevice().getGraphicsQueue()

loss_list = []

for loop in range(100):
  for i in range(10):
    # forward - render
    optimizer.zero_grad(set_to_none=False)

    commandEncoder = SECtx.getDevice().createCommandEncoder(cmdEncDesc)
    rdrCtx.cmdEncoder = commandEncoder
    fwdPass.execute(rdrCtx, rdrDat)
    cmdBuffer = commandEncoder.finish()
    queue.submit([cmdBuffer])
    SECtx.getDevice().waitIdle()
    
    # forward - loss
    loss = torch.sum((image_tensor.as_torch() - gt_torch)**2)
    image_tensor.as_torch().grad.zero_()
    # trangrad_torch.zero_()
    loss.backward()
    core.rhi.CUDAContext.synchronize()

    # backward - render
    commandEncoder = SECtx.getDevice().createCommandEncoder(cmdEncDesc)
    rdrCtx.cmdEncoder = commandEncoder
    bwdPass.setSeed(2)
    bwdPass.execute(rdrCtx, rdrDat)
    cmdBuffer2 = commandEncoder.finish()
    queue.submit([cmdBuffer2])
    SECtx.getDevice().waitIdle()

    # gradient descent
    optimizer.step()
    # tran_torch -= 1e-6 * trangrad_torch
    core.rhi.CUDAContext.synchronize()
    
    x_data.append(loop*10 + i)
    y_data.append(loss.item())

    # visualize the result
    if i == 9:
      # Append new data to the dataset
      new_x = loop*10 + i
      x_data.append(new_x)
      y_data.append(loss.item())

      if loop == 0:
        ax2.set_ylim(0, y_data[0] + 10)
       
      # Update the data of the plot
      line.set_data(x_data, y_data)
      img_data.set_data(image_tensor.as_torch().grad.detach().cpu().numpy())
     
      # Adjust x-axis limits dynamically
      if new_x >= ax2.get_xlim()[1]:
          ax2.set_xlim(ax2.get_xlim()[0], new_x + 10)
      
      # line.set_data(x_data, y_data)
      # Adjust x-axis limits dynamically
      # if (loop*10 + i) >= subfig1.get_xlim()[1]:
      #   subfig1.set_xlim(subfig1.get_xlim()[0], (loop*10 + i))
      
      # subfig2.plot(loss_list, color='blue')
      # subfig2.set_aspect(1)
      # subfig2.set_xticks([])
      # fig.canvas.draw()
      fig.canvas.flush_events()
      print(loop)

del fwdPass
del bwdPass

del image_tensor
del trans_tensor

del SECtx
print("end safely")