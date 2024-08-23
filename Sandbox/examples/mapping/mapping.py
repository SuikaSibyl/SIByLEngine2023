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
import numpy as np
import torch

class PMFBuildPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "guiding/octmap-sum.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], True)
        self.init(self.comp.get())
        self.octmap_tensor:SETensor = None

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        return reflector

    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        self.updateBindings(rdrCtx, [
            ["octmap_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.octmap_tensor.prim(), 0, self.octmap_tensor.prim().size()))],
        ])
    
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(1, 1, 1)
        encoder.end()

    def renderUI(self):
        pass
    
class MappingTestPass(core.rdg.ComputePass):
    def __init__(self):
        core.rdg.ComputePass.__init__(self)
        [self.comp] = core.gfx.Context.load_shader_slang(
            "mapping/_shaders/mapping-test.slang",
            [("ComputeMain", core.rhi.EnumShaderStage.COMPUTE)], [], False)
        self.init(self.comp.get())
        self.i_tensor:SETensor = None
        self.d_tensor:SETensor = None
        self.h_tensor:SETensor = None
        self.d2_tensor:SETensor = None
        self.warp_tensor:SETensor = None
        self.octmap_tensor:SETensor = None
        self.warp_3d_tensor:SETensor = None

    def reflect(self) -> core.rdg.PassReflection:
        reflector = core.rdg.PassReflection()
        return reflector

    def execute(self, rdrCtx:core.rdg.RenderContext, rdrDat:core.rdg.RenderData):
        self.updateBindings(rdrCtx, [
            ["input_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.i_tensor.prim(), 0, self.i_tensor.prim().size()))],
            ["disk_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.d_tensor.prim(), 0, self.d_tensor.prim().size()))],
            ["sphere_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.h_tensor.prim(), 0, self.h_tensor.prim().size()))],
            ["disk2_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.d2_tensor.prim(), 0, self.d2_tensor.prim().size()))],
            ["warp_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.warp_tensor.prim(), 0, self.warp_tensor.prim().size()))],
            ["warp_3d_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.warp_3d_tensor.prim(), 0, self.warp_3d_tensor.prim().size()))],
            ["octmap_buffer", core.rhi.BindingResource(core.rhi.BufferBinding(
                self.octmap_tensor.prim(), 0, self.octmap_tensor.prim().size()))],
        ])
    
        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(8192, 1, 1)
        encoder.end()

    def renderUI(self):
        pass


class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
    
    def onInit(self):
        # draw 64 * 64 grid
        grid_size = 16
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        # flatten X and Y
        X = X.flatten()
        Y = Y.flatten()
        # plot scatter with matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(221)
        # let the color be rgb value
        # where r is x value, y is y value, b is 0
        colors = [(r, g, 1) for r, g in zip(X, Y)]
        ax.scatter(X, Y, c=colors)
        # make each subplot a square
        ax.set_aspect('equal', adjustable='box')

        # create a SETensor
        i_tensor = SETensor(core.gfx.Context.getDevice(), [grid_size*grid_size, 2])
        d_tensor = SETensor(core.gfx.Context.getDevice(), [grid_size*grid_size, 2])
        h_tensor = SETensor(core.gfx.Context.getDevice(), [grid_size*grid_size, 4])
        d2_tensor = SETensor(core.gfx.Context.getDevice(), [grid_size*grid_size, 2])
        warp_tensor = SETensor(core.gfx.Context.getDevice(), [256 * 8192, 2])
        warp_3d_tensor = SETensor(core.gfx.Context.getDevice(), [256 * 8192, 4])
        i_tensor.as_torch()[:, 0] = torch.tensor(X)
        i_tensor.as_torch()[:, 1] = torch.tensor(Y)


        # create a 8*8 2d grid, intensity grows with x and y axis
        pmf = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pmf[i, j] = (5 - abs(4-i)) + (5 - abs(4-j))*1.5
        # pmf level-2
        # sum pooling it
        pmf_level2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                pmf_level2[i, j] = np.sum(pmf[2*i:2*i+2, 2*j:2*j+2])
        pmf_level3 = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                pmf_level3[i, j] = np.sum(pmf_level2[2*i:2*i+2, 2*j:2*j+2])
        # unroll the pmf levels and concatenate them
        pmf_flat = np.concatenate((pmf.flatten(), pmf_level2.flatten(), pmf_level3.flatten()))
        # make a SETensor
        octmap_tensor = SETensor(core.gfx.Context.getDevice(), [1, pmf_flat.size])
        octmap_tensor.as_torch()[0] = torch.tensor(pmf_flat)
        octmap_tensor.as_torch()[0, 64:].zero_()
        
        build_pass = PMFBuildPass()
        build_pass.octmap_tensor = octmap_tensor
        self.ctx.executeOnePass(build_pass)
        
        compute_pass = MappingTestPass()
        compute_pass.i_tensor = i_tensor
        compute_pass.d_tensor = d_tensor
        compute_pass.h_tensor = h_tensor
        compute_pass.d2_tensor = d2_tensor
        compute_pass.warp_tensor = warp_tensor
        compute_pass.warp_3d_tensor = warp_3d_tensor
        compute_pass.octmap_tensor = octmap_tensor
        self.ctx.executeOnePass(compute_pass)
        
        d_points = d_tensor.as_torch().detach().cpu().numpy()
        ax = fig.add_subplot(222)
        ax.scatter(d_points[:, 0], d_points[:, 1], c=colors)
        ax.set_aspect('equal', adjustable='box')
        
        d2_points = d2_tensor.as_torch().detach().cpu().numpy()
        ax = fig.add_subplot(223)
        ax.scatter(d2_points[:, 0], d2_points[:, 1], c=colors)
        ax.set_aspect('equal', adjustable='box')
        
        # 3d scatter plot
        ax = fig.add_subplot(224, projection='3d')
        h_points = h_tensor.as_torch().detach().cpu().numpy()
        ax.scatter(h_points[:, 0], h_points[:, 1], h_points[:, 2], c=colors)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()

        # visualize
        fig = plt.figure()
        ax = fig.add_subplot(231)
        ax.imshow(pmf, cmap='gray')
        ax = fig.add_subplot(232)
        ax.imshow(pmf_level2, cmap='gray')
        ax = fig.add_subplot(233)
        ax.imshow(pmf_level3, cmap='gray')
        ax = fig.add_subplot(234)
        w_points = warp_tensor.as_torch().detach().cpu().numpy()
        # set to uniform random variable in the same shape
        ax.hist2d(w_points[:, 0], w_points[:, 1], bins=8, cmap='gray', density=True)
        ax = fig.add_subplot(235)
        w_points = warp_tensor.as_torch().detach().cpu().numpy()
        ax.hist2d(w_points[:, 0], w_points[:, 1], bins=4, cmap='gray', density=True)
        # 3d scatter plot
        ax = fig.add_subplot(236)
        w_points = warp_tensor.as_torch().detach().cpu().numpy()
        ax.hist2d(w_points[:, 0], w_points[:, 1], bins=2, cmap='gray', density=True)
        # ax = fig.add_subplot(236, projection='3d')
        # w_points = warp_3d_tensor.as_torch().detach().cpu().numpy()[:200,:]
        # ax.scatter(w_points[:, 0], w_points[:, 1], w_points[:, 2])
        # ax.set_aspect('equal', adjustable='box')
        plt.show()

        i_tensor = None
        d_tensor = None
        h_tensor = None
        d2_tensor = None
        warp_tensor = None
        warp_3d_tensor = None
        octmap_tensor = None
        compute_pass = None
        build_pass = None
        # self.ctx.executeOnePass()
        # directly end the application
        self.end()

editor = EditorApp()
print("Script End Successfully!")

