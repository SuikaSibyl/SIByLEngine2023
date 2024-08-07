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
        ])

        encoder = self.beginPass(rdrCtx)
        encoder.dispatchWorkgroups(4, 1, 1)
        encoder.end()

    def renderUI(self):
        pass


class EditorApp(EditorApplication):
    def __init__(self):
        super().__init__()
    
    def onInit(self):
        # draw 64 * 64 grid
        grid_size = 32
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
        i_tensor.as_torch()[:, 0] = torch.tensor(X)
        i_tensor.as_torch()[:, 1] = torch.tensor(Y)

        compute_pass = MappingTestPass()
        compute_pass.i_tensor = i_tensor
        compute_pass.d_tensor = d_tensor
        compute_pass.h_tensor = h_tensor
        compute_pass.d2_tensor = d2_tensor
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

        i_tensor = None
        d_tensor = None
        h_tensor = None
        d2_tensor = None
        compute_pass = None
        # self.ctx.executeOnePass()
        # directly end the application
        self.end()

editor = EditorApp()
print("Script End Successfully!")

