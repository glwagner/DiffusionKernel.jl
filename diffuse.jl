using DiffusionKernels

arch = CPU()
model = DiffusionKernels.Model(arch, 32, 32, 32)

set!(c, (x, y, z) -> randn())
@show maximum(c)

DiffusionKernels.loop!(model, 100)
@show maximum(c)
