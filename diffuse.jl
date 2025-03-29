using DiffusionKernels
using Oceananigans
using Reactant
using GLMakie

# arch = CPU()
arch = Oceananigans.Architectures.ReactantState()

@info "Generating model..."
model = DiffusionKernels.Model(arch, size=(32, 32, 32))

@info "Before set!: $(maximum(parent(model.c)))"
set!(model.c, (x, y, z) -> randn())

@info "After set!: $(maximum(parent(model.c)))"

Nt = 1000
if arch isa Oceananigans.Architectures.ReactantState
    rNt = ConcreteRNumber(Nt)
    @info "Compiling loop..."
    #rloop! = @compile DiffusionKernels.loop!(model, rNt)
    rstep! = @compile DiffusionKernels.time_step!(model)

    @info "Running loop on $arch..."
    for n = 1:Nt
        rstep!(model)
    end
else
    @info "Running loop on $arch..."
    DiffusionKernels.loop!(model, Nt)
end

@info "After set!: $(maximum(parent(model.c)))"
