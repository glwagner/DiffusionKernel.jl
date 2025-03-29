module DiffusionKernels

using Adapt
using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans

Base.@propagate_inbounds δ²x(i, j, k, c) = c[i+1, j, k] - 2 * c[i, j, k] + c[i-1, j, k]
Base.@propagate_inbounds δ²y(i, j, k, c) = c[i, j+1, k] - 2 * c[i, j, k] + c[i, j-1, k]
Base.@propagate_inbounds δ²z(i, j, k, c) = c[i, j, k+1] - 2 * c[i, j, k] + c[i, j, k-1]

@kernel function _diffusion!(G, c)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = δ²x(i, j, k, c) +
                           δ²y(i, j, k, c) +
                           δ²z(i, j, k, c)
end

@kernel function _advance!(c, G, dt)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k] += dt * G[i, j, k]
end

mutable struct Model{Gr, F, DT}
    grid :: Gr
    G :: F
    c :: F
    dt :: DT
end

Adapt.adapt_structure(to, model::Model) = Model(
    adapt(to, model.grid),
    adapt(to, model.G),
    adapt(to, model.c),
    adapt(to, model.dt),
)

function Model(arch; size)
    Nx, Ny, Nz = size
    topology = (Bounded, Bounded, Bounded)

    dx = 360 / Nx
    dy = 160 / Nx
    dz = convert(Oceananigans.defaults.FloatType, 1)
    longitude = collect(0:dx:360)
    latitude = collect(-80:dy:80)
    z = collect(0:dz:Nz)
    grid = LatitudeLongitudeGrid(arch; size, longitude, latitude, z, topology)

    # x = (0, size[1])
    # y = (0, size[2])
    # z = (0, size[3])
    # This might be better but we do not support RectilinearGrid on ReactantState:
    # grid = RectilinearGrid(arch; size, x, y, z, topology)
    
    G = CenterField(grid)
    c = CenterField(grid)
    dt = convert(eltype(grid), 5e-2) # conservative

    return Model(grid, G, c, dt)
end

function time_step!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.BoundaryConditions.fill_halo_regions!(model.c)
    Oceananigans.Utils.launch!(arch, grid, :xyz, _diffusion!, model.G, model.c)
    Oceananigans.Utils.launch!(arch, grid, :xyz, _advance!, model.c, model.G, model.dt)
    return nothing
end

function loop!(model, Nt)
    @trace for n = 1:Nt
        time_step!(model)
    end
    return nothing
end

end # module

