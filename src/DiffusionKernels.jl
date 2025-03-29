module DiffusionKernels

using Reactant
using KernelAbstractions: @kernel, @index
using Oceananigans

@propagate_inbounds δ²x(i, j, k, c) = c[i+1, j, k] - 2 * c[i, j, k] + c[i-1, j, k]
@propagate_inbounds δ²y(i, j, k, c) = c[i, j+1, k] - 2 * c[i, j, k] + c[i, j-1, k]
@propagate_inbounds δ²z(i, j, k, c) = c[i, j, k+1] - 2 * c[i, j, k] + c[i, j, k-1]

@kernel function _diffusion!(G, c)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] = δ²x(i, j, k, c) +
                           δ²y(i, j, k, c) +
                           δ²z(i, j, k, c)
end

@kernel function _advance!(G, c, dt)
    i, j, k = @index(Global, NTuple)
    @inbounds c[i, j, k] += dt * G[i, j, k]
end

struct Model{Gr, T, F, DT}
    grid :: Gr
    G :: T
    c :: F
    dt :: DT
end

function Model(arch, sz)
    x = (0, sz[1])
    y = (0, sz[2])
    z = (0, sz[3])
    topology = (Bounded, Bounded, Bounded)
    grid = RectilinearGrid(arch, size=sz; x, y, z, topology)
    c = CenterField(grid)
    G = CenterField(grid)
    dt = convert(eltype(grid), 1e-3) # conservative
    return Model(grid, G, c, dt)
end

function time_step!(model)
    grid = model.grid
    arch = grid.architecture
    Oceananigans.BoundaryConditions.fill_halo_regions!(model.c)
    Oceananigans.Utils.launch!(arch, grid, :xyz, _diffusion!, model.G, model.c)
    Oceananigans.Utils.launch!(arch, grid, :xyz, _advance!, model.c, model.c, model.dt)
    return nothing
end

function loop!(model, Nt)
    @trace for n = 1:Nt
        time_step!(model)
    end
    return nothing
end

end # module

