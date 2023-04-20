#=
    If you're going to include files, please do so up here. Note that they
    must be saved in project1_jl and you must use the relative path
    (not the absolute path) of the file in the include statement.

    [Good]  include("somefile.jl")
    [Bad]   include("/pathto/project1_jl/somefile.jl")
=#

# Example
# include("myfile.jl")

using LinearAlgebra
using Plots

"""
    optimize(f, g, x0, n, prob)

Arguments:
    - `f`: Function to be optimized
    - `g`: Gradient function for `f`
    - `x0`: (Vector) Initial position to start from
    - `n`: (Int) Number of evaluations allowed. Remember `g` costs twice of `f`
    - `prob`: (String) Name of the problem. So you can use a different strategy for each problem. E.g. "simple1", "secret2", etc.

Returns:
    - The location of the minimum
"""
function optimize(f, g, x0, n, prob)
    method = DecayingGradientDescent(f, g, 1.0, 0.75)
    steps = convert(Int, n/method.step_cost)
    x0 = reshape(x0, 1, length(x0)) # Getting some inconsistencies
    history = vcat(x0, zeros(eltype(x0), steps, length(x0)))
    for i in 1:steps
        history[i+1,:] = step(method, history[i,:])
    end
    #animate_history(f, g, history, "$prob.gif")
    history[end, :]
end

struct ConjugateGradientDescent
    f::Function
    g::Function
    α::Real
    step_cost::Int

    function ConjugateGradientDescent(f, g, α)
        new(f, g, α, 2)
    end
end

function step(M::ConjugateGradientDescent, x)
    x, f, g, α = M.f, M.g, M.α
    β = max(0, dot(g′, g′ - g)/(g⋅g))
    d′ = -g′ + β*d
    
    x′ = line_search(f, x, d′)
    M.d, M.g = d′, g′
    return x
end

function graph(f, g, history, output_filename)
    mins = map(minimum, eachcol(history))
    maxs = map(maximum, eachcol(history))

    # Define the range of values for the x and y axes
    xrange = mins[1]:(maxs[1] - mins[1])/100:maxs[1]
    yrange = mins[2]:(maxs[2] - mins[2])/100:maxs[2]

    z = ((a,b)->f([a,b])).(xrange', yrange)

    # Create a contour plot of the function
    contour(xrange, yrange, z, xlabel="x", ylabel="y", title="f(x,y)")

    plot!(history[:, 1], history[:, 2], color="red", label="history")

    scatter!([point[1] for point in history], [point[2] for point in history], label="", markersize=3)
    for (i, point) in enumerate(history)
        annotate!(point[1], point[2], text(string(i), 8, :black))
    end

    savefig(output_filename)
end

function animate_history(f, g, history, output_filename)
    mins = map(minimum, eachcol(history))
    maxs = map(maximum, eachcol(history))

    # Define the range of values for the x and y axes
    xrange = mins[1]:(maxs[1] - mins[1])/100:maxs[1]
    yrange = mins[2]:(maxs[2] - mins[2])/100:maxs[2]

    xrange = -2:0.1:2
    yrange = -2.0:0.1:3.0

    # Create a contour plot of the function
    p = contourf(xrange, yrange, (x,y)->f([x,y]), title="f(x,y)", levels=100, c=:viridis)

    overlay_quiver(g, -2:0.2:2, -2.0:0.2:3.0, "quiver.png")

    # Create and save the animation as a gif
    anim = @animate for i in 1:size(history, 1)
        plot!(history[1:i,1], history[1:i,2], c=:red, lw=1.5, m=:circle, ms=5)
    end
    gif(anim, output_filename, fps=10)
end

function overlay_quiver(g::Function, xrange, yrange, output_filename)
    # Evaluate the vector field at each point in the grid and prepare arguments for quiver function
    tmp = [[x, y, g([x,y])] for x in xrange, y in yrange]
    X = [p[1] for p in tmp]
    Y = [p[2] for p in tmp]
    U = [p[3][1] for p in tmp]
    V = [p[3][2] for p in tmp]

    # Normalize (and shrink) the vectors
    magnitudes = 5*sqrt.(U.^2 .+ V.^2)
    U_norm = U ./ magnitudes
    V_norm = V ./ magnitudes

    # Create the quiver plot with arrows at each (x,y) coordinate
    quiver!(X, Y, quiver=(U_norm, V_norm), scale=:identity, aspect_ratio=:equal, color=:white)
end

struct NaiveGradientDescent
    f::Function
    g::Function
    α::Real
    step_cost::Int

    function NaiveGradientDescent(f, g, α)
        new(f, g, α, 2)
    end
end

function step(m::NaiveGradientDescent, x)
    g = m.g(x)
    x - m.α * g / norm(g)
end


mutable struct DecayingGradientDescent
    f::Function
    g::Function
    α::Real
    γ::Real
    step_cost::Int

    function DecayingGradientDescent(f, g, α, γ)
        new(f, g, α, γ, 2)
    end
end

function step(m::DecayingGradientDescent, x)
    g = m.g(x)
    m.α *= m.γ
    x - (m.α / m.γ) * g / norm(g)
end

mutable struct SharedBudget available end
struct BudgetViolation <: Exception end
function make_budgeted(budget::SharedBudget, fx::Function, cost::Real)
    function wrapped(args...)
        @sync begin
            if budget.available >= cost
                budget.available -= cost
            else
                throw(BudgetViolation)
            end
        end
        fx(args...)
    end
end

budget = SharedBudget(20)
f = make_budgeted(budget, f, 1)
g = make_budgeted(budget, g, 2)

