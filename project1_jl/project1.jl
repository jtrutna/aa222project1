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
    # XXX: Must keep signature since autograder calls into this even though we need optimization
    #      history for writeup of assignment.
    #history = _optimize(f, g, x0, n)
    #history[end, :]

    method = NelderMead(f, length(x0))
    optimize(method)
end

function _optimize(f, g, x0, n)
    method = DecayingGradientDescent(f, g, 1.0, 0.75)
    steps = convert(Int, n/method.step_cost)
    x0 = reshape(x0, 1, length(x0)) # Getting some inconsistencies
    history = vcat(x0, zeros(eltype(x0), steps, length(x0)))
    for i in 1:steps
        history[i+1,:] = step(method, history[i,:])
    end
    history
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

mutable struct NelderMead
    f::Function
    ϵ::Real
    α::Real
    β::Real
    γ::Real
    S::Vector{}
    step_cost::Int

    function NelderMead(f, dimensions)
        tmp = Matrix{Float64}(I(dimensions))
        S = [tmp[:, i] for i in 1:dimensions]
        new(f, 0.1, 1.0, 2.0, 0.5, S, dimensions+2)
    end
end

function optimize(m::NelderMead)
    f, S, ϵ, α, β, γ = m.f, m.S, m.ϵ, m.α, m.β, m.γ
    
    Δ, y_arr = Inf, f.(S)
    while Δ > ϵ
        p = sortperm(y_arr)
        S, y_arr = S[p], y_arr[p]
        xl, yl = S[1], y_arr[1]
        xh, yh = S[end], y_arr[end]
        xs, ys = S[end-1], y_arr[end-1]
        xm = mean(S[1:end-1])
        xr = xm + α*(xm - xh)
        yr = f(xr)

        if yr < yl
            xe = xm + β*(xr-xm)
            ye = f(xe)
            S[end], y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
        elseif yr ≥ ys
            if yr < yh
                xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
            end
            xc = xm + γ*(xh - xm)
            yc = f(xc)
            if yc > yh
                for i in 2 : length(y_arr)
                    S[i] = (S[i] + xl)/2
                    y_arr[i] = f(S[i])
                end
            else
                S[end], y_arr[end] = xc, yc
            end
        else
            S[end], y_arr[end] = xr, yr
        end

        Δ = std(y_arr, corrected=false)
    end
    return S[argmin(y_arr)]
end


