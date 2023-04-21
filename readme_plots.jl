using Plots

# Include the other relevant files:
include(joinpath("project1_jl", "helpers.jl"))
include(joinpath("project1_jl", "project1.jl"))

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
    #xrange = mins[1]:(maxs[1] - mins[1])/100:maxs[1]
    #yrange = mins[2]:(maxs[2] - mins[2])/100:maxs[2]

    xrange = -2:0.1:2
    yrange = -2.0:0.1:3.0

    # Create a contour plot of the function
    p = contourf(xrange, yrange, (x,y)->f([x,y]), title="f(x,y)", levels=100, c=:viridis)

    overlay_quiver(g, -2:1.0/5:2, -2.0:1.0/5:3.0, "quiver.png")

    plot!(xlims=(-2, 2))
    plot!(ylims=(-2, 2))
    savefig("plot.png")
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

function hw_readme_rosenbrack(f, g, x0, n, gif_filename)
    history = _optimize(f, g, x0, n)
    animate_history(f, g, history, gif_filename)
end

function hw_readme_convergence(f, g, x0, n, plot_filename)
    history = _optimize(f, g, x0, n)
    yvalues = [f([xy[1], xy[2]]) for xy in eachrow(history)]
    plot(1:length(yvalues), yvalues, seriestype=:line, xlabel="Iteration", ylabel="Function Value", title="Convergence Plot", legend=false)
    savefig(plot_filename)
end

f, g, _, n = PROBS["simple1"]
hw_readme_rosenbrack(f, g, [-1.0, -1.0], 20, "rosenbrock_1.gif")
hw_readme_rosenbrack(f, g, [1.5, -1.5], 20, "rosenbrock_2.gif")
hw_readme_rosenbrack(f, g, [0.0, 0.0], 20, "rosenbrock_3.gif")

f, g, x0, n = PROBS["simple1"]
hw_readme_convergence(f, g, x0(), n, "convergence_rosenbrock.png")