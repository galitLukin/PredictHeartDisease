using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
include("evaluate.jl")

function _solve(c, df)
    model = RobustModel(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, z[1:n] >= 0)
    @objective(model, Min, sum(z[i]*c[i] for i=1:n))
    for i in 1:n
        @constraint(model, z[i] >= 1 - y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
    end

    #Uncertinety constraints
    # @uncertain(model, z[i=1:n])
    # @uncertain(model, r[i=1:n])
    # @constraint(model, equality[i=1:n], r[i] == meu[i] + sigma[i] * z[i])

    # @constraint(model, norm(z, Inf) <= 1)
    # @constraint(model, norm(z, 2) <= theta)
    #
    # @constraint(model, sum(x[i] * r[i] for i=1:n) >= t)


    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    z = getvalue(z)
    objective = getobjectivevalue(model)
    println(objective)
    return b, w, z, objective
end

function buildC(df)
    c = copy(df[:,16])
    for i in 1:size(c,1)
        if df[i,16] == 1
            c[i] = 5
        else
            c[i] = 1
        end
    end
    return c
end

df = readtable("Framingham.csv", header=true, makefactors=true)
c = buildC(df)
b, w, z, objective = _solve(c,df)
measures = DataFrame( model = String[],accuracy = Float64[], precision = Float64[], recall = Float64[])
evaluate(w, b, z, c, df, measures, "Nominal")
