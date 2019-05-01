using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils
include("evaluate.jl")

function solveNominal(c, df)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, z[1:n] >= 0)

    for i in 1:n
        x_ = convert(Array, x[i,:])[1,:]
        @constraint(model, z[i] >= 1 - y[i]*(sum(w .* x_) - b))
    end
    @objective(model, Min, sum(z.*c))
    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    @show objective
    return b, w, objective
end

function runNominalModel(ctrain, train, test, measures)
    b, w, objective = solveNominal(cTrain,train)
    measures = evaluate(w, b, train, "train", measures, "Nominal", 0, objective)
    measures = evaluate(w, b, test, "test", measures, "Nominal", 0, objective)
    return measures
end
