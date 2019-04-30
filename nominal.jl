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
        @constraint(model, z[i] >= 1 - y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
    end
    @objective(model, Min, sum(z[i]*c[i] for i=1:n))
    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    @show objective
    @show b
    @show w
    return b, w, objective
end

function runNominalModel(ctrain, train, measures)
    for i in 11:13
        s = sum(train[:,i])
        train[:,i] = train[:,i]/s
    end
    b, w, objective = solveNominal(cTrain,train)
    measures = evaluate(w, b, train, "train", measures, "Nominal")
    measures = evaluate(w, b, test, "test", measures, "Nominal")
    return measures
end
