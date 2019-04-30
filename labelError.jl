using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

function solveLabel(c, df)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)
    gamma = gamma * n

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, z[1:n] >= 0)

    @variable(model, u >= 0)
    @variable(model, v[1:n] >= 0)
    @variable(model, q[1:n] >= 0)
    #@variable(model, cy[1:n])
    @variable(model, err[1:n], Bin)
    @variable(model, erru[1:n] >= 0)
    @variable(model, errz[1:n] >= 0)

    for i in 1:n
        @constraint(model, u + v[i] >= q[i] - z[i])
        @constraint(model, z[i] >= 1 - y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
        @constraint(model, q[i] >= 1 + y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
        #@constraint(model, cy[i] == c[i] - 4 * y[i] * err[i])
        @constraint(model, v[i] >= 1 - 100000*(1-err[i]))
        @constraint(model, erru[i] <= 1*err[i])
        @constraint(model, erru[i] <= u)
        @constraint(model, erru[i] >= u + 1*(err[i]-1))
        @constraint(model, errz[i] <= 1*err[i])
        @constraint(model, errz[i] <= z[i])
        @constraint(model, errz[i] >= z[i] + 1*(err[i]-1))
    end

    @objective(model, Min, sum( c[i] * (gamma * u + v[i] + z[i]) - 4 * y[i] * ( gamma * erru[i] + v[i] + errz[i] ) for i=1:n))

    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    @show objective
    @show b
    @show w
    return b, w, objective
end

function runLabelModel(ctrain, train, measures, gamma)
    b, w, objective = solveLabel(cTrain,train, gamma)
    measures = evaluate(w, b, train, "train", measures, "labelError_$gamma")
    measures = evaluate(w, b, test, "test", measures, "labelError_$gamma")
    return measures
end
