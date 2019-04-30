using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

function solveLab(c, df, epsilon)
    model = RobustModel(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, z[1:n] >= 0)
    @uncertain(model, lab[1:n,1:p])

    for i in 1:n
        @constraint(model, z[i] >= 1 - y[i]*(sum{w[j] * (x[i,j] + lab[i,j]), j=1:p} - b))
    end
    for i in 1:n
        for j in 1:9
            @constraint(model, lab[i,j] == 0)
        end
    end
    @constraint(model, lab .<= epsilon)
    @constraint(model, lab .>= -epsilon)
    @objective(model, Min, sum{z[i]*c[i], i=1:n})

    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    @show objective
    @show b
    @show w
    return b, w, objective
end

function runLabModel(ctrain, train, measures, epsilon)
    b, w, objective = solveLab(cTrain,train, epsilon)
    measures = evaluate(w, b, train, "train", measures, "labErrors_$epsilon")
    measures = evaluate(w, b, test, "test", measures, "labErrors_$epsilon")
    return measures
end
