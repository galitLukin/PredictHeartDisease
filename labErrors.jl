using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

function solveLab(c, df, rho)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, xsi[1:n] >= 0)

    srand(1)
    r = rand(1,n)
    for i in 1:n
        x_ = convert(Array, x[i,:])[1,:]
        @constraint(model, y[i]*(sum(w .* x_) - b) - rho*norm(w,2) >= 1 - xsi[i])
    end

    @objective(model, Min, sum(c.*xsi))

    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    @show objective
    return b, w, objective
end

function runLabModel(ctrain, train, test, measures, rho)
    b, w, objective = solveLab(cTrain, train, rho)
    measures = evaluate(w, b, train, "train", measures, "labErrors_$rho",rho, objective)
    measures = evaluate(w, b, test, "test", measures, "labErrors_$rho",rho, objective)
    return measures
end
