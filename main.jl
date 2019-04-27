using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

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
srand(1)
train, test =  splitobs(shuffleobs(df), at=0.67)
cTrain = buildC(train)
measures = DataFrame( model = String[], trainORtest = String[], accuracy = Float64[], precision = Float64[], recall = Float64[])
b, w, z, objective = _solve(cTrain,train)
measures = evaluate(w, b, train, "train", measures, "Nominal")
measures = evaluate(w, b, test, "test", measures, "Nominal")
writetable("measures.csv",measures)
