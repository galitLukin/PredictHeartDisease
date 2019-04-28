using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

include("evaluate.jl")

function _solve(c, df, modelName)
    model = RobustModel(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)

    @variable(model, b)
    @variable(model, w[1:p])
    @variable(model, z[1:n] >= 0)

    if modelName == "Nominal"
        for i in 1:n
            @constraint(model, z[i] >= 1 - y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
        end
        @objective(model, Min, sum(z[i]*c[i] for i=1:n))
    elseif modelName == "DocError"
        @variable(model, u)
        @variable(model, t)
        @variable(model, v[1:n] >= 0)
        for i in 1:n
            @constraint(model, z[i] >= 1 + y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
            @constraint(model, v[i] >= 1 - y[i]*(sum(w[j] * x[i,j] for j=1:p) - b))
            @constraint(model, u >= c[i]*(z[i]-v[i]))
        end
        @constraint(model, t >= sum(c[i]*v[i] for i=1:n) + u)
        @objective(model, Min, t)
    end

    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    objective = getobjectivevalue(model)
    println(objective)
    println(b)
    println(w)
    return b, w, objective
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

function runModel(ctrain, train, modelName, measures)
    b, w, objective = _solve(cTrain,train, modelName)
    measures = evaluate(w, b, train, "train", measures, modelName)
    measures = evaluate(w, b, test, "test", measures, modelName)
    return measures
end

df = readtable("Framingham.csv", header=true, makefactors=true)
srand(1)
train, test =  splitobs(shuffleobs(df), at=0.67)
cTrain = buildC(train)
measures = DataFrame( model = String[], trainORtest = String[], accuracy = Float64[], precision = Float64[], recall = Float64[])
# Nominal
measures = runModel(cTrain, train, "Nominal", measures)
# Lab Errors
# measures = runModel(cTrain, train, "LabErrors", measures)
# Lies
# measures = runModel(cTrain, train, "Lies", measures)
# doctors error
measures = runModel(cTrain, train, "DocError", measures)

writetable("measures.csv",measures)
