using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

function solveLabel(df, gamma)
    model = Model(solver=GurobiSolver(OutputFlag=0))
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)
    gamma = floor(gamma * n)

    @variable(model, b)
    @variable(model, w[1:p])

    @variable(model, q >= 0)
    @variable(model, r[1:n] >= 0)
    @variable(model, xsi[1:n] >= 0)
    @variable(model, theta[1:n] >= 0)

    @variable(model, s[1:n], Bin)
    @variable(model, t[1:n], Bin)

    M = 100000
    for i in 1:n
        x_ = convert(Array, x[i,:])[1,:]
        @constraint(model, q + r[i] >= theta[i] - xsi[i])
        @constraint(model, xsi[i] >= 1 - y[i]*(sum(w .* x_) - b))
        @constraint(model, xsi[i] <= 1 - y[i]*(sum(w .* x_) - b) + M*(1-s[i]))
        @constraint(model, xsi[i] <= M*s[i])
        @constraint(model, theta[i] >= 1 + y[i]*(sum(w .* x_) - b))
        @constraint(model, theta[i] <= 1 + y[i]*(sum(w .* x_) - b) + M*(1-t[i]))
        @constraint(model, theta[i] <= M*t[i])
    end

    @objective(model, Min,gamma*q+ sum(xsi[i]+r[i] for i=1:n))

    solve(model)
    b = getvalue(b)
    w = getvalue(w)
    q = getvalue(q)
    r = getvalue(r)

    objective = getobjectivevalue(model)
    @show objective

    return b, w, objective
end

function downsample(df)
    res = df[df[:TenYearCHD] .== 1,:]
    neg = df[df[:TenYearCHD] .== -1,:]
    r = rand(1,size(neg,1))
    for i in 1:size(neg,1)
        if r[i] >= 0.8
            push!(res, convert(Array,neg[i,:]))
        end
    end
    return res
end

function runLabelModel(train, test, measures, gamma)
    dtrain = downsample(train)
    dtest = downsample(test)
    b, w, objective = solveLabel(dtrain, gamma)
    measures = evaluate(w, b, dtrain, "train", measures, "labelError_$gamma", gamma, objective)
    measures = evaluate(w, b, dtest, "test", measures, "labelError_$gamma", gamma, objective)
    return measures
end
