using JuMP, JuMPeR, Gurobi, DataFrames, Distributions
using MLDataUtils

include("nominal.jl")
include("labErrors.jl")
include("labelError.jl")

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

function normalizeData(df)
    n = size(df,1)
    ndf = DataFrame(male = zeros(n), age = zeros(n), education = zeros(n), currentSmoker = zeros(n), cigsPerDay = zeros(n), BPMeds = zeros(n), prevalentStroke = zeros(n), prevalentHyp = zeros(n), diabetes = zeros(n), totChol = zeros(n), sysBP = zeros(n), diaBP = zeros(n), BMI = zeros(n), heartRate = zeros(n), glucose = zeros(n), TenYearCHD = zeros(n))
    for i in 1:size(df,2)
        ndf[:,i] = df[:,i] * 1.0
    end
    return ndf
end

df = readtable("Framingham.csv", header=true, makefactors=true)
srand(1)
train, test =  splitobs(shuffleobs(df), at=0.67)
cTrain = buildC(train)
train = normalizeData(train)
test = normalizeData(test)
measures = DataFrame( model = String[], trainORtest = String[], param = Float64[], accuracy = Float64[], precision = Float64[], recall = Float64[], objective = Float64[])
# Nominal
measures = runNominalModel(cTrain, train, test, measures)
# Feature Errors
for err in 0:0.5:5
    measures = runLabModel(cTrain, train, test, measures, err)
end
# doctors error
for gamma in 0:0.01:0.1
    measures = runLabelModel(train, test, measures, gamma)
end

writetable("measures.csv",measures)
