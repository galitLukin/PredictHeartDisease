using DataFrames

function evaluate(w, b, df, trainORtest, measures, modelName, err, objective)
    n = size(df,1)
    y = df[:,16]
    x = df[:,1:15]
    p = size(x,2)
    accuracyVal = accuracy(n, p, w, b, x, y, measures)
    precisionVal = precision(n, p, w, b, x, y, measures)
    recallVal = recall(n, p, w, b, x, y, measures)
    auc(n, p, w, b, x, y, measures, trainORtest, modelName)
    push!(measures,[modelName,trainORtest,err,accuracyVal,precisionVal,recallVal, objective])
    return measures
end

function accuracy(n, p, w, b, x, y, measures)
    correct = 0
    for i in 1:n
        if y[i]*(sum(w[j] * x[i,j] for j=1:p) - b) > 0
            correct = correct + 1
        end
    end
    return correct/n
end

function precision(n, p, w, b, x, y, measures)
    tp = 0
    fp = 0
    for i in 1:n
        if y[i]*(sum(w[j] * x[i,j] for j=1:p) - b) > 0 && y[i] > 0
            tp = tp + 1
        end
        if y[i]*(sum(w[j] * x[i,j] for j=1:p) - b) <= 0 && y[i] < 0
            fp = fp + 1
        end
    end
    return tp/(tp+fp)
end

function recall(n, p, w, b, x, y, measures)
    tp = 0
    fn = 0
    for i in 1:n
        if y[i]*(sum(w[j] * x[i,j] for j=1:p) - b) > 0 && y[i] > 0
            tp = tp + 1
        end
        if y[i]*(sum(w[j] * x[i,j] for j=1:p) - b) <= 0 && y[i] > 0
            fn = fn + 1
        end
    end
    return tp/(tp+fn)
end

function auc(n, p, w, b, x, y, measures, trainORtest, modelName)
    prediction =  zeros(n)
    for i in 1:n
        if (sum(w[j] * x[i,j] for j=1:p) - b) < 0
            prediction[i] = -1
        else
            prediction[i] = 1
        end
    end
    auc = DataFrame( prediction = prediction, y = y)
    writetable("auc_$trainORtest$modelName.csv",auc)
end
