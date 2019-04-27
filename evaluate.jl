using DataFrames

function evaluate(w, b, z, c, train, test, measures, modelName)
    df = [train, test]
    type = ["train", "test"]
    for i in 1:2
        n = size(df[i],1)
        y = df[i][:,16]
        x = df[i][:,1:15]
        p = size(x,2)
        accuracyVal = accuracy(n, p, w, b, x, y, measures)
        precisionVal = precision(n, p, w, b, x, y, measures)
        recallVal = recall(n, p, w, b, x, y, measures)
        # aucVal = auc(n, p, w, b, x, y, measures)
        penalty(n, z, c)
        push!(measures,[modelName,type[i],accuracyVal,precisionVal,recallVal])
    end
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

# function auc(n, p, w, b, x, y, measures)
#
# end

function penalty(n, z, c)
    pen =  zeros(n)
    for i in 1:n
        pen[i] = z[i]*c[i]
    end
    penalty = DataFrame( penalty = pen)
    writetable("penalty.csv",penalty)
end
