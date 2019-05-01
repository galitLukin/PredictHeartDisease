using DataFrames
using MLDataUtils
using OptimalTrees

function runtree(big_X,big_y,tt_X,tt_y,method::String,depthrange;seed::Int=0,kwargs...)
    (tr_X, tr_y), (vl_X, vl_y) = splitobs(shuffleobs((big_X, big_y)), at=0.67)

    sample_weight = Dict(
      1 => 1.0,
      -1 => 5.0,
    )

    if method == "cart"
      push!(kwargs, (:localsearch, false))
    elseif method == "random_cart"
    elseif method == "random_hyper"
      push!(kwargs, (:ls_num_hyper_restarts, 3))
      push!(kwargs, (:local_sparsity, :all))
    end

    lnr = OptimalTrees.OptimalTreeClassifier(;normalize_X=false, treat_unknown_categoric_missing = true,kwargs...)
    grid = OptimalTrees.GridSearch(lnr,Dict(
        :max_depth => depthrange,
        :minbucket => [10],
    ))

    OptimalTrees.fit!(grid, tr_X, tr_y, vl_X, vl_y,
                      sample_weight=sample_weight,
                      sample_weight=sample_weight,
                      validation_criterion=:auc, verbose=false)
    @show grid.best_score, grid.best_params
    lnr = grid.best_lnr

    tr_auc = OptimalTrees.score(lnr, big_X, big_y, criterion=:auc)
    tt_auc = OptimalTrees.score(lnr, tt_X,  tt_y,  criterion=:auc)
    tr_mc = OptimalTrees.score(lnr, big_X, big_y, criterion=:misclassification)
    tt_mc = OptimalTrees.score(lnr, tt_X,  tt_y,  criterion=:misclassification)
    @show tr_auc, tt_auc
    @show tr_mc, tt_mc
    plotname = "tree.dot"
    OptimalTrees.writedot("tree.dot",lnr)
    run(`dot -Tpng $plotname -o $(replace(plotname, ".dot", ".png"))`)
    return tr_auc,tt_auc
end

function runTree(train, test)
    trauc,ttauc = runtree(train[1:15],train[:TenYearCHD],test[1:15],test[:TenYearCHD],"random_cart",4:5; ls_num_tree_restarts=50,ls_num_categoric_restarts=50,criterion=:entropy,)
    @show trauc
    @show ttauc
end

df = readtable("Framingham.csv", header=true, makefactors=true)
srand(1)
train, test =  splitobs(shuffleobs(df), at=0.67)
runTree(train,test)
