clear; clc;
load('usps.mat')

if exist('y', 'var')
    Y = y;
end
if exist('fea', 'var')
    X = fea;
end
clear fea;
if exist('gnd', 'var')
    Y = gnd;
end
clear gnd;
Y = Y(:);

nCluster = length(unique(Y));
nSmp = length(Y);

batch_size = 1000;
knn_size = 5;
nOrder = 9;

Si = ConstructWHuge(X, knn_size, batch_size);
Ts = cell(1, nOrder);
Ts{1, 1} = speye(size(X, 1));
Ts{1, 2} = Si;
for jOrder = 3:nOrder
    tmp1 = multi_blockSize(Si, Ts{1, jOrder-1});
    Ts{1, jOrder} = sparse(2 * tmp1 - Ts{1, jOrder-2});
end

TXs = cell(1, nOrder);
for iOrder = 1:nOrder
    TXs{1, iOrder} = multi_blockSize(Ts{1, iOrder}, X);
end

A1 = zeros(nOrder, nOrder);
for iOrder = 1:nOrder
    for jOrder = iOrder:nOrder
        e2_ij = sum(sum(TXs{1, iOrder} .* TXs{1, jOrder}));
        A1(iOrder, jOrder) = e2_ij;
        A1(jOrder, iOrder) = e2_ij;
    end
end

function fitness = pso_fitness(params, X, Y, nCluster, knn_size, nOrder, TXs, A1)
    totalSamples = size(X, 1);
    disp(params)
    initCentroids = max(1, min(totalSamples, round(params * totalSamples)));
    disp(initCentroids)

    [label, ~, ~] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1, 'initCentroid', initCentroids);

    res = my_eval_y(label, Y);
    fitness = -res(1);
end

function fitness = fitnessOrder(params, X, Y, nCluster, batch_size, knn_size, Best_centroid)
    nOrder = round(params(1));
    disp(["nOrder" nOrder])
    totalSamples = size(X, 1);
    initCentroids = max(1, min(totalSamples, round(Best_centroid * totalSamples)));

    Si = ConstructWHuge(X, knn_size, batch_size);
    Ts = cell(1, nOrder);
    Ts{1, 1} = speye(size(X, 1));
    Ts{1, 2} = Si;
    for jOrder = 3:nOrder
        tmp1 = multi_blockSize(Si, Ts{1, jOrder-1});
        Ts{1, jOrder} = sparse(2 * tmp1 - Ts{1, jOrder-2});
    end

    TXs = cell(1, nOrder);
    for iOrder = 1:nOrder
        TXs{1, iOrder} = multi_blockSize(Ts{1, iOrder}, X);
    end

    A1 = zeros(nOrder, nOrder);
    for iOrder = 1:nOrder
        for jOrder = iOrder:nOrder
            e2_ij = sum(sum(TXs{1, iOrder} .* TXs{1, jOrder}));
            A1(iOrder, jOrder) = e2_ij;
            A1(jOrder, iOrder) = e2_ij;
        end
    end

    [label, ~, ~] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1, 'initCentroid', initCentroids);

    res = my_eval_y(label, Y);
    fitness = res(1);
    disp(["acc" fitness])
end

LB = zeros(1, nCluster);
UB = ones(1, nCluster);
dim = nCluster;

options = optimoptions('particleswarm', 'SwarmSize', 20, 'MaxIterations', 100);

[Best_centroid, Best_score1] = particleswarm(...
    @(params) pso_fitness(params, X, Y, nCluster, knn_size, nOrder, TXs, A1), ...
    dim, LB, UB, options);

LB = 1;
UB = 9;
nOrder_values = LB:UB;

Best_nOrder = 0;
Best_score = -Inf;

for nOrder = nOrder_values
    fitness = fitnessOrder(nOrder, X, Y, nCluster, batch_size, knn_size, Best_centroid);

    if fitness > Best_score
        Best_score = fitness;
        Best_nOrder = nOrder;
    end
end

totalSamples = size(X, 1);
optimal_centroids = max(1, min(totalSamples, round(Best_centroid * totalSamples)));
disp(['Optimal nOrder: ', num2str(Best_nOrder)]);
disp('Centroid optimal: ');
disp(optimal_centroids);
disp(['Fitness terbaik: ', num2str(-Best_score)]);
