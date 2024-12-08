clear; clc;
disp('loading');

load caltech101.mat;

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
knn_size = 9;
disp('construct w');
Si = ConstructWHuge(X, 5, batch_size);

initCentroid = [1, 8677, 1, 8677, 1, 1, 1, 8677, 5084, 1, 1, 8677, 2979, 8677, 8677, 1, 4942, 8677, 8677, 4095, 1, 1, 8677, 8677, 8677, 1, 8677, 8677, 8677, 1, 1, 6309, 6458, 8677, 8677, 1, 8677, 1, 8677, 1, 1, 887, 8677, 8677, 8677, 4682, 4816, 1, 8677, 1, 8677, 8677, 8677, 5291, 8677, 8677, 1, 8677, 1019, 8677, 2441, 3689, 1, 8677, 489, 1, 1, 3972, 1, 8677, 647, 8677, 8677, 5796, 8677, 1, 8677, 1, 8677, 8677, 8677, 2869, 1, 8677, 1, 8677, 8677];

accuracies = zeros(1, 9);

for nOrder = 1:9
    disp(['Processing nOrder = ', num2str(nOrder)]);
    
    Ts = cell(1, nOrder);
    Ts{1, 1} = speye(nSmp);
    if nOrder > 1
        Ts{1, 2} = Si;
        for jOrder = 3:nOrder
            tmp1 = multi_blockSize(Si, Ts{1, jOrder-1});
            Ts{1, jOrder} = sparse(2 * tmp1 - Ts{1, jOrder-2});
        end
        clear tmp1;
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

    [~, o_2] = eig(A1);
    disp(['min eigval is ', num2str(min(diag(o_2)))]);

    disp('cgfkm');
    [label, objHistory, beta] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1, 'initCentroid', initCentroid);

    result_10 = my_eval_y(label, Y);
    acc = result_10(1);

    accuracies(nOrder) = acc;
end

disp(num2str(accuracies, '%.4f\t'));
