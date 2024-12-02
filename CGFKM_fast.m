function [label, objHistory, beta] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1, varargin)

initCentroid = 0;
    
% Cari pasangan nama-nilai
for k = 1:2:length(varargin)
    if strcmp(varargin{k}, 'initCentroid')
        initCentroid = varargin{k+1};
        break;
    end
end

if ~exist('nOrder', 'var') || isempty(nOrder)
    nOrder = 5;
end

if ~exist('knn_size', 'var') || isempty(knn_size)
    knn_size = 10;
end

[nSmp, nFea] = size(X);
opt = [];
opt.Display = 'off';


[label, center] = litekmeans(X, nCluster, 'MaxIter', 50, 'Replicates', 10, 'initCentroid', initCentroid);
beta = ones(1, nOrder)/nOrder;

objHistory = [];
maxiter = 2;
myeps = 1e-5;
for iter = 1:maxiter
    %*********************************************************************
    % Update beta
    %*********************************************************************
    UV = center(label, :); % nSmp * nFea
    fi = zeros(nOrder, 1);
    for iOrder = 1:nOrder
        fi(iOrder) = sum(sum( TXs{1, iOrder} .* UV ));
    end
    Hi = A1;
    
    %     max_val = max(max(Hi(:)), max(fi(:)));
    %     fi = fi/max_val;
    %     Hi = Hi/max_val;
    Hi = (Hi + Hi')/2;
    lb = zeros(nOrder, 1);
    ub = ones(nOrder, 1);
    Aeq = ones(1, nOrder);
    beq = 1;
    beta_old = beta;
    beta = quadprog(Hi,-fi,[],[],Aeq,beq,lb,ub,beta_old,opt);
    obj = compute_obj(X, TXs, label, center, beta);
    objHistory = [objHistory; obj]; %#ok

    %*********************************************************************
    % Update U, V
    %*********************************************************************
    TX = sparse(nSmp, nFea);
    for iOrder = 1:nOrder
        TX = TX + beta(iOrder) * TXs{1, iOrder};
    end
    label_old = label;
    center_old = center;
    [label, center, ~, obj_cluster] = litekmeans(TX, nCluster, 'MaxIter', 1, 'Replicates', 10, 'Start', center_old);

    obj = compute_obj(X, TXs, label, center, beta);
    objHistory = [objHistory; obj]; %#ok
    if iter > 2 && abs( (objHistory(end-1) - objHistory(end))/objHistory(end-1) ) < myeps
        break;
    end
end

end

function obj = compute_obj(X, TXs, label, center, beta)
[nSmp,nFea] = size(X);
nOrder = size(TXs, 2);
TX = sparse(nSmp,nFea);
for iOrder = 1:nOrder
%     Ti = Ti + beta(iOrder) * Ts{1, iOrder};
    TX = TX + beta(iOrder)*TXs{1, iOrder};
end
UV = center(label, :);
E = TX - UV;
obj = sum(sum( E.^2 ));
end