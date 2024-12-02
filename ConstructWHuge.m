function W = ConstructWHuge(X, k, batch_size)
[nSmp, nFean] = size(X);

if ~exist('batch_size', 'var')
    batch_size = 5000;
end



ids = 1:nSmp;
num_bins = ceil(nSmp / batch_size);  % Calculate the number of bins needed
bins = cell(1, num_bins);
disp('construct w 1');
for i = 1:num_bins
    start_idx = (i - 1) * batch_size + 1;
    end_idx = min(i * batch_size, nSmp);
    bins{i} = ids(start_idx:end_idx);
end

X_norm = sum(X.^2, 2);

r_idx_cell = cell(num_bins, 1);
c_idx_cell = cell(num_bins, 1);
v_cell = cell(num_bins, 1);
% As = cell(num_bins, num_bins);
disp('construct w 2');
for i1 = 1:num_bins
    %disp(i1);
    id_a = bins{i1};
    na = length(id_a);
    Xa = X(id_a, :);
    norm_a = X_norm(id_a, :);
    Idx_a_k_cell = cell(1, num_bins);
    D_a_k_cell = cell(1, num_bins);
    r_idx = (i1 - 1) * batch_size + [1:na];
    disp(['construct w 3', num2str(i1)]);
    for i2 = 1:num_bins
        id_b = bins{i2};
        nb = length(id_b);
        Xb = X(id_b, :);
        norm_b = X_norm(id_b, :);
        D_ab = norm_a + norm_b' - 2 * Xa * Xb';
        if i1 == i2
            D_ab = D_ab + eye(size(D_ab)) * 1e8;
        end
        [D_ab_k, Idx_ab_k] = sort(D_ab, 2, 'ascend');
        D_a_k_cell{i2} = D_ab_k(:, [1:k+1]);  % batch_size * (k+1)
        Idx_a_k_cell{i2} =(i2 - 1) * batch_size + Idx_ab_k(:, [1:k+1]);  % batch_size * (k+1), id-remapping
    end
    D_a_k = cell2mat(D_a_k_cell); % batch_size * (k * num_bins)
    [D_k_1, Id_k_1] = sort(D_a_k, 2, 'ascend');
    D_a_k_2 = D_k_1(:, [1:k+1]);
    Id_k_2 = Id_k_1(:, [1:k+1]);
    lidx_a_k_2 = sub2ind(size(D_a_k), repmat((1:na)', k+1, 1), Id_k_2(:));
    Idx_a_k = cell2mat(Idx_a_k_cell); % batch_size * ((k+1) * num_bins)
    Idx_a_k_2 = Idx_a_k(lidx_a_k_2);
    % D_a_k_3 = reshape(D_a_k_2(Idx_a_k_2), na, k+1);
    v1 = (D_a_k_2(:, k+1) - D_a_k_2(:, [1:k]));
    v2 = k*D_a_k_2(:, k+1) - sum(D_a_k_2(:, [1:k]),2 );
    v12 = v1 ./ (v2 + eps) + eps;
    Id_k_3 = Idx_a_k_2(:);
    r_idx_cell{i1} = repmat(r_idx', k, 1);
    c_idx_cell{i1} = Id_k_3(1:na*k);
    v_cell{i1} = v12(:);
end

r_idx = cell2mat(r_idx_cell);
c_idx = cell2mat(c_idx_cell);
v = cell2mat(v_cell);
W = sparse(r_idx, c_idx, v, nSmp, nSmp, nSmp * k);
end