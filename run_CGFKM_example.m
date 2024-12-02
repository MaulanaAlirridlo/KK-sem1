clear;clc;

load jaffe_213n_676d_10c_uni.mat;
disp('loading');
% Load dataset Isolet
%data = load('isolet.mat');
%X = data.X; % Asumsi fitur berukuran 7797x617
%y = data.y; % Asumsi label berukuran 7797x1

% Tentukan parameter
%num_classes = 26;   % Jumlah kelas
%samples_per_class = 60; % Sampel per kelas
%total_samples = num_classes * samples_per_class; % Total sampel (1560)

% Inisialisasi matriks untuk data yang diratakan
%X_balanced = [];
%y_balanced = [];

% Proses per kelas
%for cls = 1:num_classes
    % Cari indeks sampel yang termasuk dalam kelas ini
 %   idx = find(y == cls);
    
    % Acak dan pilih sejumlah 'samples_per_class' sampel
  %  idx_selected = idx(randperm(length(idx), samples_per_class));
    
    % Tambahkan sampel ke dataset baru
   % X_balanced = [X_balanced; X(idx_selected, :)];
   % y_balanced = [y_balanced; y(idx_selected)];
%end

% Pastikan data tersusun rapi
%X = double(X_balanced); % Konversi ke tipe double jika diperlukan
%y = double(y_balanced);
%load mnist.mat
%rng(42);
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
initCentroid = [1 2 3 4 5 6 7 8 9 10];
disp('construct w');
Si = ConstructWHuge(X, 5, batch_size);
nOrder = 5;
Ts = cell(1, nOrder);
Ts{1, 1} = speye(nSmp);
Ts{1, 2} = Si;
disp('ts');
for jOrder = 3:nOrder
    tmp1 = multi_blockSize(Si, Ts{1, jOrder-1});
    Ts{1, jOrder} =sparse(2*tmp1 - Ts{1, jOrder-2});
end
clear tmp1;

TXs = cell(1, nOrder);
for iOrder = 1:nOrder
    TXs{1, iOrder} = multi_blockSize(Ts{1, iOrder}, X);
end

A1 = zeros(nOrder, nOrder);
%*********************************************************************
% Merge T and T'
%*********************************************************************
for iOrder = 1:nOrder
    for jOrder = iOrder:nOrder
        e2_ij = sum(sum( TXs{1, iOrder} .* TXs{1, jOrder} ));
        A1(iOrder, jOrder) = e2_ij;
        A1(jOrder, iOrder) = e2_ij;
    end
end
[~, o_2] = eig(A1);
disp(['min eigval is ', num2str(min(diag(o_2)))]);
disp('cgfkm');
[label, objHistory, beta] = CGFKM_fast(X, nCluster, nOrder, knn_size, TXs, A1, 'initCentroid', initCentroid);
result_10 = my_eval_y(label, Y);
