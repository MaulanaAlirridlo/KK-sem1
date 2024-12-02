function resultMatrix = multi_blockSize(A, B)
[nSmp, ~] = size(A);
[~, nFea] = size(B);
blockSize = 10;

resultMatrix = sparse(nSmp, nFea);

for i = 1:blockSize:nSmp
    %disp(i);
    endIndex = min(i + blockSize - 1, nSmp);
    currentBlock = A(i:endIndex, :);
    resultMatrix(i:endIndex, :) = currentBlock * B;
end
end