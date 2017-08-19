% Before using this method, set num_train to be the number of training
% examples you wish to read.
num_train = 100;

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('MATRIX.TRAIN.%d', num_train));

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (sparseTrainMatrix > 0);

numTrainDocs = size(Xtrain, 1);
numTokens = size(Xtrain, 2);

% Xtrain is a (numTrainDocs x numTokens) sparse matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents if the j-th token appears in
% email i.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
% vector ytrain.

% This vector should be output by this method
average_alpha = zeros(numTrainDocs, 1);

%---------------
% YOUR CODE HERE
% tau = 8;
% lambda = 1./(64.*numTrainDocs);
% % The learning rate is such that at iteration t of stochastic  gradient
% % descent we use rate 1./sqrt(t)
% K = rbf_kernel(Xtrain, tau);
% 
% for t = 1:40*numTrainDocs
%     ix = randi([1, numTrainDocs], 1, 1);
%     learning_rate = 1./sqrt(t);
%     Ki = K(ix, :);  % Vector of example ix's similarities to other examples
%     yi = ytrain(ix);
%     average_alpha = average_alpha - learning_rate .* cost_gradient(average_alpha, K, Ki, yi, lambda);
% end

Xtrain = full(Xtrain);
current_alpha = zeros(numTrainDocs, 1);
sum_alpha = zeros(numTrainDocs, 1);
tau = 8;
lambda = 1/(64 * numTrainDocs);
I = ones(numTrainDocs, 1);
K = zeros(numTrainDocs, numTrainDocs);

for i = 1:numTrainDocs
    temp = I * Xtrain(i, :);
    temp2 = Xtrain - temp;
    K(:, 1) = exp(-sum(temp2.^2, 2)./(2 * tau^2));
end

for t = 1:40 * numTrainDocs
    i = randi(numTrainDocs);
    if ytrain(i) * dot(K(:, i), current_alpha) < 1
        subgradient = -ytrain(i)*K(:,i);
    else
        subgradient = zeros(numTrainDocs, 1);
    end
    eta = 1/sqrt(t);
    current_alpha = current_alpha - eta * ...
        (subgradient + numTrainDocs * lambda * current_alpha(i) * K(:, i));
    sum_alpha = sum_alpha + current_alpha;
end 

average_alpha = sum_alpha / (40 * numTrainDocs);
%---------------
