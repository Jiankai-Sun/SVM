% First run nb_train.m, then run nb_test.m

[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE
numSpam = sum(trainCategory);
numNon = numTrainDocs - numSpam;

p0 = (numNon + 1)/(numTrainDocs + 2);  % Laplace smoothed ML of p0
p1 = (numSpam + 1)/(numTrainDocs + 2);  % Laplace smoothed ML of p1

phi = zeros(2, numTokens);

for docNum = 1:numTrainDocs
    phi(trainCategory(docNum) + 1, :) = phi(trainCategory(docNum)+1,:) + trainMatrix(docNum,:);
end

n0 = sum(sum(trainMatrix(trainCategory == 0, :)));
n1 = sum(sum(trainMatrix(trainCategory == 1, :)));

% Laplace Smoothed log-(ML of phi) 
logphi0 = log(phi(1, :) + ones(1, numTokens)) - log(n0 + numTokens);
logphi1 = log(phi(2, :) + ones(1, numTokens)) - log(n1 + numTokens);