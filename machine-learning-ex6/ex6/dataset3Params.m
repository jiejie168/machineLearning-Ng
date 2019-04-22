function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
C=C(:);% transfer C to a column vector.
sigma=sigma(:); % transfer sigma to a column vector.

a1=size(C,1);
b1=size(sigma,1);
minT=1;
for i =1:a1,
    for j=1:b1,
        model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
        predictions=svmPredict(model,Xval);
    if (minT>=mean(double(predictions~=yval))),
        minT=mean(double(predictions~=yval));
        C_temp=C(i);
        sigma_temp=sigma(j);
    end 
    end 
end

C=C_temp;
sigma=sigma_temp;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
% =========================================================================

end
