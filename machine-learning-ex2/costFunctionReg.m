function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z=X*theta;% the parameter z for hypothesis: m*1 vector
g=sigmoid(z);% hypothesis w.r.t z value: m*1 vector
predictions=-y.*log(g)-(1-y).*log(1-g); %predictions of hypothesis on all m examples

J=(1/m)*sum(predictions)+(lambda/2/m)*sum(theta(2:end,1).^2);

grad(1)=(1/m)*sum((g-y).*X(:,1));
for i = 2:size(theta),
    grad(i,1)=(1/m)*sum((g-y).*X(:,i))+(lambda/m)*theta(i);  %m*1,m*1
end 




% =============================================================

end
