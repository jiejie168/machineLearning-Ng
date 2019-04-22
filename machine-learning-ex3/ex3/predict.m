function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X=[ones(m,1) X];
n1=size(X,2); % number of unit in first layer, 401
n2=size(Theta1, 1);  % number of unit in second layer (exclude the bias unit), 25

z2=Theta1*X'; %  Theta1: n2 *n1, X: m*n1, z2: n2*m 
a2=sigmoid(z2);  % a2: n2*m
a2=a2';  % m*n2

a2=[ones(size(a2,1),1) a2];  %a2:(n2+1)*1
z3=a2*Theta2';  %Theta2: num_label*(n2+1), a2: (n2+1)*m;  (n2+1)*num_label
a3=sigmoid(z3);  % (n2+1)*num_label

[val,index]=max(a3,[],2);
p=index;


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================


end
