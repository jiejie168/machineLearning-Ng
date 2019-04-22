function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X=[ones(m,1) X];
n1=input_layer_size+1; % number of unit in first layer, 401
n2=hidden_layer_size;  % number of unit in second layer (exclude the bias unit), 25
n3=num_labels ;    % number of output. 10

for i =1:m,
    
z2=X(i,:)*Theta1'; %  Theta1: n2 *n1, X: 1*n1, z2: 1*n2
a2=sigmoid(z2);  % a2: 1*n2

a2=[ones(size(a2,1),1) a2];  %a2:1*(n2+1)
z3=a2*Theta2';  %Theta2: n3*(n2+1), a2: 1*(n2+1);  1*n3
a3=sigmoid(z3);  % 1*n3,  prediction value for every sample
yi=zeros(1,n3);  % 1*n3
yi(y(i))=1;
predictions=-yi.*log(a3)-(1-yi).*log(1-a3); %predictions of hypothesis on every sample. 
    % 1*n3  
J=J+sum(predictions); % 
end 

J=J/m;
Jr=(lambda/2/m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
J=J+Jr;

% yk=zeros(m,n3);
% yk(sub2ind(size(yk),1:m,y'))=1;% it is hard to understand.
% 
% predictions=-yk.*log(a3)-(1-yk).*log(1-a3); %predictions of hypothesis on all m examples for each output. 
%    % m*n3  
% J=sum(predictions); % a row vector with the sum over each column.
% J=(1/m)*sum(J);

delta_sum1=zeros(size(Theta1));   % n2*n1;  
delta_sum2=zeros(size(Theta2));   % n3*(n2+1)

for i =1:m,
a1=X(i,:)'; % a1: n1*1
z2=Theta1*a1; %  Theta1: n2 *n1, X: 1*n1, z2: n2*1

a2=[1; sigmoid(z2)];  % (n2+1)*1
z3=Theta2*a2;  % n3*(n2+1)   , z3: n3*1
a3=sigmoid(z3);  % n3*1

yi=zeros(n3,1);  % n3*1
yi(y(i))=1;

delta3=a3-yi;  % delta3: n3*1
delta2=(Theta2'*delta3).*[1; sigmoidGradient(z2)]; %Theta2: n3*(n2+1),delta3: n3*1, sigmoidG:n2*1
delta2=delta2(2:end); % (n2-1)*1

delta_sum1=delta_sum1+delta2*a1';  % delta2, (n2-1)*1; a1:n1*1;
delta_sum2=delta_sum2 + delta3*a2'  ;  % delta 3, n3*1, a2: (n2+1)*1

end 

Theta1_grad = delta_sum1/m+(lambda/m)*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = delta_sum2/m+(lambda/m)*[zeros(size(Theta2,1),1) Theta2(:,2:end)];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
