function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z)); %returns the two-element row vector d containing the number of rows and columns in obj
% so g has the same size with z.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i=1:size(z,1),
   for j=1:size(z,2),
       g(i,j)=1/(1+exp(-z(i,j)));
   end
end    
    


% =============================================================

end
