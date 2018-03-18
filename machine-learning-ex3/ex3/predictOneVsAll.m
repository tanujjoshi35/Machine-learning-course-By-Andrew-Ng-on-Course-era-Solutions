function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
display(size(X));
%display(size(all_theta));
h=zeros(num_labels,size(X,1)); % h is a 10x5000 matrix which store the calculated value of h_theta(x) for all 5000 examples, for all 10 nodes/classes
for i=1:num_labels,
	h(i,:)=sigmoid(all_theta(i,:)*X');
end;

for i=1:m,

	[val, index]=max(h(:,i));
	
	if index==10,
		p(i)=0;
	else,
		p(i)=index;
		
	end;

end;

%display(p(1:200,1));
%display(p(find(p>0)));



% =========================================================================


end;
