function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
A1=[ones(size(X,1),1) , X]; %A1 is 5000x401
display(size(A1));
A2=(sigmoid(A1*Theta1'))'; % A2 is 25x5000 Matrix
A2=[ones(1,size(X,1));A2]; % A2 26x5000 Matrix
A3=(sigmoid(Theta2*A2));   %A3 is 10x5000

for i=1:m,
	[val, index]=max(A3(:,i));
	
	if(index==10),
		p(i)=0;
	else,
		p(i)=index;
	end;

end;






% =========================================================================


end
