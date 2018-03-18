function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(97,num_iters);
delta=zeros(2,1);
for iter = 1:num_iters,
	prediction=X*theta;
	delta(1,1)=sum((prediction-y).*X(:,1));
	delta(2,1)=sum((prediction-y).*X(:,2));

	theta=theta- (alpha/m)*delta;


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %






    % ============================================================

    % Save the cost J in every iteration    
%	J_history(iter,:) = (computeCost(X, y, theta))';
	J_history(:,iter) = computeCost(X, y, theta);

end;
	display("Successfull");
	display(X);
	display (y);
	display(theta);
	display(computeCost(X, y, theta));
%end;
