function [U_S, stock1wealth, stock2wealth, bm] = cover_up_rec(x)


[n, ~] = size(x);

% Initialize vectors for the accumulative wealth of each stock and the
% universal portfolio
stock1wealth=zeros(n,1);
stock2wealth=zeros(n,1);
universalwealth=zeros(n,1);

% Initialize Sn which will be a matrix that contains the wealth accumulated
% from the different constant rebalanced portfolios
Sn = zeros(n,21);

% Initialize bm to be a vector that represents the fraction of wealth
% assigned to iroqu
bm = zeros(n,1);

% Set the initial fraction of wealth to be equally distributed among the 2
% stocks
bm(1) = 0.5;

for i = 1:n
    % Calculate the wealth gained at time i for iroqu and kinar
    stock1wealth(i) = prod(x(1:i,1));
    stock2wealth(i) = prod(x(1:i,2));
    
    % Calculate the wealth gained at time i from using different
    % rebalanced portolios 
    Sn(i,:) = (0:0.05:1)*x(i,1) +(1:-0.05:0)*x(i,2);
    
    % i>1 beacause bm(1) is already initialized
    if i >1
        % Calculate new value of wealth assigned to iroqu
        b = sum((0:0.05:1)*(prod(Sn(1:i,:))'))/(sum(prod(Sn(1:i,:))));
        bm(i) = b;
    end
    
    % Calculate the wealth achieved at time i from the universal portfolio
    universalwealth(i) = (sum(prod(Sn(1:i,:))))/21;

end
U_S = universalwealth;
end