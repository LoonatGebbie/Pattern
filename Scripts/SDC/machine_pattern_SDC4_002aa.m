%% Pattern Matching algorithm test script: Case One
%
%
% Authors: T Gebbie, F Loonat 
%
%
% Uses:

%% 1. Data Description
% 
% Random test data to be used by the pattern class

%% 2. Data Cases

%% 3. Clear workspace
close all;
clear p4 p4a

%% 4. Set Paths

userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
% Project Paths:
% -- Modify this line to be your prefered project path ----->
projectpath = 'QuERILAB/machine';
% <----------------------------------------------------------
addpath(fullfile(userpathstr,projectpath,'data'));
addpath(fullfile(userpathstr,projectpath,'functions'));
addpath(fullfile(userpathstr,projectpath,'scripts','Random Data'));
addpath(fullfile(userpathstr,projectpath,'html'));

%% 5. Help files for PATTERN
help pattern;

%% 6.1 Random data test Case 4:
% 10 stocks and 1000 date-times 1 feature
m = 10; % number of stocks
n = 1; % number of features
p = 1000; % number of date-times

% mean and var of lognormal distribution
x_mean =repmat([1.001*ones(7,1); 0.999*ones(3,1)],[1,1,p]);
var = 0.0002;

% mean and std dev of associated normal distribution
mu = log((x_mean.^2)./sqrt(var+x_mean.^2));
sigma = sqrt(log(var/(x_mean.^2)+1));

[M,V]= lognstat(mu,sigma);

% initialize random number generator

seed = 7;
rng(seed);


% generate random returns from lognormal distribution
x = lognrnd(mu,sigma,[m,n,p]); % Should not learn

plot(cumprod(squeeze(x)'))
%% 6.4.1 Active Portfolios Case 4
% construct the class
p4 = pattern(x,1:5,1:10);
% run the offline learning
p4 = offline(p4);

%% 6.4.2 Absolute Portfolios Case 4
% construct the class
p4a = pattern(x,1:5,1:10,[],'absolute');
% run the offline learning
p4a = offline(p4a);
%% 6.4.3 Save Output

% active
save(fullfile(userpathstr,projectpath,'scripts\Random Data\Results\',strcat('randomcase4_seed',int2str(seed),'_active.mat')),'p4','x');
% absolute
save(fullfile(userpathstr,projectpath,'scripts\Random Data\Results\',strcat('randomcase4_seed',int2str(seed),'_absolute.mat')),'p4a','x')


