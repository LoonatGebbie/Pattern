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
clear all;
clc;

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

%% 6.1 Random data test Case 1:
% 10 stocks and 1000 date-times 1 feature
m = 10; % number of stocks
n = 1; % number of features
p = 1000; % number of date-times

% mean and var of lognormal distribution
x_mean = 1.00;
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
semilogy(cumprod(squeeze(x)'))

%% 6.1.1 Active Portfolios Case 1
% construct the class
p1 = pattern(x,1:5,1:10);
% run the offline learning
p1 = offline(p1);

%% 6.1.2 Absolute Portfolios Case 1
% construct the class
p1a = pattern(x,1:5,1:10,[],'absolute');
% run the offline learning
p1a = offline(p1a);
%% 6.1.3 Save Output

% active
save(fullfile(userpathstr,projectpath,'scripts\Random Data\',strcat('randomcase1_seed',int2str(seed),'_active.mat')),'p1','x');
% absolute
save(fullfile(userpathstr,projectpath,'scripts\Random Data\',strcat('randomcase1_seed',int2str(seed),'_absolute.mat')),'p1a','x')


