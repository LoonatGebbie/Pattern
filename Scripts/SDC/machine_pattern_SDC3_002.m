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

%% 6.3 Random data test Case 3:
% 10 stocks and 1000 date-times 1 feature
m = 10; % number of stocks
n = 1; % number of features
p = 1000; % number of date-times

% mean and var of lognormal distribution
mean = max(0,min(0.05+0.15*randn(m,1),0.20)); % different random returns
mean = 1+repmat(mean,[1,n,p]); 
var = 0.002;

% mean and std dev of associated normal distribution
mu = log((mean.^2)./sqrt(var+mean.^2));
sigma = sqrt(log(var/(mean.^2)+1));

[M,V]= lognstat(mu,sigma);

% initialize random number generator
seed = 7;
rng(seed);

% generate random returns from lognormal distribution
x = lognrnd(mu,sigma,[m,n,p]); % Should be able to learn


%% 6.3.1 Active Portfolios Case 3
% construct the class 
p3 = pattern(x,1:5,1:10);
% run the offline learning
p3 = offline(p3);

%% 6.3.2 Absolute Portfolios Case 3
% construct the class 
p3a = pattern(x,1:5,1:10,[],'absolute');
% run the offline learning
p3a = offline(p3a);

%% 6.3.3 Save Output

% active
save(fullfile(userpathstr,projectpath,'scripts\Random Data\',strcat('randomcase3_seed',int2str(seed),'_active.mat')),'p3','x');
% absolute
save(fullfile(userpathstr,projectpath,'scripts\Random Data\',strcat('randomcase3_seed',int2str(seed),'_absolute.mat')),'p3a','x')


