%% Assignment/Project Script Test File : MACHINE_PATTERN_NYSEOLDDATAIBMCOKE_001aa.m
% Authors: T. Gebbie, F.Loonat

%% 1. Data Description
% % 
% NYSEMERGED data set taken from http://www.cs.bme.hu/~oti/portfolio/data.html.

% -----------------------------------------------------------------------
% nysemerged.zip (from Yahoo! Finance, data cleaning and data preparation
%                             by Ga'bor Gelencse'r)
% -----------------------------------------------------------------------
% This data set includes daily prices of 23 assets.
% from: 1962.07.03
% until: 2006.11.24
% number of the trading days: 11178


%% 2. Data Cases
 
%% 3. Clear workspace
close all;
clear all;
%clc;

%% 4. Set Paths (implement configuration control)

userpathstr = userpath;
userpathstr = userpathstr(1:end-1);
% Project Paths:
% -- Modify this line to be your prefered project path ----->
projectpath = 'QuERILAB/machine';
% <----------------------------------------------------------
addpath(fullfile(userpathstr,projectpath,'data'));
addpath(fullfile(userpathstr,projectpath,'data/nysemerged'));
addpath(fullfile(userpathstr,projectpath,'functions'));
addpath(fullfile(userpathstr,projectpath,'scripts'));
addpath(fullfile(userpathstr,projectpath,'scripts/NYSEmerged'));
addpath(fullfile(userpathstr,projectpath,'html'));

%% 5. Load data
%
% The Data is taken from the nyseold data set used in Gyorfi et el 


stock1name = 'comme';
stock2name = 'kinar';

stock1 = csvread(fullfile(userpathstr,projectpath,'data/nysemerged',strcat(stock1name,'.csv')));
stock2 = csvread(fullfile(userpathstr,projectpath,'data/nysemerged',strcat(stock2name,'.csv')));

% x is the return vector of the stocks
x = [stock1(:,2) stock2(:,2)];

%% 6. Active Portfolios
% data into x var (price relatives)
%x2 = fts2mat(nysefts);
x2 = x;

%x2 = exp(diff(log(x2)))
% pad NaN
x2(isnan(x2))=1;
% size of x
[m,n]=size(x2);
% reshape x to [Stocks,Features,Times]
x2 = reshape(x2',n,1,m);
% estimate the portfolios and performance
p2 = pattern(x2,1:5,1:10);

%p2 = pattern(x2,1:5,1:10,[],'absolute');
% offline estimation
p2 = offline(p2);

p2.SH(end,:)
%% 7. Absolute Portfolios
% run for the absolute case
p3 = pattern(x2,1:5,1:10,[],'absolute');

% offline estimation
p3 = offline(p3);

p3.SH(end,:)

%% 8. Save Output

% active
save(strcat(fullfile('/home/fayyaaz/Dropbox/Masters Work/Code Resilts/Results/'),'nysemerged_',stock1name,'_',stock2name,'_active.mat'),'p2');
% absolute
save(strcat(fullfile('/home/fayyaaz/Dropbox/Masters Work/Code Resilts/Results/'),'nysemerged_',stock1name,'_',stock2name,'_absolute.mat'),'p3')

% EOF