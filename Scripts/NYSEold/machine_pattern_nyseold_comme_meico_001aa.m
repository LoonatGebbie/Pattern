%% Assignment/Project Script Test File : MACHINE_PATTERN_NYSEOLDDATACOMMEMEICO_001aa.m
% Authors: T. Gebbie, F.Loonat

%% 1. Data Description
% 
% % 
% NYSEOLD data set taken from http://www.cs.bme.hu/~oti/portfolio/data.html.
%
% ----------------------------
% nyse.zip (from Yoram Singer)
% ----------------------------
% This data set includes daily prices of 36 assets.
% from: 1962.07.03
% until: 1984.12.31
% number of the trading days: 5651

%% 2. Data Cases
 
%% 3. Clear workspace
close all;
clear all;
clc;

%% 4. Set Paths (implement configuration control)
% Specify the path directory containing the data files
% addpath c:\Users\user\Documents\MATLAB
% use the userpathstr.
%
%  # Project Path:              userpath/MATLAB/<PROJECT>
%  # Data Path:                 userpath/MATLAB/<PROJECT>/data        
%  # Script Files:              userpath/MATLAB/<PROJECT>/scripts     
%  # Classes and Function:      userpath/MATLAB/<PROJECT>/functions   
%  # Published Output:          userpath/MATLAB/<PROJECT>/html        
%  # Legacy Code:               userpath/MATLAB/<PROJECT>/code
%

userpathstr = userpath;
userpathstr = userpathstr(~ismember(userpathstr,';'));
% Project Paths:
% -- Modify this line to be your prefered project path ----->
projectpath = 'QuERILAB/machine';
% <----------------------------------------------------------
addpath(fullfile(userpathstr,projectpath,'data'));
addpath(fullfile(userpathstr,projectpath,'data/nyseold'));
addpath(fullfile(userpathstr,projectpath,'functions'));
addpath(fullfile(userpathstr,projectpath,'scripts'));
addpath(fullfile(userpathstr,projectpath,'scripts/NYSEold'));
%addpath(fullfile(userpathstr,projectpath,'test_code'));
addpath(fullfile(userpathstr,projectpath,'html'));

%% 5. Load data
%
% The Data is taken from the nyseold data set used in Gyorfi et el 

stock1 = csvread('comme.csv');
stock2 = csvread('meico.csv');

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
save(strcat(fullfile(userpathstr,projectpath,'scripts\NYSEold\'),'comme_meico_active.mat'),'p2');
% absolute
save(strcat(fullfile(userpathstr,projectpath,'scripts\NYSEold\'),'comme_meico_absolute.mat'),'p3')

%% 9. Print Ouput
% plot the required output
figure;
plotVar = {'p2.S','cumsum(transpose(squeeze(x2))-1)','p2.SH','p2.b'};
plotTitle = {'Case : NYSEOLD commemeico'};
plotX = {'time'};
plotY = {'S','x2','SH','b'};
for i=1:4
    subplot(2,2,i);
    plot(eval(plotVar{i}));
    title(plotTitle{1});
    ylabel(plotY{i});
    xlabel(plotX{1});
end
% EOF