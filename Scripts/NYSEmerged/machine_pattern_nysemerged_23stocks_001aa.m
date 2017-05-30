%% Assignment/Project Script Test File : MACHINE_PATTERN_NYSEDATA_001aa.m
% Authors: T. Gebbie


%% 1. Data Description
% 
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
userpathstr = userpathstr(1:end-1);
% Project Paths:
% -- Modify this line to be your prefered project path ----->
projectpath = 'QuERILAB/machine';
% <----------------------------------------------------------
addpath(fullfile(userpathstr,projectpath,'data'));
addpath(fullfile(userpathstr,projectpath,'data/nysemerged'));
addpath(fullfile(userpathstr,projectpath,'functions'));
addpath(fullfile(userpathstr,projectpath,'scripts'));
addpath(fullfile(userpathstr,projectpath,'test_code'));
addpath(fullfile(userpathstr,projectpath,'html'));

%% 5. Process and Load data

ListOfFiles=dir(fullfile(userpathstr,projectpath,'data/nysemerged'));
%  Assign the number of files in the directory to a
[a, b] =size(ListOfFiles);
% j is the index of the column for the output B matrix
j=1;

% i starts from 3 because the first two items in ListOfFiles is . and ..
for i = 3:a
    % The purpose of this if statement is to skip the NyseTicker,
    % NyseTickerMerged and ReadMe files.
    if strcmp(ListOfFiles(i).name(end-3:end),'.csv') && ~strcmp(ListOfFiles(i).name,'NyseTicker.csv') && ~strcmp(ListOfFiles(i).name,'Bond.csv') && ~strcmp(ListOfFiles(i).name,'Cash.csv') && ~strcmp(ListOfFiles(i).name,'NyseTickerMerged.csv');
        %ListOfFiles(i).name,
        
        % The matrix A is a two column matrix, the first column is the date and the second column is the data 
        A = csvread(fullfile(userpathstr,projectpath,'data/nysemerged',ListOfFiles(i).name));
        
        % Matrix B will hold all the data to later be transfered to a
        % single .csv file
        % The second column of A is assigned to the j th column of matrix B
        % Can B be preallocated?
        B(:,j) = A(:,2);
        
        % increment j
        j=j+1;
    end   
end

[col1, col2] = textread(ListOfFiles(i).name,'%s%n%*[^\n]','delimiter',',');
temp1 = datenum(strcat('19', col1(1:9443)),'yyyymmdd');
temp2 = datenum(strcat('20', col1(9444:end)),'yyyymmdd');
dates = [temp1;temp2];
% D = m2xdate(dates);
% This sets the first column of B to be the date
% B = [D B];

% write matrix B to a .csv file
%csvwrite('testoutputnew.csv',B);
% 

%% 6. Active Portfolios
% data into x var (price relatives)
%x2 = fts2mat(nysefts);
x2 = B;

%x2 = exp(diff(log(x2)));
% pad NaN
x2(isnan(x2))=1;
% size of x
[m,n]=size(x2);
% reshape x to [Stocks,Features,Times]
x2 = reshape(x2,n,1,m);
% estimate the portfolios and performance
p2 = pattern(x2,1:5,1:10);
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
save(strcat(fullfile('/home/fayyaaz/Dropbox/Masters Work/Code Resilts/Results/'),'nysemerged_23stocks_active.mat'),'p2');
% absolute
save(strcat(fullfile('/home/fayyaaz/Dropbox/Masters Work/Code Resilts/Results/'),'nysemerged_23stocks_absolute.mat'),'p3')

% EOF