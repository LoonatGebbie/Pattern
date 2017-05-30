%% Assignment/Project Script Test File : MACHINE_PATTERN_NYSEOLDDATA36STOCKS_001aa.m
% Authors: T. Gebbie, F. Loonat

%
% 

%% Notes:


%% 1. Data Description

% 
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
addpath(fullfile(userpathstr,projectpath,'html'));

%% 5. Process and Load data
ListOfFiles=dir(fullfile(userpathstr,projectpath,'data\nyseold'));
%  Assign the number of files in the directory to a
[a, b] =size(ListOfFiles);
% j is the index of the column for the output B matrix
j=1;

% i starts from 3 because the first two items in ListOfFiles is . and ..
for i = 3:a
    % The purpose of this if statement is to skip the NyseTicker,
    % NyseTickerMerged and ReadMe files.
    if strcmp(ListOfFiles(i).name(end-3:end),'.csv') && ~strcmp(ListOfFiles(i).name,'NyseTicker.csv') && ~strcmp(ListOfFiles(i).name,'NyseTickerMerged.csv');
        %ListOfFiles(i).name,
        
        % The matrix A is a two column matrix, the first column is the date and the second column is the data 
        A = csvread(ListOfFiles(i).name);
        ListOfFiles(i).name;
        % Matrix B will hold all the data to later be transfered to a
        % single .csv file
        % The second column of A is assigned to the j th column of matrix B
        % Can B be preallocated?
        B(:,j) = A(:,2);
        
        % increment j
        j=j+1;
    end   
end

% [col1, col2] = textread(ListOfFiles(i).name,'%s%n%*[^\n]','delimiter',',');
% temp1 = datenum(strcat('19', col1(1:9443)),'yyyymmdd');
% temp2 = datenum(strcat('20', col1(9444:end)),'yyyymmdd');
% dates = [temp1;temp2];
% D = m2xdate(dates);
% This sets the first column of B to be the date
%B = [D B];

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
x2 = reshape(x2',n,1,m);
% estimate the portfolios and performance
p2 = pattern(x2,1:5,1:10);



% offline estimation 
p2 = offline(p2);
p2.SH(end,:)

%% 7. Absolute Portfolios

% The absolute case
p3 = pattern(x2,1:5,1:10,[],'absolute');

% offline estimation 
p3 = offline(p3);
p3.SH(end,:)

%% 8. Save Output

% active
save(strcat(fullfile(userpathstr,projectpath,'scripts\NYSEold\'),'nyseold_36stocks_active.mat'),'p2');
% absolute
save(strcat(fullfile(userpathstr,projectpath,'scripts\NYSEold\'),'nyseold_36stocks_absolute.mat'),'p3')


%% 9. Print Ouput
% plot the required output
figure;
plotVar = {'p2.S','cumsum(transpose(squeeze(x2))-1)','p2.SH','p2.b'};
plotTitle = {'Case : NYSE 36 stocks'};
plotX = {'time'};
plotY = {'S','x2','SH','b'};
for i=1:4
    subplot(2,2,i);
    plot(eval(plotVar{i}));
    title(plotTitle{1});
    ylabel(plotY{i});
    xlabel(plotX{1});
end


figure;
plotVar = {'p3.S','cumsum(transpose(squeeze(x2))-1)','p3.SH','p3.b'};
plotTitle = {'Case : NYSE 36 stocks'};
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