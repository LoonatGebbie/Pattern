classdef pattern
    % PATTERN Pattern matching and learning class
    %
    % The class implements both online and offline pattern matching and
    % learning over k-tuples for M objects and N features as specified in
    % an MxNxP data matrix X. The algorithm searches for L nearest-neighbour
    % K-tuple matches in the provided partition of data. A qaudratic
    % approximation is used to find the log-optimal portfolio using T+1
    % expected return subsequent to the pattern matching times T for each
    % k and ell to find H(K,L,CI;T) and SH(K,L,CI;T) for each K-tuple
    % L value and cluster CI. The controls are then aggregate using machine
    % learning to provide the controls B and realised accumulated
    % performance S. If the K matching patterns are predefined then K is
    % the index over the matching patterns. The default is and empty
    % matching pattern and to use the last k-tuple as the matching pattern.
    %
    % See Also: PATTERN/MATCH, PATTERN/OFFLINE, PATTERN/LEARN, PATTERN/ONLINE, HFTS
    
    %
    % A. OHLC patterns
    % B. Fundamental model patterns
    % C. (side information partitioning)
    %
    % 1. *Data* 
    %   2.1 M   (date-times)
    %   2.2 N  entities (objects e.g. stocks)
    %   2.3 P  features (OHLC)
    % 2. *Pattern* (k-tuple) [historic or user provided]
    %   2.1 k-tuple (k free parameter)
    %       2.2.1. k=1...K
    %       2.2.2. k=1,...,N*K multiples of N (DSI)
    %   2.2 k-tuple ConSet=[A,b] A*x>=b active/absolute (0,1)
    %       2.2.1. active (many stocks,stock+cash)
    %       2.2.2. absolute (single stock, long only portfolio)
    % 3. *Segment (s) and Partition (p)*
    %   3.1. [111...1] : full partition search for L nearest neighbours
    %   3.2. [10..0],[010...0],...,[0...01] : L partitions for single best fit in each
    %        partition
    %   3.3. [001],[011],[111] for partitions weight to current time
    %   3.4. ball radius (1-sigma ball relative to average distance)
    % 4. *Distance*
    %   4.1. surface norm
    %   4.2. vector norm
    % 5. *Predictors*
    %   5.1. dependent (correlated) matching times j(n,ell)
    %   5.2. independent matching times j(1...n,ell)
    %   5.3. different matching times ???
    %   5.4. [equi-probable] view is geometric average E[r]_t = exp(r_1+...+r_L)
    % 6. *Agents*
    %   6.1. Active/Absolute mean-variance
    %   6.2. Canonical Agents:
    %       6.2.1. controls H (NxMxT) M objects, N agents, T times)
    %       6.2.2. performance SH (arithmetic) (TxN for T times and N agents)
    %       6.2.3. horizon parameter (t to include delay)
    %   6.3. allow mixture of long-only with cash neutral.
    % 7. *Algorithm*
    %   7.1. Online/Offline
    %   7.2. Parallel computing
    %   7.3. Interface agents (h,SH)
    % 8. *Machine Learning* (EG,EW,EWMA,UNIV + abs/act + part/comb)
    %   8.1. Weighted Arithmetic Average over all agents to create optimal predictors
    %       8.1.1. Performance weighted averaging (using arithmetic averaging)
    %           B(T) = SUM(K,L) (SH(T-1|K,L) H(T|K,L)) / SUM(K,L) SH(T-1|K,L)
    %           this is probability wieghted where the probability is
    %           proportional to the returns.
    %       8.1.2. Exponentially weight more recent performance data
    %   8.2. parameters: window W_L, forgetting factor Lamba_L
    %   8.3. either fully invested or active.
    % 9. *Sectors and States* (clusters)
    %   9.1. CI PxN for P clusters and N objects
    
    % Author: Tim Gebbie
    
    %% public properties
    properties
        b = []; % aggregated controls    (T,M) Time x Objects
        S = []; % aggregated performance (T,1) Time x 1
        h = []; % agent controls      (N,M,T)  Agents x Objects x Time
        SH = []; % agents performance (T,N)    Time x Agents
    end
    %% private properties
    properties (Access = private)
        x = []; % data as price relatives (M,F,T) Objects x Features x Time
        k = []; % k-tuple size
        ell = []; % partition size
        p = []; % current partition
        s = []; % segment specification
        ci = []; % clusters  (C,M) Cluster x Objects
        mnp = []; % dimensionality (M,F,T)
        ntype = 'active'; % agent normalisation (estimation)
        lntype = 'absolute'; % agent relative (learning) normalisation
        ltype = 'univlearning';
        lparam = []; % learning parameters
        ptype = 'trivial';
        qH = []; % agents controls (T,N) Time x Agents
        xnk = {}; % predefined patterns
        horz = 1;
    end
    %% methods
    methods
        %% constructor
        function p = pattern(varargin)
            % P = PATTERN/PATTERN constructor
            %
            % P = PATTERN(X,K,ELL,CI,NTYPE,LNTYPE,LTYPE) For data X (price relatives)
            %   and is MxNxP dimensional data for M objects, N features and
            %   P date-times. dim(X) is dim(Price)-1. A typical object is
            %   a stocks, e.g. AGL, a typical feature is a factors such as
            %   OPEN, HIGH, LOW, CLOSE (OHLC), and date-times are the time
            %   stamps for the data.
            %
            % Example 1:
            % >> x = 1+0.2*randn(10,1,1000);
            % >> p = offline(p);
            % >> p.S
            %
            %  P = PATTERN(X,XNK,ELL,CI,NTYPE,LNTYPE,LTYPE) For matching
            %   pattern XNK instead of K-tuple range.
            %
            %  P = PATTERN(X,XNK,ELL,CI,NTYPE,LNTYPE,LTYPE,TREND)
            %   LONGTERM is TRUE to use long-term relative views. This is 
            %   by default false.
            %
            % The data X is homogenize in time. K is the set of tuple
            % sizes if there is no matching pattern e.g. [1:3], if there
            % is a matching pattern it is the number of matching patterns. 
            % The number of matching times is ELL this is typically 10 
            % per partition. The cluster definitions is CI, this is by 
            % default the trivial cluster (all the objects) when CI is 
            % empty. CI is KxM for K clusters of M objects. NTYPE is the 
            % strategy normalisation and can be either 'active', 'absolute' 
            % or 'gyorfi_opt'. LNTYPE is the agent normalisations and can
            % be either 'active' or 'absolute'.
            % The LTYPE is the learning type this is by default 'univ'.
            %
            % Note: X are price relatives (P(t)/P(t-1)). These can be
            % conveniently computed using EXP(DIFF(LOG(P))).
            %
            % See Also PATTERN/DISPLAY, PATTERN/SUBSREF
            
            if nargin==0
                % null constructor
            else
                %% Compute price relatives as returns
                x0 = varargin{1}; % (dim = n-1);
                % size of the data (M=OBJECTS, N=FEATURES, P=DATATIMES)
                mnp0 = size(x0);
                % catch the case of M,N,1 (data sliced at a single time)
                if length(mnp0)==2
                    mnp0(1:2) = mnp0;
                    mnp0(3) = 1;
                end
                xnk0 = varargin{2};
                % trivial cluster
                ci0 = true(1,mnp0(1));
                % set defaults for optional inputs
                optargs = {x0 xnk0 10 ci0 'active' ...
                    'absolute' 'univlearning' 'trivial' false};
                % now put these defaults into the valuesToUse cell array,
                optargs(1:nargin) = varargin;
                % number of k's
                p.k=size(xnk0);
                % class test
                switch class(xnk0)
                    case 'cell'
                        % Case 1: multiple (k0) user defined patterns
                        % dimensions of the possible matching pattern input
                        xnlmnp = size(xnk0{1});
                    case 'double'
                        % Case 2: single user defined pattern or k-tuples
                        % dimensions of the possible matching pattern input
                        xnlmnp = size(xnk0(1));
                end     
                % check for either tuple or pattern
                if (xnlmnp(1)==mnp0(1)) && (xnlmnp(2)==mnp0(2))
                    % user defined patterns
                    [p.x, p.xnk, p.ell, p.ci, p.ntype, p.lntype, p.ltype, p.ptype]...
                        = optargs{:};
                else                 
                    % Place optional args in memorable variable names
                    [p.x, p.k, p.ell, p.ci, p.ntype, p.lntype, p.ltype, p.ptype]...
                        = optargs{:};
                end
                switch p.ltype
                    case 'eglearning'
                        p.lparam = 0.01;
                    case 'univlearning'
                        p.lparam = Inf;
                    case 'ewlearning'
                        p.lparam = 0.99;
                end
                % ensure the inputs are feasible
                if isempty(p.ci) || strcmp(p.ci,':'), p.ci = ci0; end;
                % update the size property
                p.mnp = mnp0;
                % control parameters
                L = size(p.ell,2); % partitions (param 1)
                W = size(p.ci,1); % cluster
                K = size(p.k,2); % tuples (param 2)
                % initialise state variables and pre-allocate memory
                p.b = nan(p.mnp(3)+1,p.mnp(1)); % Time x Objects        
                p.h = nan(L*W*K,p.mnp(1),p.mnp(3)+1); % Agents x Objects x Time    
                p.qH = nan(p.mnp(3)+1,L*W*K); % Time x Agents
                p.S = ones(p.mnp(3)+1,1); % Time x 1
                p.SH = ones(p.mnp(3)+1,L*K*W); % Time x  Agents
                %% partition the data
                p=partition(p);
            end
        end
        %% offline (to force offline estimation)
        function p=offline(p)
            % PATTERN/OFFLINE Offline estimation
            %
            % P = OFFLINE(P) to estimate (H,SH) for T=K*L:T0 for K and L.
            %
            % See Also: PATTERN/ONLINE
            
            tmin0 = 3*max(p.ell);
            t0 = p.mnp(3);
            if t0<tmin0
                error('pattern:offline','Not enough Data L*K>T');
            end
            % Find matching times j for pattern [offline loop]
            for t=tmin0:t0 % time loop
                p = online(p,t);
            end
            
        end
        %% online (to force online estimation)
        function p=online(varargin)
            % PATTERN/ONLINE Offline estimation
            %
            % P = ONLINE(P) to estimate (H,SH,B,S) at T for the range of
            %   K and L over the specified clusters CI. This requires the
            %   online structure for learning to have been initialised. T
            %   is taken to be the last time in the object.
            %
            % P = ONLINE(P,T) to estimate online values at time T using the
            %   data from times 1 to T (1:T).
            %
            % See Also: PATTERN/ONLINE, PATTERN/MATCH, PATTERN/MATCH
            
            %% input parameters
            p = varargin{1}; % (dim = n-1);
            % set defaults for optional inputs
            optargs = {p p.mnp(3) {}};
            % now put these defaults into the valuesToUse cell array,
            optargs(1:nargin) = varargin;
            % Place optional args in memorable variable names
            [p, t, s0] = optargs{:};           
            %% control parameters
            L = size(p.ell,2); % partitions (param 1)
            W = size(p.ci,1); % cluster
            K = size(p.k,2); % tuples (param 2)
            %% Model Identification Loop
            % initial controls
            % t+1 rows as the last row is for an unrealised return
            ht = nan(W*K*L,p.mnp(1)); % agents controls
            xt = p.x(:,:,1:t);
            % exit if there is not enough data
            if 3*max(p.ell)>t
                error('pattern:online','Not enough data K*L>T');
            end
            if (t>p.mnp(3))
                error('pattern:online','Not enough data T> dim(X,T)');
            end
            % partition the data
            p = partition(p,t);
            % partition
            p0 = p.p; 
            % matching loop
            for w0=1:W % clusters (groups)
                ci0=p.ci(w0,:);
                % the cluster referenced data block with fixed partition
                xtw0 = xt(ci0,:,:);
                for ell0=1:L, % parameter 1 -> passed to match (ell neighbours)
                    ell1 = p.ell(ell0);
                    for k0=1:K  % parameter 2 -> passed to match (k-tuple)
                        % expert index KLrow(w0,k0,ell0)
                        KLrow = sub2ind([W*K,L],K*w0-k0+1,ell0);
                        % (k,ell)-agents matched pattern for cluster ci(w0)
                        % --- 1. select the pattern -------------------------
                        if isempty(p.xnk)
                            % k index tuple size
                            xnk0 = xtw0(:,:,end-p.k(k0)+1:end)-1; % r= R - 1
                        else
                            % k index of the matching pattern
                            xnk0 = p.xnk{p.k(k0)};
                        end
                        % --- 2. pattern matching ---------------------------
                        [hklt,s0]=match(xtw0,p0,xnk0,ell1,p.ntype,p.horz,s0,L); % online
                        % ------------------------------------------------
                        % expert controls per cluster mapping
                        ht(KLrow,ci0) = transpose(hklt);
                    end % k
                end % ell
            end % w
            % initialise h,SH if it is the trivial object
            % compute the update performance for the prior agent step
            if any(isnan(p.h(:,:,t)))
                dSH = ones(size(p.SH(t,:)));
            else
                dSH = transpose((p.h(:,:,t) * (p.x(:,1,t)-1))+1); % was exp
            end
            % remove NaN
            ht(isnan(ht))=0;
            % update the agents
            p.h(:,:,t+1) = ht;
            % update the agent accumulate performance (geometric returns)
            p.SH(t+1,:) =  p.SH(t,:) .* dSH;
            % update options
            p.s = s0;
            %% online update the learning
            p = learn(p,t);
        end
        %% learning
        function p=learn(varargin)
            % PATTERN/LEARN Machine Learning based on performance
            %
            % P = LEARN(P) The updates the aggregated agents and agent
            %   performance (B,S) using the specified learning type.
            %
            % P = LEARN(P,TYPE) will reset the learning
            %
            % Table 1: Learning types
            % +----------------+---------------------------------------+
            % | TYPE           | Description                           |
            % +----------------+---------------------------------------+
            % | 'univlearning' | Universal learning (log-optimal)      |
            % |                | [PARAM=[NONE]                         |
            % | 'eglearning'   | Exponentiated Gradient                |
            % |                | [PARAM=ETA in [0,1] typ. [0,0.2]      |
            % | 'ewlearning'   | EWMA in control based learning        |
            % |                | [PARAM=LAMBDA in [0,1] typ. [0.9,0.99]|
            % +----------------+---------------------------------------+
            %
            % References:
            % [1] Cover, T., M. (1991) Universal Portfolios
            % [2] Gyorfi, L., Udina, F., Walk, H., (2008) Experiments on universal
            %           portfolio selection using data from real markets
            % [3] Cover, T. M., (1996), Universal Portfolios with Side Information
            % [4] Algoet, P. H., Cover, T. M., (1980) Asymptotic optimality and
            %           symptotic equipartition properties of log-optimum investments
            % [5] Helmbold, D., P., Schapire, R., E., Singer, Y., Warmuth, M.,
            %           K.,(1998) On-line portfolio selection using multiplicative updates
            %
            % See Also: PATTERN/MATCH
            
            % SHX = exp(diff(log(SH)) for price path SH!
             
            %% input parameters
            p = varargin{1}; % (dim(x) = t) price t+1;
            % set defaults for optional inputs
            optargs = {p p.mnp(3)};
            % now put these defaults into the valuesToUse cell array,
            optargs(1:nargin) = varargin;
            % Place optional args in memorable variable names
            [p, t] = optargs{:};
            
            %% Machine Learning  
            if (size(p.h,1)==1)
                % only single agent
                % ---- ONLINE update ----
                b0 = p.h(1,:,t+1);
                qH0 = 1;
                % -----------------------    
            else
                % multiple agents
                switch p.ltype
                    case 'univlearning'
                        % compute weights
                        qH0 = p.SH(t+1,:);
                    case 'eglearning'
                        eta = p.lparam; % learning parameters
                        % update the weights using the exponentiated gradient
                        % algorithm
                        if all(p.qH(t,:)==0) || all(isnan(p.qH(t,:)))
                            qH0 = p.SH(t+1,:); % initialise
                        else
                            qH0 = p.qH(t,:) .* exp(eta .* ...
                                p.SH(t+1,:) ./ (p.qH(t,:) * p.SH(t+1,:)'));
                        end
                    case 'ewlearning'
                        lambda = p.lparam; % learning parameters
                        % update term
                        if all(p.qH(t,:)==0) || all(isnan(p.qH(t,:)))
                            qH0 = p.SH(t+1,:);
                        else
                            Z = (p.qH(t,:) .* p.SH(t+1,:))./ (p.qH(t,:) * p.SH(t+1,:)');
                            % update the weights using the exponentiated gradient algorithm
                            qH0 = lambda * p.qH(t,:) + (1-lambda) * Z;
                        end
                end
                %% renormalise the weights
                switch p.lntype
                    case 'active'
                        qH0 = (qH0 - mean(qH0));
                        norm0 = sum(abs(qH0));
                        if norm0>eps
                            qH0 = qH0 ./ norm0;
                        else
                            qH0 = zeros(size(qH0));
                        end
                    case 'absolute'
                        qH0 = qH0 ./ sum(qH0);
                end
                %% create the performance weighted combination of experts.
                % ONLINE
                % -------------------------------------------------------
                b0 = qH0 * p.h(:,:,t+1);
                % -------------------------------------------------------
                %% compute normalization abs(long) + abs(short)
                tb = nansum(abs(b0));
                % renormalize controls (leverage=1) [FIXME LEV]
                if tb==1
                elseif tb>eps
                    b0 = (1/tb) * b0;
                    % update the agent mixture weights for leverage
                    qH0 = (1/tb) * qH0;
                else
                    switch p.lntype
                        case 'absolute'
                            % update the agent mixture weights for leverage
                            qH0 = zeros(size(qH0)); % FIXME (should be equally weighted)
                            % zero weights
                            b0 = zeros(size(b0));
                    end
                end
            end % only 1 agent
            %% compute the leverage corrected output price relative
            % compute the updated returns
            dSH = p.SH(t+1,:)./p.SH(t,:);
            % uses the inputed online structure reference.
            if all(isnan(p.qH(t,:)))
                dS = 1;
            else
                dS = ((dSH-1) * transpose(p.qH(t,:)))+1; %  LINRET was exp
            end
            % update the properties
            p.qH(t+1,:) = qH0;
            p.b(t+1,:) = b0;
            p.S(t+1,:) = p.S(t,:) * dS;
        end
        %% partition
        function p=partition(varargin)
            % PATTERN/PARTITION Partition the data
            %
            % P = PARTITION(P) single partition: [111...1] for the
            %   partition type 'trivial'. NP=ELL as the number of partitions.
            %   The type of partition as TYPE.
            %
            % Table 1: Partition Types
            % +-------------+------------------------------------------+
            % | TYPE        | Description                              |
            % +-------------+------------------------------------------+
            % | 'trivial'   | [111...1] (single partition)             |
            % | 'overlap'   | [0...001][0...011][0...111]...[1...111]  |
            % | 'exclusive' | [100...0][010...0]...[000...1]           |
            % | 'sideinfo'  | partition using side-information         |
            % +-------------+------------------------------------------+
            % 
            % P = PARTITION(P,T) Relative to time T.
            %
            % Note 1: Side Information base learning will partition the data based
            % on the state of the side information. The required state will
            % then be used to determine which partition to use at time T
            % for the model estimation and learning.
            %
            % Note 2: Removes all days with the same returns as these are
            %   considered incorrect data days.
            %
            % See Also
            
            p = varargin{1}; % (dim = n-1);
            % set defaults for optional inputs
            optargs = {p p.mnp(3)};
            % now put these defaults into the valuesToUse cell array,
            optargs(1:nargin) = varargin;
            % Place optional args in memorable variable names
            [p, t] = optargs{:};
            % create the partitions
            switch p.ptype
                case 'trivial'
                    % trivial (default) partition [111...1]
                    p.p=true(1,t);
                case 'exclusive'
                    if any(size(p.ell)>1) && (floor(t/p.ell)<1), 
                        error('pattern:partition',...
                            'Only single ELL allowed for type EXCLUSIVE'); 
                    end
                    % ELL exclusive partitions
                    p.p=false(p.ell,t);
                    % number of partitions
                    ni = 0:floor(t/p.ell):t;
                    if ((t-ni(end))>0)
                        ni = ni(1:end-1); % drop the last partition
                        ni = [ni t]; % extend the last partition
                    end
                    for j=1:p.ell
                        p.p(j,ni(j)+1:ni(j+1))=true;
                    end
                case 'overlap'
                    % ELL overlapping partitions
                    if any(size(p.ell)>1) && (floor(t/p.ell)<1), 
                        error('pattern:partition',...
                            'Only single ELL allowed for type EXCLUSIVE'); 
                    end
                    % ELL exclusive partitions
                    p.p=false(p.ell,t);
                    % number of partitions
                    ni = 0:floor(t/p.ell):t;
                    for j=1:p.ell
                        p.p(j,ni(j)+1:t)=true;
                    end
                case 'sideinfo'
                    % ELL references the side-information factor
                    %
                    % 1. use one of the factors to partition based on
                    %   side-information.
                    % 2. The side-information factor will be excluded
                    %   from the nearest-neighbour calculation
                    error('pattern:partition','Side Information unsupported');
            end
        end
        %% sanity check data
        function p = sanitycheck(p)
            % SANITYCHECK Remove insane data days (all zero, nan, 1 @T)
        end
        function p = commutecheck(p)
            % COMMUTECHECK Check that the commutation of controls holds
            %
            % Figure 1: Commutation of the weight b, and agents H using
            %           agents weights q and the returns of the stocks r
            % +-----+         +--------+      +-----------------------+
            % |(r,H)|   q ->  | b= q H | N -> | tilde b (1/lambda) b  |
            % +-----+         +--------+      +-----------------------+
            %    |r               | r                       |r
            % +--------+     +-----------+      +----------------------+
            % |SH = H r|q -> | S = q' SH | N -> | S = (1/lambda) q' SH |
            % +--------+     | S = b r   |      | S = tilde b r        |
            %                +-----------+      +----------------------+
            %
            % See Also:
            
            % r,H -> b = q H -> S = b r
            
            % S = H r <-> S = br
        end
        %% subscript assignment 
        function p = subsasgn(p,s,b)
            % PATTERN/SUBSASGN Subscript assigment
            %
            % P.<VarName> = Value
            %
            % See Also: PATTERN/SUBSREF
            
            varName = s(1).subs;
            switch varName
                case {'learntype'}
                    p.ltype = b;
                    if ~isempty(p.SH) && ~all(all(p.SH==1))
                        switch p.ltype
                            case 'univlearning'
                                p.lparam = Inf;
                            case 'eglearning'
                                p.lparam = 0.01;
                            case 'ewlearning'
                                p.lparam = 0.99;
                            otherwise
                                error('pattern:subsasgn:learntype',...
                                    'Unrecognized Learning');
                        end
                        % update the learning
                        tmin0 = 3*max(p.ell);
                        t0 = p.mnp(3);
                        if t0<tmin0
                            error('pattern:subsasgn:learntype','Not enough Data L*K>T');
                        end
                        % Find matching times j for pattern [offline loop]
                        for t=tmin0:t0 % time loop
                            p = learn(p,t);
                        end
                    end
                case {'learnparam'}
                    p.lparam = b;
                    if ~isempty(p.SH) && ~all(all(p.SH==1))
                        % update the learning
                        tmin0 = 3*max(p.ell);
                        t0 = p.mnp(3);
                        if t0<tmin0
                            error('pattern:subsasgn:learnparam','Not enough Data L*K>T');
                        end
                        % Find matching times j for pattern [offline loop]
                        for t=tmin0:t0 % time loop
                            p = learn(p,t);
                        end
                    end
                case {'learnnorm'}
                    p.lntype = b;
                    if ~isempty(p.SH) && ~all(all(p.SH==1))
                        % update the learning
                        tmin0 = 3*max(p.ell);
                        t0 = p.mnp(3);
                        if t0<tmin0
                            error('pattern:subsasgn:learnparam','Not enough Data L*K>T');
                        end
                        % Find matching times j for pattern [offline loop]
                        for t=tmin0:t0 % time loop
                            p = learn(p,t);
                        end
                    end
                otherwise
                    error('pattern:subsref','Incorrect reference');
            end
        end
        %% subscript reference by time
        function p = subsref(p,s)
            % PATTERN/SUBSREF Sybscript reference
            %
            % P = P(T) subscript reference out the pattern object over time
            %   range T.
            %
            % P.<VarName> to reference out the required properties such as
            %   H the agent controls, SH the accumulated geometric agent
            %   performance, B the aggregated controls, and the aggregated
            %   agent performance.
            %
            % Examples: p.h(:,:,end)
            %
            % See Also: PATTERN/SUBSASGN
            
            switch s(1).type
                case '()'
                    if size(s.subs,2)==1
                        % subscript reference object by time
                        % (M,N,P) -> (:,:,P0)
                        t0=s(1).subs{1};
                        p.x = p.x(:,:,t0);
                        p.mnp = size(p.x);
                        % retain the internal state history
                        t1 = [t0 t0(end)+1]; % for online functionality
                        p.h = p.h(:,:,t1); %
                        p.SH = p.SH(t1,:);
                        p.b = p.b(t1,:);
                        p.S = p.S(t1);
                        p.qH = p.qH(t1,:);
                        % re-partition
                        p = partition(p);
                    elseif size(s.subs,2)==2
                        % subscript reference object by time and clusters
                        % (M,N,P) -> (M0,:,P0)
                        error('pattern:subsref','CI subsref not supported');
                    end
                case '.'
                    % A reference to a variable or a property.  Could be any sort of
                    % subscript following that.  Row names for () and {} subscripting
                    % on variables are inherited from the dataset.
                    varName = s(1).subs;
                    switch varName
                        case {'h','SH','S','b'}
                            p = p.(varName);
                            if size(s,2)==2
                                p=subsref(p,s(2));
                            end
                        case {'ci'}
                            p = p.ci;
                        case {'cin'}
                            % number of clusters
                            p = size(p.ci,1);
                        otherwise
                            error('pattern:subsref','Incorrect reference');
                    end
                otherwise
                    error('pattern:subsref','Incorrect reference');
            end
        end
        function display(p)
            % PATTERN/DISPLAY Display the Pattern object
            %
            % See Also PATTERN 
            
            disp(p);
            if ~isempty(p.x)
                fprintf('\tParameters\n');
                fprintf('\t--------------------\n');
                fprintf('\tk-tuples       : %s\n',num2str(p.k));
                fprintf('\tell neighbours : %s\n',num2str(p.ell));
                fprintf('\t#clusters      : %d\n',size(p.ci,1));
                fprintf('\tlearn param.   : %3.2f\n',p.lparam);
                fprintf('\thorizon        : %d\n',p.horz);
                fprintf('\n');
                fprintf('\tData x(M,N,P)\n');
                fprintf('\t--------------------\n');
                fprintf('\tobjects (N) : %d\n',p.mnp(1));
                fprintf('\tfactors (M) : %d\n',p.mnp(2));
                fprintf('\tstates  (P) : %d\n',p.mnp(3));
                fprintf('\tpartitions  : %d\n',size(p.p,1));
                fprintf('\n');
                fprintf('\tAlgorithm Constraints\n');
                fprintf('\t--------------------\n');
                fprintf('\tmatching cons. : %s\n',p.ntype);
                fprintf('\tlearning       : %s\n',p.ltype);
                fprintf('\tlearning cons. : %s\n',p.lntype);
                fprintf('\tpartition      : %s\n',p.ptype);
            end
        end
    end % end methods
end
%% helper functions
function [w,s]=quadbet(varargin)
% QUADBET Quadratic optimal bet
%
% W = QUADBET(f,H,gamma) Solve the optimal tactical bet. The expected
%   view F and covariance matrix H are required as is the risk aversion
%   GAMMA.
%
% W(1) is the benchmark bet (fully invested minimum variance
%   portfplio).
% W(2) is the tactical bet based on the long-term eqiulibrium
%   view F. This is by default zero.
%
% use speudo inverse pinv(omega) is possible is conditioning is an issue

%% initialise the inputs
f = varargin{1};
H = varargin{2};
s = [];
% initialise inputs
optargs = {f H 1 s};
% now put these defaults into the valuesToUse cell array,
optargs(1:nargin) = varargin;
% Place optional args in memorable variable names
[f, H, gamma, s] = optargs{:};

%% compute the optimal portfolio
% get the size of the covariance matrix
[m,n] = size(H);
% diagonals of one
I = ones(m,1);
% invert the covariance matrix
invHI = H\I;
invH = inv(H);
% benchmark weights
w(:,1) = (I' * invHI) \ (invHI);
% active weights
w(:,2) = (1/gamma) * (I' * invHI) \ (invH * ( f(:) * I' - I * f(:)') * invH ) * I;

end
function [hkl,s] = match(varargin)
% PATTERN/MATCH Pattern Match for a given K-tuple and matching data set.
%
% [HKL]=MATCH(X0,P0,XNK,L0,NTYPE,TREND,HORZ) X0 are the factor relatives. P0 the 
%   partition. K0 is the k-tuple size over the current data set, it is 
%   computed from the user definied pattern (k-tupel) XNK. XNK is a MxNxK0 
%   double for X0 a MxNxP size double for T>>K0. L0 is the number of 
%   neighbours to include from the partitioning P0 of the data. NTYPE is 
%   the normalisation type. This excludes the time loop over the the 
%   matching horizon. This is the online version of the pattern matching 
%   and learning algorithm. HKL is a Mx1 control vector that satifies the 
%   normalisation type NTYPE. 
%
% See Also: PATTERN/ONLINE, PATTERN/OFFLINE, QUADBET

% Author: Tim Gebbie

%% initialise the input variables
x0 = varargin{1};
p0 = varargin{2};
xnk = varargin{3};
optargs = {x0,p0,xnk,20,'active',1,{}};
% now put these defaults into the valuesToUse cell array,
optargs(1:nargin) = varargin;
% Place optional args in memorable variable names % Edited by FL ( added L
% as arg)
[x0,p0,xnk,ell0,ntype,horz,s,L] = optargs{:};

%% Loop parameters wrt to partition
q0 = size(p0,1); % number of partitions
m0 = size(x0,1); % number of objects

%% Portfolio agent constraints
switch ntype
    case 'absolute'
        hkl = (1/m0)*ones(1,m0);
    case 'active'
        hkl = zeros(1,m0);
    case 'gyorfi_opt' % Edited by FL
        hkl = (1/m0)*ones(1,m0);
end
% the pattern as returns computed from price relatives
k0 = size(xnk,3);
% check consistency
if any(size(k0)>1), error('portchoice:pattern:match','Incorrect K'); end
if any(size(ell0)>1), error('portchoice:pattern:match','Incorrect L'); end

%% Find matching times j for pattern
for q=q0 % maximum of ell partitions [P(ell)][temporal]
    % the partition
    psi = p0(q,:);
    % for agent h(k,ell) for cluster ci, and partition psi
    snk = x0(:,:,psi)-1; 
    % find the times (this allows for inhomogenous partitions)
    jell = find(psi);
    % reset the distance measure for partition
    ed = Inf(size(snk,3),m0);
    %% get the test tuples by looping over partition snk
    for j=k0:size(snk,3) - horz % only allow jell+1 in partition
        % distance (element-wise difference)
        edi = xnk - snk(:,:,j-k0+1:j);
        
        if (k0==1)
            % 2-norm computed sum_i sqrt(a_i^2)
            ed(j,1:m0) = norm(edi);
        else
            % reshape the distance by objects and factors (Edited by FL)
            % computes distance like Gyorfi
            edi = reshape(edi,size(edi,1),size(edi,2)*size(edi,3));
            ed(j,1:m0) = norm(edi);
            % 2-norm computed for each object independently (Edited by FL)
%             for a=1:m0
%                 ed(j,a) = norm(edi(a,:))
%                 norm(edi(a,:))
%                 norm(edi)
%             end
        end
    end % j segment loop
    %% sort the matching times
    
    % Edited by FL
    
    pl = 0.02 + 0.5*((ell0-1)/(L-1));
    pell0 = floor(pl*j);
   
    
    if (q0==1)
        % ell matches in a single partition
        [~,nj]=sort(ed,'ascend'); % ~ -> pat.ed not required
        
        % Take into account ties of the norm (Edited by FL)
        if pell0>0
            while pell0<length(nj) && (ed(nj(pell0))==ed(nj(pell0+1)))
                pell0 = pell0 + 1;
            end
        end
        
        % update the matching times (Edited by FL)
        njell = jell(nj(1:pell0,:));
        % update the norm (Edited by FL)
        ed = ed(nj(1:pell0,:));
        
        % update the distances
        % pat.ed = pat.ed(nj(pat.ell0,:));
        % find the ell matching times
        % --- LOOK AHEAD RULE ------------
        jnk = njell+horz; % 1-step look ahead
        % --------------------------------
    else
        % find the single closest match in the ell partitions
        [~,nj]=min(ed); % ~ -> pat.ed not required
        % find the matching time
        njell = jell(nj);
        % the uncertainty as the norm
        ed(q) = ed(nj);
        % update the matching times
        % --- LOOK AHEAD RULE ---------------
        jnk(q) = njell+horz; % 1-step look ahead
        % -----------------------------------
    end
end % end partition loop
%% find the predictions (Edited by FL)
E = eye(m0,m0);
hatx = zeros(pell0,m0);
% initialise prediction vector
for mj = 1:m0 % loop over objects
    % select first factor (price relative) @JNK matching times
    hatx(:,mj) = x0(mj,1,jnk(:,mj))-1; % r = R - 1
    % matching error (diag covariance matrix)
    E(mj,mj) = mean(ed(:,mj).^2);
end % loop over objects/stocks
%% find the agents using mean-variance approximation

% -------------------------------
f = mean(hatx,1); % expected view : r = R-1

% -------------------------------
% E = (1/sum(diag(E)))*E; % relative distance error
% T = mean(range(hatx,2)) * eye(size(E)); % entropy
% construct the experts for the k,ell choice
if size(hatx,1) > 2
    H = cov(hatx); % r = R-1
    if any(diag(H)==0)
        % remove zero covariance
        zi = (diag(H)==0);
        H(zi,zi) = nanmean(diag(H))*eye(sum(zi));
    elseif all(H==0)
        % all are zero use diagonal matrix
        H=eye(m0);
    end
else
    H = eye(m0);
end
if rcond(H)<1e-2,
    % warning('portchoice:newbcrp','Bad H conditioning using diag(H)');
    H = diag(diag(H));
    if rcond(H)<1e-2,
        H=eye(m0);
    end
end;
% combine intrinsic uncertainty with estimation uncertainty
% ------------------------------
% H = H + E + T; % intrinsic + estimation + entropy
% H = H + E; % intrinsic + estimation
% H = H; % intrinsic uncertainty
% ------------------------------
% approximate log optimal k,l-th expert using quadratic approx.
% ----------------------------
[hkl0,s] = quadbet(f,H);
% ----------------------------
switch ntype
    case 'absolute'
        % fully invested
        hkl = sum(hkl0,2);
        if sum(abs(hkl))>1+eps
            % find the biggest short-sold asset
            [~,i0] = min(hkl);
            % aggressiveness factor
            n0 = abs(hkl0(i0,1) / hkl0(i0,end));
            % rescale the tactical portfolio
            hkl = hkl0(:,1) + n0 * hkl0(:,end);
        end
    case 'active'
        % cash neutral
        hkl = hkl0(:,end);
        % leverage unity
        hkl = hkl/sum(abs(hkl));
    case 'gyorfi_opt' % Edited by FL
        optfun = inline('-log(prod(x*transpose(b)))','b','x');
        xx = hatx +1;
        [~, m] = size(xx);
        b0 = (1/m)*ones(1,m);
        [hkl2, ~] = fmincon(@(b) optfun(b,xx),b0,[-eye(m);eye(m)],...
            [zeros(1,m)';ones(1,m)'],ones(1,m),1,[],[],[],...
            optimset('Algorithm','sqp','Display','off'));
        hkl = hkl2';
end
end