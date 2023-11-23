function run_izhikevich_sim(seed, Ne, Ni, S0, inpmean, inpvar, p, T, normalizevol, showplot, outfolder, ...
    mvgc, mvgcdata, momaxs)
    % Inputs:
    % seed: random seed for reproducibility.
    % Ne: number of excitatory neurons
    % Ni: number of inhibitory neurons
    % S0: synaptic strength
    % inpmean: external input mean
    % inpvar: external input variance
    % p: connection probability (0<=p<=1)
    % T: total simulation time steps
    % normalizevol: if True, normalize voltage data
    % showplot: if True, show plots
    % outfolder: directory for saving data
    % mvgc: if True, run MVGC analysis
    % mvgcdata: data to analyze with MVGC (options: 'V', 'Vnorm')
    % momaxs: array, MVGC maximum model orders

    %% Build network
    rng(seed); 
    re=rand(Ne,1);          ri=rand(Ni,1);
    a=[0.02*ones(Ne,1);     0.02+0.08*ri];
    b=-0.1*ones(Ne+Ni,1);
    c=[-65+15*re.^2;        -65*ones(Ni,1)];
    d=[8-6*re.^2;           2*ones(Ni,1)];
    params = [a,b,c,d];
    inpstd = sqrt(inpvar);
       
    % random network
    P0 = rand(Ne+Ni);  % a (Ne+Ni)x(Ne+Ni) matrix with uniformly distributed random numbers
    S = P0 < p;
    mask = [ones(Ne+Ni,Ne), -ones(Ne+Ni,Ni)];  % distinguish between excitatory and inhibitory connections
    S = S.*mask;
    S = S - diag(diag(S));  % zero out the diagonal elements
    S = S0*S;
    
    figure(1)
    imagesc(S)
    colorbar
    hold on
    lx = linspace(0,Ne+Ni);
    ly = linspace(0,Ne+Ni);
    plot(lx,ly, color='k')
    set(gca,'xaxisLocation','top')
    xlabel('neuron number') 
    ylabel('neuron number')
    title('connectivity matrix')
    hold off
    %% Simulate activities
    % code adapted from https://www.izhikevich.org/publications/spikes.htm

    v=-65*ones(Ne+Ni,1);    % Initial values of v
    u=b.*v;                 % Initial values of u
    firings=[];             % spike timings
    V = [v];  % n_neuron x total_time
    U = [u];  % n_neuron x total_time
    inps = [];  % n_neuron x total_time
    
    for t=1:T            % simulation of T ms
      % find fired neurons
      fired=find(v>=30);    % indices of spikes
      firings=[firings; t+0*fired,fired];
      v(fired)=c(fired);
      u(fired)=u(fired)+d(fired);
      I=[inpmean+inpstd*randn(Ne,1);inpmean+inpstd*randn(Ni,1)]; % thalamic input
      inps = [inps, I];
      I=I+sum(S(:,fired),2); % add synaptic input
      v=v+0.5*(0.04*v.^2+4.1*v+108-u+I); % step 0.5 ms
      v=v+0.5*(0.04*v.^2+4.1*v+108-u+I); % for numerical
      u=u+a.*(b.*v-u);                 % stability
      V = [V, v];
      U = [U, u];
    end;

    if showplot
        figure(2)
        plot(firings(:,1),firings(:,2),'.') % col1: time, col2: neuron number
        hold on
        plot([0, 1000], [Ne, Ne], '-k')
        xlabel('time [ms]') 
        ylabel('neuron number') 
        title('spike raster for t=1:1000')
        hold off
        
        figure(3)
        sgtitle('membrane potential for t=1:1000')
        for i=1:(Ne+Ni)
            subplot(Ne+Ni,1,i);
            x = 1:1000;
            y = V(i, 1:1000);
            plot(x, y)
            title(['neuron ', num2str(i)])
        end

    end
    %% Save data
    formatSpec = outfolder + "p%d/inpvar%d_mean%d_t%d/seed%d/";
    folder = sprintf(formatSpec,p*10, inpvar, inpmean, T, seed);
    disp(folder)
    mkdir(folder)
    writematrix(firings, folder+"firings.txt");
    writematrix(V, folder+"v_alltimes.txt")
    writematrix(inps, folder+"inp.txt")
    writematrix(U, folder+"u_alltimes.txt")
    writematrix(params, folder+"params_abcd.txt")
    writematrix(S, folder+"connectivity.txt")

    %% Process data

    if normalizevol
        V1 = V;
        % 1. replace all firings with threshold voltage 30
        V1(V1>30) = 30;
        % 2.normalize each neuron by its mean and std
        V_normed = zscore(V1,1,2);

        if showplot
            figure(4)
            sgtitle('normalized membrane potential for t=1:1000')
            for i=1:(Ne+Ni)
                subplot(Ne+Ni,1,i);
                x = 1:1000;
                y = V_normed(i, 1:1000);
                plot(x, y)
                title(['neuron ', num2str(i)])
            end
        end
        writematrix(V_normed, folder+"v_normed_alltimes.txt")
    end
    
    %% MVGS analysis
    % code adapted from mvgc_demo_statespace.m at https://www.mathworks.com/matlabcentral/fileexchange/78727-the-multivariate-granger-causality-mvgc-toolbox
    
    if mvgc
        switch mvgcdata
            case 'V'  % use voltage
                X = V; 
            case 'Vnorm'  % use normalized voltage
                X = V_normed;
        end

        % MVGC parameters
        ntrials   = 1;     % number of trials
        nobs      = T;   % number of observations per trial
        regmode   = 'LWR';  % VAR model estimation regression mode ('OLS', 'LWR' or empty for default)
        icregmode = 'LWR';  % information criteria regression mode ('OLS', 'LWR' or empty for default)        
        tstat     = 'F';    % statistical test for MVGC:  'F' for Granger's F-test (default) or 'chi2' for Geweke's chi2 test
        alpha     = 0.05;   % significance level for significance test
        mhtc      = 'FDRD'; % multiple hypothesis test correction (see routine 'significance')
        fs        = 1;    % sample rate (Hz)  -- not very relevant here
        criteria = {'AIC', 'BIC'}; 
        
        for momax = momaxs
        disp(momax);
        % Calculate information criteria up to specified maximum model order.
        ptic('\n*** tsdata_to_infocrit\n');
        [AIC,BIC,moAIC,moBIC] = tsdata_to_infocrit(X,momax,icregmode);
        ptoc('*** tsdata_to_infocrit took ');
        % Plot information criteria.
        figure(5); clf;
        plot_tsdata([AIC BIC]',{'AIC','BIC'},1/fs);
        title('Model order estimation');        
        fprintf('\nbest model order (AIC) = %d\n',moAIC);
        fprintf('best model order (BIC) = %d\n',moBIC);        
        % Select model order using both AIC and BIC
        for c = 1:length(criteria) 
            cri = criteria{c};
            disp(cri)
        if strcmpi(cri,'AIC')
            morder = moAIC;
            fprintf('\nusing AIC best model order = %d\n',morder);
        elseif strcmpi(cri,'BIC')
            morder = moBIC;
            fprintf('\nusing BIC best model order = %d\n',morder);
        else
            fprintf('\nusing specified model order = %d\n',cri);
        end

        % Estimate VAR model of selected order from data.
        ptic('\n*** tsdata_to_var... ');
        [A,SIG] = tsdata_to_var(X,morder,regmode);
        assert(~isbad(A),'VAR estimation failed - bailing out');
        ptoc;
        
        % Report information on the estimated VAR, and check for errors.
        %
        % _IMPORTANT:_ We check the VAR model for stability and symmetric
        % positive-definite residuals covariance matrix. _THIS CHECK SHOULD ALWAYS BE
        % PERFORMED!_ - subsequent routines may fail if there are errors here. If there
        % are problems with the data (e.g. non-stationarity, colinearity, etc.) there's
        % also a good chance they'll show up at this point - and the diagnostics may
        % supply useful information as to what went wrong.
        
        info = var_info(A,SIG);
        assert(~info.error,'VAR error(s) found - bailing out');

        % Calculate time-domain pairwise-conditional causalities from VAR model parameters
        % by state-space method [4]. The VAR model is transformed into an equivalent state-
        % space model for computation. Also return p-values for specified test (F-test or
        % likelihood-ratio test; this is optional - if p-values are not required, then it
        % is not necessary to supply time series |X|, regression mode |regmode|, or test
        % specification |tstat|).
        
        ptic('*** var_to_pwcgc... ');
        [F,pval] = var_to_pwcgc(A,SIG,X,regmode,tstat);
        ptoc;
        
        % Check for failed GC calculation
        
        assert(~isbad(F,false),'GC calculation failed - bailing out');
        
        % Significance-test p-values, correcting for multiple hypotheses.
        
        sig = significance(pval,alpha,mhtc);
        
        % Plot time-domain causal graph, p-values and significance.
        
        figure(5+c); clf;
        sgtitlex(['Pairwise-conditional Granger causality - time domain' (cri)]);
        subplot(1,3,1);
        plot_pw(F);
        colorbar;
        title('Pairwise-conditional GC');
        subplot(1,3,2);
        plot_pw(pval);
        colorbar;
        title(['p-values (' tstat '-test)']);
        subplot(1,3,3);
        plot_pw(sig);
        title(['Significant at \alpha = ' num2str(alpha)]);
        colorbar;
        
        switch cri
            case 'AIC'
                formatSpec = folder + "mvgc_aic_F_p%d.txt";
                filename = sprintf(formatSpec,momax);
                writematrix(F, filename);
            case 'BIC'
                formatSpec = folder + "mvgc_bic_F_p%d.txt";
                filename = sprintf(formatSpec,momax);
                writematrix(F, filename);
        end
        end
        end
    end
end