function [phase,amp,freq] = apfft(mode,rdata,bounds,window)
    if strcmp(mode,'cor')
        [phase,amp,freq] = apfft_cor(rdata,bounds,window);
    elseif strcmp(mode,'uncor')
        [phase,amp,freq] = apfft_ext(rdata,bounds,window);
    end
end

function [phase,amp,freq] = apfft_cor(rdata_in,bounds,window)
    ndata=size(rdata_in,1);
    if ndata==1
        rdata_in = rdata_in';
    end
    ndata=size(rdata_in,1);
    
    N = floor((ndata+1.0)/3.0);
    ndata = 3*N-1;

    rdata = rdata_in(1:ndata) - mean(rdata_in(1:ndata));

    u1 = rdata(1:2*N-1);
    u2 = rdata(N+1:3*N-1);

    [phase_u1,amp_u1,freq_u1] = apfft_ext(u1,bounds,window);
    [phase_u2,amp_u2,freq_u2] = apfft_ext(u2,bounds,window);

    d = (phase_u2-phase_u1)/2.0/pi;
    if d > 0.5
        d = d - 1.0;
    elseif d <= -0.5
        d = d + 1.0;
    end

    freq = freq_u1 + (1.0*d)/N;

    phase = 2.0*phase_u1 - phase_u2;
    if phase < 0
        phase = phase + 2.0*pi;
    elseif phase > 2.0*pi
        phase = phase - 2.0*pi;
    end

    if strcmp(window,'rec')
        if isfinite(1.0/sin(pi*d))
            amp = (pi*d/sin(pi*d))^2 * amp_u1 * 2.0;
        else
            amp = 2.0d0 * amp_u1;
        end
    elseif strcmp(window,'han')
        if isfinite(1.0/sin(pi*d))
            amp = (pi*d*(1.0-d*d)/sin(pi*d))^2 * amp_u1 * 2.0;
        else
            amp = 2.0d0 * amp_u1;
        end
    end
end

function [phase,amp,freq] = apfft_ext(rdata_in, bounds, window)
    ndata = size(rdata_in,1);
    rdata = rdata_in(1:ndata) - mean(rdata_in(1:ndata));
    N = floor((ndata+1.0)/2.0);
    cXap = complex(zeros(N,1),0);
    
    if strcmp(window,'rec')
        % rec-rec window
        cXap(1) = complex(N*rdata(N),0.0);
        for i=2:N
            cXap(i) = complex((N-i+1)*rdata(N+i-1) + (i-1)*rdata(i-1),0.0);
        end
        cXap = cXap/N/N;
    elseif strcmp(window,'han')
        rdata = hanhan(N).*rdata;
        cXap(1) = complex(rdata(N),0.0);
        for i=2:N
            cXap(i) = complex(rdata(N+i-1) + rdata(i-1),0.0);
        end
    end
    cXap = N*ifft(cXap);
    lb = round(N*bounds(1))+1;
    ub = round(N*bounds(2))+1;
    [~,maxloc] = max(abs(cXap(lb:ub)));
    maxix = maxloc + lb - 1;
    phase = twopi_atan2(imag(cXap(maxix)),real(cXap(maxix)));
    amp = abs(cXap(maxix));
    freq = (maxix-1.0)/N;
end

function arg = twopi_atan2(Y,X)
    arg = atan2(Y,X);
    if(arg <= 0)
        arg = arg+2*pi;
    end
    arg = 2*pi - arg;    
end

function hh = hanhan(N)
    h = han(N);
    hh = zeros(2*N-1,1);
    for i=1:N
        hh(i) = sum( h(N-i+1:N) .* h(1:i) );
        hh(2*N-i) = hh(i);
    end
end

function result = han(N)
    result = sin(pi*(1:N)/(N-1.0)).^2 / (N-1.0) * 2;
end
