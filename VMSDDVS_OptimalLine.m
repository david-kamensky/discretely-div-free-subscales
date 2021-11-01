function [Optimal_H1] = VMSDDVS_OptimalLine(x,y)
%% Author: Sajje <Sajje@COMA-PC>
%% Created: 2021-10-31

%This function generates new data points, based on the input data read from
%the .csv files, that follow the exepected optimal convergence rate.

%Make sure that the required .csv files are located in the same directory
%as this function.

%% Initialize Constants and Optimal Values Vector:
%Expected Optimal Convergence Rate:
EORC = 2;   %The slope of the optimal convergence rate should be 2 in the log-log plot.

%Vector containing the optimal H1 values to be plotted in the convergence
%function:
Optimal_H1 = zeros(size(x,1),size(x,2));
%% Generate Optimal Values Vector:
%First generate the starting value in Optimal_H1:
Optimal_H1(1,1) = y(1,1);

%Now generate the remaining values in Optimal_H1:
for i = 2:size(x,1)
    %Since the simulations were limited by computing resources, H1 velocity
    %data could not be obtained for "very small" element size h. In the
    %.csv files, the H1-error is listed as "Out of Memory" when the
    %computer literally runs out of memory and crashes when running the
    %simulation. The "Out of Memory" becomes a "NaN" value when MATLAB
    %reads the .csv files.
    
    %The if statement below has to be used or else the optimal H1 vector
    %will have more entries than the input H1 (y) vector. The sizes of the
    %H1 vectors must be consistent or the resulting plots will look ugly.
    
    %If H1 error has been obtained, then plot:
    if double(isnan(y(i,1))) == 0
        Optimal_H1(i,1) = exp(EORC*(log(x(i,1)) - log(x(1,1))) + log(Optimal_H1(1,1)));
    %If computer ran out of memory, then place a NaN value in the vector:
    elseif double(isnan(y(i,1))) == 1
        Optimal_H1(i,1) = NaN;
    end
end


end

