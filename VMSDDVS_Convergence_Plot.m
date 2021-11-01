function [] = VMSDDVS_Convergence_Plot()
%% Author: Sajje <Sajje@COMA-PC>
%% Created: 2021-10-07

%This function reads the data from the .csv files makes a log-log plot
%of the H1-Error against the grid-size h. Each plot also contains a line
%plotting the slope of the expected/optimal convergence rate of the
%H1-Error.

%Make sure that the required .csv files are located in the same directory
%as this function.

clear all;          %Clear all variables from workspace.
clc;                %Clear command window.
warning('off');     %Disable warnings related to readtable.

%% Obtain data from .csv files.
%Taylor-Green vortex benchmark problem using Isogeometric Taylor-Hood
%elements at Re = 100.
IGA_TaylorGreen_Re100_DS = readtable('MAE 299 Convergence - Isogeometric Taylor Hood - Re100 tg dynamic subscale.csv');
IGA_TaylorGreen_Re100_QS = readtable('MAE 299 Convergence - Isogeometric Taylor Hood - Re100 tg quasi-static subscale.csv');

%Taylor-Green vortex benchmark problem using regular Taylor-Hood
%elements at Re = 100.
Reg_TaylorGreen_Re100_DS = readtable('MAE 299 Convergence - Taylor Hood - Re100 tg dynamic subscale.csv');
Reg_TaylorGreen_Re100_QS = readtable('MAE 299 Convergence - Taylor Hood - Re100 tg quasi-static subscale.csv');

%Regularized Lid-Driven Cavity benchmark problem using regular and isogeometric Taylor-Hood
%elements at Re = 100. Dynamic Subscales are not used in the regularized
%lid-drive cavity problem since the time derivative terms are zero.
IGA_LDC_Re100 = readtable('MAE 299 Convergence - Isogeometric Taylor Hood - Re100 ldc.csv');
Reg_LDC_Re100 = readtable('MAE 299 Convergence - Taylor Hood - Re100 ldc.csv');

%% Plot convergence data for Taylor-Green problem using Isogemeotric Taylor-Hood elements.
figure(1);
%IGA Taylor-Green, Dynamic Subscale Data:
x_DS = table2array(IGA_TaylorGreen_Re100_DS(:,2));
y_DS = table2array(IGA_TaylorGreen_Re100_DS(:,3));
%IGA Taylor-Green, Quasi-Static Subscale Data:
x_QS = table2array(IGA_TaylorGreen_Re100_QS(:,2));
y_QS = table2array(IGA_TaylorGreen_Re100_QS(:,3));
%Generated "optimal convergence rate" data:
OF = 0.6;                                      %Offset Factor for y.
x_OPT = x_DS;                                  %x optimal values.
[y_OPT] = VMSDDVS_OptimalLine(x_DS,OF*y_DS);   %y optimal values.

%Plot convergence data:
loglog(x_DS,y_DS,'LineStyle','-','Marker','o','Color','blue');
hold on;
loglog(x_QS,y_QS,'LineStyle','-.','Color','red');
loglog(x_OPT,y_OPT,'LineStyle','--','Color','blue');
axis([7.2e-3 1.1e-1 9e-4 5e-1])
set(gca,'FontSize',14,'FontName','Times New Roman') 
title('$H^1$ Error of Velocity Field (Isogeometric)','FontSize',14,'Interpreter','Latex');
xlabel('h','FontSize',14,'Interpreter','Latex');
ylabel('${\left\vert\kern-0.2ex\left\vert \textbf{u} - \textbf{u}^h \right\vert\kern-0.2ex\right\vert}_{H^1 \left(\Omega\right)}$','FontSize',16,'Interpreter','Latex');
set(gca,'FontSize',14,'FontName','Times New Roman')
le = legend({'Dynamic','Quasi-Static','Optimal'},'Location','northwest');
set(le,'FontSize',14,'Interpreter','Latex');
set(gcf,'Position',[350 150 700 500]);
saveas(gcf,'Convergence - Isogeometric Taylor-Green','png')
hold off;

%% Plot convergence data for Taylor-Green problem using regular Taylor-Hood elements.
figure(2);
%Regular Taylor-Green, Dynamic Subscale Data:
x_DS = table2array(Reg_TaylorGreen_Re100_DS(:,2));
y_DS = table2array(Reg_TaylorGreen_Re100_DS(:,3));
%Regular Taylor-Green, Quasi-Static Subscale Data:
x_QS = table2array(Reg_TaylorGreen_Re100_QS(:,2));
y_QS = table2array(Reg_TaylorGreen_Re100_QS(:,3));
%Generated "optimal convergence rate" data:
OF = 0.3;                                      %Offset Factor for y.
x_OPT = x_DS;                                  %x optimal values.
[y_OPT] = VMSDDVS_OptimalLine(x_DS,OF*y_DS);   %y optimal values.

%Plot convergence data:
loglog(x_DS,y_DS,'LineStyle','-','Marker','o','Color','blue');
hold on;
loglog(x_QS,y_QS,'LineStyle','-.','Color','red');
loglog(x_OPT,y_OPT,'LineStyle','--','Color','blue');
axis([4.5e-3 1.15e-1 4e-4 7.8e-1])
set(gca,'FontSize',14,'FontName','Times New Roman') 
title('$H^1$ Error of Velocity Field (Regular)','FontSize',14,'Interpreter','Latex');
xlabel('h','FontSize',14,'Interpreter','Latex');
ylabel('${\left\vert\kern-0.2ex\left\vert \textbf{u} - \textbf{u}^h \right\vert\kern-0.2ex\right\vert}_{H^1 \left(\Omega\right)}$','FontSize',16,'Interpreter','Latex');
set(gca,'FontSize',14,'FontName','Times New Roman')
le = legend({'Dynamic','Quasi-Static','Optimal'},'Location','northwest');
set(le,'FontSize',14,'Interpreter','Latex');
set(gcf,'Position',[350 150 700 500]);
saveas(gcf,'Convergence - Regular Taylor-Green','png')
hold off;

%% Plot convergence data for Lid-Driven Cavity case.
figure(3);
%IGA Lid-Driven Cavity Data:
x_IGA_LDC = table2array(IGA_LDC_Re100(:,2));
y_IGA_LDC = table2array(IGA_LDC_Re100(:,3));
%Regular Lid-Driven Cavity Data:
x_Reg_LDC = table2array(Reg_LDC_Re100(:,2));
y_Reg_LDC = table2array(Reg_LDC_Re100(:,3));
%Generated "optimal convergence rate" data:
OF = 0.2;                                                %Offset Factor for y.
x_OPT = x_Reg_LDC;                                       %x optimal values.
[y_OPT] = VMSDDVS_OptimalLine(x_Reg_LDC,OF*y_Reg_LDC);   %y optimal values.

%Plot convergence data:
loglog(x_IGA_LDC,y_IGA_LDC,'LineStyle','-','Marker','o','Color','blue');
hold on;
loglog(x_Reg_LDC,y_Reg_LDC,'LineStyle','-.','Color','red');
loglog(x_OPT,y_OPT,'LineStyle','--','Color','blue');
axis([4.5e-3 1.2e-1 5e-5 2e-1])
set(gca,'FontSize',14,'FontName','Times New Roman') 
title('$H^1$ Error of Velocity Field (LDC)','FontSize',14,'Interpreter','Latex');
xlabel('h','FontSize',14,'Interpreter','Latex');
ylabel('${\left\vert\kern-0.2ex\left\vert \textbf{u} - \textbf{u}^h \right\vert\kern-0.2ex\right\vert}_{H^1 \left(\Omega\right)}$','FontSize',16,'Interpreter','Latex');
set(gca,'FontSize',14,'FontName','Times New Roman')
le = legend({'IGA','Regular','Optimal'},'Location','northwest');
set(le,'FontSize',14,'Interpreter','Latex');
set(gcf,'Position',[350 150 700 500]);
saveas(gcf,'Convergence - Lid-Driven Cavity','png')
hold off;

end

