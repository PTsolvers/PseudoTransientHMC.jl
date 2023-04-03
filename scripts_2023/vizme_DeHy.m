function [ ] = VISU_Julia();
clear variables, close all, clc
% Colormap lapaz from https://www.fabiocrameri.ch/colourmaps/
load lapaz.mat; colormap(flipud(lapaz))
axx = [-1 1 -1 1]*7;
% Change to output directory
cd ../output_dehy_500x500
FileNum     = [1 800 2000 4000]; % Four result files to be loaded
pcount      = 0;
for ilo=1:length(FileNum)
    pcount          = pcount+1;
    string_number   = num2str(FileNum(ilo),['%',num2str(4),'.',num2str(4),'d']);
    file_name       = {['DEHY_', string_number,'.mat']};
    load(char(file_name))
    % Scales
    rho_0           = rho_0;         % Density scale [kg/m^3]
    Pini_Pappl      = Pini_Pappl;    % Stress scale  [1/Pa]
    t_char          = Ch_ti_fluid_dif_phi;              % Characteristc time 
    [X2Dc,Y2Dc  ]   = ndgrid(xc,    yc);                % Mesh coordinates
    nx              = size(X2Dc,1);
    step            = 16;
    Ii              = [1:step:nx];
    VX_C            = (Vx(1:end-1,:)+Vx(2:end,:))/2;    % Solid velocities X
    VY_C            = (Vy(:,1:end-1)+Vy(:,2:end))/2;    % Solid velocities Y
    %%% DENSITY %%%
    cax = [2550 3000];                                  % Color axis
    if pcount==1
        subplot('position',[0.075        0.7 0.2 0.25])
    elseif pcount==2
        subplot('position',[0.075+0.21   0.7 0.2 0.25])
    elseif pcount==3
        subplot('position',[0.075+0.21*2 0.7 0.2 0.25])
    elseif pcount==4
        subplot('position',[0.075+0.21*3 0.7 0.2 0.25])
    end
    hold on
    [c,h] = contour(X2Dc,Y2Dc,Pf/Pini_Pappl/1e8,[12.7 12.7],'-k','linewidth',0.5);
    [cl,hl]=contour(X2Dc,Y2Dc,Phi,[0.15 0.15],'-r');
    pcolor(X2Dc,Y2Dc,Rho_s*rho_0), shading interp
    caxis(cax)
    [c,h] = contour(X2Dc,Y2Dc,Pf/Pini_Pappl/1e8,[12.7 12.7],'-k','linewidth',0.5);
    [cl,hl]=contour(X2Dc,Y2Dc,Phi,[0.15 0.15],'-r');
    quiver(X2Dc(Ii,Ii),Y2Dc(Ii,Ii),(VX_C(Ii,Ii)),(VY_C(Ii,Ii)),1.5,'color',[1 1 1]*0.3);
    axis equal
    axis(axx)
    if pcount==1
        le = legend('12.7 kbar','0.15');
        set(le,'fontsize',8,'orientation','horizontal')
        set(le,'Position',[0.125 0.9225 0.1 0.015])
        ylabel('y / r')
        title('A) Solid density')
        text(-6.5,-6.5,['Time: ',num2str(timeP/t_char,4)])
    elseif pcount==4
        col=colorbar;
        set(col,'position',[0.91 0.7 0.03 0.25])
        text(7,8.25,'[kg/m^3]')
        title('D) Solid density')
        text(-6.5,-6.5,['Time: ',num2str(timeP/t_char,4)])
    end
    if pcount==2
        title('B) Solid density')
        text(-6.5,-6.5,['Time: ',num2str(timeP/t_char,4)])
    elseif pcount==3
        title('C) Solid density')
        text(-6.5,-6.5,['Time: ',num2str(timeP/t_char,4)])
    end
    if pcount>1
        set(gca,'yticklabel',[])
    end
    set(gca,'box','on','FontSize',10)
    xlabel('x / r')
end
set(gcf,'position',[186.6000 72.2000 892.8000 690.8000])
cd ..
