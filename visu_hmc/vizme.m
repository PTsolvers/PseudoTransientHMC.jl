clear
close all

radius          = 1;            % Radius of initial P-perturbation [m]
eta_mat         = 1;            % Viscosity scale [Pa s]
P_ini           = 1;            % Initial ambient pressure [Pa]
% Nondimensional parameters
ellipse_factor  = 2.0;          % Use 1 for circle
angle           = 0;            % Counterclockwise angle of long axis with respect to vertical direction
% -------------------------------------------------------------------------
Lx_rad          = 40;           % LAMBDA_1 in equation (15) in the manuscript; Model height divided by inclusion radius;
Lc_rad2         = 1e1;          % LAMBDA_2 in equation (15) in the manuscript; Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
Ly_Lx           = 1;            % Model height divided by model width
Pini_Pappl      = P_ini/12.8e8; % Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
% Dependant parameters
ks              = 1e11*Pini_Pappl;          % Solid elastic bulk modulus
k_etaf          = Lc_rad2*radius^2/eta_mat; % Permeability divided by fluid viscosity; [m^2/(Pa s)]
Lx              = Lx_rad*radius;            % Model width [m]
Ly              = Ly_Lx*Lx;                 % Model height [m]
% Numerical resolution
nx              = 1023;                              % Numerical resolution width
ny              = nx;                               % Numerical resolution height
dx              = Lx/(nx-1);                    % Grid spacing
dy              = Ly/(ny-1);                    % Grid spacing
X1d             = [-Lx/2:dx:Lx/2];              % Coordinate vector
Y1d             = [-Ly/2:dy:Ly/2];              % Coordinate vector
[X2D,   Y2D  ]  = ndgrid(X1d,    Y1d);          % Mesh coordinates
rad_a           = radius;
rad_b           = rad_a*ellipse_factor;
X_Ellipse       =  rad_a*cos([0:0.01:2*pi])*cosd(angle)+rad_b*sin([0:0.01:2*pi])*sind(angle);
Y_Ellipse       = -rad_a*cos([0:0.01:2*pi])*sind(angle)+rad_b*sin([0:0.01:2*pi])*cosd(angle);
ELLIPSE_XY      = [-X_Ellipse; Y_Ellipse];

nsave = 1

for isave = 1:nsave
    
    itp = 1+50*(isave-1);
    load(['pt_hmc_Atg_' int2str(itp)])
    
    figure(2)
    ax  = Lx_rad/2*[-1 1 -1 1];
    sp  = 5;
    % colormap(jet)
    % Parameters for scaling results for specific values of parameters
    visc_char           = 1e19;
    Length_model_m      = radius;
    Time_model_sec      = radius^2/(k_etaf*ks);
    Length_phys_meter   = 0.01;
    Time_phys_sec       = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8));
    Vel_phys_m_s        = (Length_phys_meter/Length_model_m) / (Time_phys_sec/Time_model_sec);
    
    subplot('position',[0.1 0.72 0.25 0.25])
    pcolor(X2D,Y2D,Pf/Pini_Pappl/1e8);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--k','linewidth',1)
    contour(X2D,Y2D,Pf/Pini_Pappl/1e8,[12.65 12.65],'-w');
    title(['A) p_f [kbar]; time step: ',num2str(itp)])
    axis equal; axis(ax)
    ylabel('Height / radius [ ]')
    
    subplot('position',[0.36 0.72 0.25 0.25])
    pcolor(X2D,Y2D,Ptot/Pini_Pappl/1e8);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    title(['B) p [kbar]'])
    axis equal; axis(ax)
    
    subplot('position',[0.62 0.72 0.25 0.25])
    pcolor(X2D,Y2D,divV*Time_model_sec/Time_phys_sec);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    title(['C) \nabla(v_s) [1/s]'])
    axis equal; axis(ax)
    
    subplot('position',[0.1 0.4 0.25 0.25])
    pcolor(X2D,Y2D,Velf*Vel_phys_m_s);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    title(['D) vel_f'])
    ylabel('Height / radius [ ]')
    axis equal; axis(ax)
    
    subplot('position',[0.36 0.4 0.25 0.25])
    % VX_C    = (VX(1:end-1,:)+VX(2:end,:))/2;
    % VY_C    = (VY(:,1:end-1)+VY(:,2:end))/2;
    % pcolor(X2D,Y2D,sqrt(VX_C.^2+VY_C.^2)*Vel_phys_m_s);
    pcolor(X2D,Y2D,Vel*Vel_phys_m_s);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    title(['E) Vel'])
    %         title(['E) [ (v_x^s)^2 + (v_y^s)^2 ]^{1/2} [m/s]'])
    axis equal; axis(ax)
    
    % subplot('position',[0.62 0.4 0.25 0.25])
    % pcolor(X2D,Y2D,PHI);
    % col=colorbar;  shading interp;hold on
    % plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    % title(['F) \phi'])
    % axis equal; axis(ax)
    
    % X2D_C       = ( X2D(  1:end-1,:) + X2D(  2:end,:) )/2;
    % X2D_C       = ( X2D_C(:,1:end-1) + X2D_C(:,2:end) )/2;
    % Y2D_C       = ( Y2D(  1:end-1,:) + Y2D(  2:end,:) )/2;
    % Y2D_C       = ( Y2D_C(:,1:end-1) + Y2D_C(:,2:end) )/2;
    
    % subplot('position',[0.1 0.08 0.25 0.25])
    % pcolor(X2D,Y2D,RHO_F*rho_0);
    % col=colorbar;  shading interp; hold on
    % plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    % title(['G) \rho_f [kg/m^3]'])
    % axis equal; axis(ax)
    % ylabel('Height / radius [ ]')
    % xlabel('Width / radius [ ]');
    
    subplot('position',[0.36 0.08 0.25 0.25])
    pcolor(X2D,Y2D,TauII/Pini_Pappl/1e6);
    col=colorbar;  shading interp; hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    axis equal
    axis(ax)
    title(['H) \tau_{II} [MPa]'])
    xlabel('Width / radius [ ]'); %ylabel('Height / radius [ ]')
    
    subplot('position',[0.62 0.08 0.25 0.25])
    caxis([-0.1 0.05])
    pcolor(X2D,Y2D,log10(Eta*visc_char));
    col=colorbar;  shading interp;hold on
    plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
    title(['I) log_{10} \eta^s [Pas]'])
    axis equal; axis(ax)
    xlabel('Width / radius [ ]'); %ylabel('Height / radius [ ]')
    
    set(gcf,'position',[524.2000 79.8000 986.4000 698.4000])
    drawnow
    
end