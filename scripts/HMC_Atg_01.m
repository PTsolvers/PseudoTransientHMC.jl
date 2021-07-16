function [ ] = HMC_PT_DIV();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSEUDO-TRANSIENT FINITE DIFFERENCE CODE FOR HYDRO-MECHANICAL-CHEMICAL SIMULATION
% Stefan Schmalholz, June 2021
% Description of model and parameters in corresponding manuscript of
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all, close all, clc
ires            = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPORT DATA for DENSITIES AND MASS FRACTION for reaction atg + bru = fo + H2O
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load LOOK_UP_atg
RHO_s_LU        = Rho_s;        % Precalculated solid density as function of fluid pressure
Rho_f_LU        = Rho_f;        % Precalculated fluid density as function of fluid pressure
X_LU            = X_s_vec;      % Precalculated mass fraction as function of fluid pressure
P_LU            = P_vec*1e8;    % Corresponding fluid pressure array; converted to Pa
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NUMERICAL MODEL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% === INPUT INPUT INPUT ===================================================
% Independent parameters
radius          = 1;            % Radius of initial P-perturbation [m]
eta_mat         = 1;            % Viscosity scale [Pa s]
P_ini           = 1;            % Initial ambient pressure [Pa]
rho_0           = 3000;         % Density scale [kg/m^3]
% Nondimensional parameters
ellipse_factor  = 2.0;          % Use 1 for circle
angle           = 0;            % Counterclockwise angle of long axis with respect to vertical direction
phi_ini         = 4e-2;         % Initial porosity
phi_exp         = 30;           % Parameter controlling viscosity-porosity relation
% -------------------------------------------------------------------------
Lx_rad          = 40;           % LAMBDA_1 in equation (15) in the manuscript; Model height divided by inclusion radius;
Lc_rad2         = 1e1;          % LAMBDA_2 in equation (15) in the manuscript; Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
lam_eta         = 2;            % LAMBDA_3 in equation (15) in the manuscript; lam_eta = lambda / eta_mat; []; Ratio of bulk to shear viscosity
Da              = 0.0391;       % LAMBDA_4 in equation (15) in the manuscript; Da   = eb*eta_mat/P_ini; []; Ratio of viscous stress to initial stress
Ly_Lx           = 1;            % Model height divided by model width
Pini_Pappl      = P_ini/12.8e8; % Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
% Dependant parameters
ks              = 1e11*Pini_Pappl;          % Solid elastic bulk modulus
k_etaf          = Lc_rad2*radius^2/eta_mat; % Permeability divided by fluid viscosity; [m^2/(Pa s)]
lambda          = lam_eta*eta_mat;          % Bulk viscosity [Pa s]
eb              = Da*P_ini/eta_mat;         % Background strain rate in matrix [1/s]
Lx              = Lx_rad*radius;            % Model width [m]
Ly              = Ly_Lx*Lx;                 % Model height [m]
P_LU            = P_LU*Pini_Pappl;          % Transform look-up table stress to PT stress scale
kd              = ks/2;                     % Elastic bulk modulus, drained
alpha           = 1 - kd/ks;                % Biot-Wilis coefficient
% Characteristic time scales
Ch_ti_fluid_dif     = radius^2 /(k_etaf          *ks);
Ch_ti_fluid_dif_phi = radius^2 /(k_etaf*phi_ini^3*ks);
Ch_ti_relaxation    = eta_mat  /ks;
Ch_ti_kinetic       = 1.0 * 1/3*Ch_ti_fluid_dif_phi;
Ch_ti_deformation   = 1.0 * 1.0*Ch_ti_fluid_dif_phi;
eb                  = 1/Ch_ti_deformation;
Da                  = eb/(P_ini/eta_mat);   % Re-scaling of Da (LAMBDA_4)
% Numerical resolution
nx              = 301;                              % Numerical resolution width
ny              = nx;                               % Numerical resolution height
tol             = 1e-5;                             % Tolerance for pseudo-transient iterations
cfl             = 1/16.1;                           % CFL parameter for PT-Stokes solution
dtp             = Ch_ti_fluid_dif /2;               % Time step physical
time_tot        = 5e3*dtp;                          % Total time of simulation
kin_time        = 1e1*Ch_ti_fluid_dif_phi;
kin_time_final  = Ch_ti_kinetic;
Kin_time_vec    = [1e25*Ch_ti_fluid_dif_phi*ones(1,1) linspace(kin_time,kin_time_final,20) kin_time_final*ones(1,time_tot/dtp)];
% === INPUT INPUT INPUT ===================================================
% Configuration of grid, matrices and numerical parameters
nsave           = 50;                           % Interval of output time step
nrestart        = 100;                          % Interval of data saving for restart option
dx              = Lx/(nx-1);                    % Grid spacing
dy              = Ly/(ny-1);                    % Grid spacing
X1d             = [-Lx/2:dx:Lx/2];              % Coordinate vector
Y1d             = [-Ly/2:dy:Ly/2];              % Coordinate vector
[X2D,   Y2D  ]  = ndgrid(X1d,    Y1d);          % Mesh coordinates
Pf              = P_ini*ones(nx,  ny);          % Initial ambient fluid pressure
Ptot            = Pf;                           % Initial total pressure
rad_a           = radius;
rad_b           = rad_a*ellipse_factor;
X_ROT           =  X2D*cosd(angle)+Y2D*sind(angle);
Y_ROT           = -X2D*sind(angle)+Y2D*cosd(angle);
X_Ellipse       =  rad_a*cos([0:0.01:2*pi])*cosd(angle)+rad_b*sin([0:0.01:2*pi])*sind(angle);
Y_Ellipse       = -rad_a*cos([0:0.01:2*pi])*sind(angle)+rad_b*sin([0:0.01:2*pi])*cosd(angle);
ELLIPSE_XY      = [-X_Ellipse; Y_Ellipse];
Pf_inip         = Pf;
% ANALYTICAL FIT OF THERMODYNAMIC TABLE; Densities and mass fraction
% Parameters for solid density
p_min_ana           = min(P_LU);
p_max_ana           = max(P_LU);
rho_s_up            = 50;
slope_ana           = (Pf-p_min_ana)./p_max_ana*rho_s_up;
rho_s_max           = RHO_s_LU(1);
rho_s_min_ana       = min(RHO_s_LU);
rho_s_dif_ana       = rho_s_max-rho_s_min_ana;
p_reaction_ana      = 12.65*1e8*Pini_Pappl;
% Parameters for fluid density
rho_f_max_ana       = max(Rho_f_LU);
% Parameters for mass fraction
x_max               = max(X_LU);
x_min_ana           = min(X_LU);
x_dif_ana           = x_max-x_min_ana;
% Density, compressibility and gamma from concentration and pressure from thermodynamic data base
RHO_S           = (-tanh( 6e2*(Pf-p_reaction_ana) ) *(rho_s_dif_ana/2+rho_s_up/3) + (rho_s_dif_ana/2-rho_s_up/3) + rho_s_min_ana + slope_ana) /rho_0;
RHO_F           = (rho_f_max_ana*log(Pf+1).^(1/3.5)) /rho_0;
X_S             =  -tanh( 6e2*(Pf-p_reaction_ana) ) *x_dif_ana/2 + x_dif_ana/2 + x_min_ana;
% Initialize required matrices (or arrays)
PHI_INI         = phi_ini*ones(nx,  ny);
PHI_INI(sqrt(X_ROT.^2./rad_a^2+Y_ROT.^2./rad_b^2)<1) = 2.5*phi_ini;  % Fluid pressure perturbation
for smo=1:2 % Smooting of initial porosity
    Ii               = [2:nx-1];
    PHI_INI(Ii,:)    = PHI_INI(Ii,:) + 0.4*(PHI_INI(Ii+1,:)-2*PHI_INI(Ii,:)+PHI_INI(Ii-1,:));
    PHI_INI(:,Ii)    = PHI_INI(:,Ii) + 0.4*(PHI_INI(:,Ii+1)-2*PHI_INI(:,Ii)+PHI_INI(:,Ii-1));
end
PHI             = PHI_INI;
K_DARCY         = k_etaf*ones(nx,  ny);
DIVV            = zeros(nx,  ny);
DIVV_RHO_X      = zeros(nx,  ny);
DIVV_RHO_T      = zeros(nx,  ny);
TAUXX           = zeros(nx,  ny);               % Deviatoric stress
TAUYY           = zeros(nx,  ny);               % Deviatoric stress
TAUXY           = zeros(nx-1,ny-1);             % Deviatoric stress
% Centered coordinates needed for staggered grid for solid velocities
X1d_vx          = [X1d(1)-dx/2,( X1d(1:end-1) + X1d(2:end) )/2,X1d(end)+dx/2];  % Horizontal vector for Vx which is one more than basic grid
Y1d_vy          = [Y1d(1)-dy/2,( Y1d(1:end-1) + Y1d(2:end) )/2,Y1d(end)+dy/2];  % Verical    vector for Vy which is one more than basic grid
[X2D_vx,Y2D_vx] = ndgrid(X1d_vx, Y1d);
[X2D_vy,Y2D_vy] = ndgrid(X1d,    Y1d_vy);
VX              =  eb*X2D_vx;                   % Pure shear, shortening in x
VY              = -eb*Y2D_vy;                   % Pure shear, extension in y
% Parameters for time loop and pseudo-transient iterations
time            = 0;                            % Initial time
it              = 0;                            % Integer count for iteration loop
itp             = 0;                            % Integer count for time loop
save_count      = 0;
Time_vec        = [0];
% === TIME LOOP ===
while time < time_tot
    if ires==1
        load RES_HMC_atg_restart.mat; ires=0;
    end
    err_M                           = 1e10;     % Error at beginning of each PT interation
    itp                             = itp + 1
    time                            = time + dtp;
    Time_vec                        = [Time_vec time];
    RHO_S_old                       = RHO_S;
    RHO_F_old                       = RHO_F;
    X_S_old                         = X_S;
    RHO_X_old                       = RHO_S_old.*X_S_old;
    PHI_old                         = PHI;
    Ptot_old                        = Ptot;
    Pf_old                          = Pf;
    RHO_T_old                       = RHO_F_old.*PHI_old + RHO_S_old.*(1-PHI_old);
    
    ETA                             = eta_mat*exp(-phi_exp*(PHI-phi_ini));  % Shear viscosity
    LAMBDA                          = lam_eta*ETA;                          % Volumetric viscosity
    
    kin_time                        = Kin_time_vec(itp);
    
    % === PSEUDO-TIME ITERATION LOOP ===
    it_tstep                        = 0;
    itmax                           = 3e4;
    while err_M > tol & it_tstep<itmax
        it                          = it+1;
        it_tstep                    = it_tstep+1;
        dt_Stokes                   = cfl*max(dx,dy)^2/max(max(ETA));                       % Pseudo time step for Stokes
        dt_Pf                       = cfl*max(dx,dy)^2/max(max(K_DARCY.*PHI.^3 *(4*ks)));   % Pseudo time step for fluid pressure
        % === EVOLUTION OF FLUID PRESSURE ===
        Ix = [2:nx-1]; Iy = [2:ny-1];
        RHO_T                       = RHO_F.*PHI + RHO_S.*(1-PHI);
        para_cx                     = ( RHO_F(1:end-1,:).*K_DARCY(1:end-1,:).*PHI(1:end-1,:).^3. + RHO_F(2:end,:).*K_DARCY(2:end,:).*PHI(2:end,:).^3 )/2;
        para_cy                     = ( RHO_F(:,1:end-1).*K_DARCY(:,1:end-1).*PHI(:,1:end-1).^3. + RHO_F(:,2:end).*K_DARCY(:,2:end).*PHI(:,2:end).^3 )/2;
        q_f_X                       = -para_cx.*( diff(Pf,1,1)/dx );        % Correct   Darcy flux with fluid pressure
        q_f_Y                       = -para_cy.*( diff(Pf,1,2)/dy   );      % Correct   Darcy flux with fluid pressure
        q_f_X_Ptot                  = -para_cx.*( diff(Ptot,1,1)/dx );      % Incorrect Darcy flux with total pressure
        q_f_Y_Ptot                  = -para_cy.*( diff(Ptot,1,2)/dy );      % Incorrect Darcy flux with total pressure
        VX_f                        = swell2( q_f_X./(RHO_F(1:end-1,:)./PHI(1:end-1,:)+RHO_F(2:end,:)./PHI(2:end,:))*2 ,1);
        VY_f                        = swell2( q_f_Y./(RHO_F(:,1:end-1)./PHI(:,1:end-1)+RHO_F(:,2:end)./PHI(:,2:end))*2 ,2);
        VX_f_Ptot                   = swell2( q_f_X_Ptot./(RHO_F(1:end-1,:)./PHI(1:end-1,:)+RHO_F(2:end,:)./PHI(2:end,:))*2 ,1);
        VY_f_Ptot                   = swell2( q_f_Y_Ptot./(RHO_F(:,1:end-1)./PHI(:,1:end-1)+RHO_F(:,2:end)./PHI(:,2:end))*2 ,2);
        DIV_q_f                     = diff(q_f_X(:,2:end-1),1,1)/dx + diff(q_f_Y(2:end-1,:),1,2)/dy;
        RES_Pf                      = ( -DIV_q_f -(RHO_T(Ix,Iy)-RHO_T_old(Ix,Iy))/dtp - DIVV_RHO_T(Ix,Iy) ); % CONSERVATION OF TOTAL MASS EQUATION
        Pf(Ix,Iy)                   = Pf(Ix,Iy) + dt_Pf.*RES_Pf;
        Pf([1 end],:)               = Pf([2 end-1],:);
        Pf(:,[1 end])               = Pf(:,[2 end-1]);
        % === LOOK-UP OF DENSITIES ===
        RHO_F                       = (rho_f_max_ana*log(Pf+1).^(1/3.5)) /rho_0;
        RHO_S_EQ                    = (-tanh( 6e2*(Pf-p_reaction_ana) ) *(rho_s_dif_ana/2+rho_s_up/3) + (rho_s_dif_ana/2-rho_s_up/3) + rho_s_min_ana + slope_ana) /rho_0;
        X_S_EQ                      =  -tanh( 6e2*(Pf-p_reaction_ana) ) *x_dif_ana/2 + x_dif_ana/2 + x_min_ana;
        if itp>1
            X_S                         = X_S_old   + dtp.*(X_S_EQ   - X_S  )/kin_time;
            RHO_S                       = RHO_S_old + dtp.*(RHO_S_EQ - RHO_S)/kin_time;
        end
        % === POROSITY EVOLUTION ===
        RHO_X                       = RHO_S.*X_S;
        RES_PHI                     = ( (1-PHI).*RHO_X - (1-PHI_old).*RHO_X_old )/dtp + DIVV_RHO_X;     % CONSERVATION OF MASS OF MgO EQUATION
        PHI                         = PHI + dtp.*RES_PHI;
        % === VISCOSITIES ===
        ETA                         = eta_mat*exp(-phi_exp*(PHI-phi_ini));
        LAMBDA                      = lam_eta*ETA;
        % === STRAIN RATES ===
        DIVV_RHO_X                  = (diff( swell2((1-PHI).*RHO_X,1).*VX,1,1)/dx + diff( swell2((1-PHI).*RHO_X,2).*VY,1,2)/dy);
        DIVV_RHO_T                  = (diff( swell2(RHO_T,1).*VX,1,1)/dx + diff( swell2(RHO_T,2).*VY,1,2)/dy);
        DIVV                        = (diff(VX,1,1)/dx + diff(VY,1,2)/dy);      % Divergence of velocity
        DXX                         = diff(VX,1,1)/dx -1/3*DIVV;                % Horizontal deviatoric strain rate
        DYY                         = diff(VY,1,2)/dy -1/3*DIVV;                % Vertical deviatoric strain rate
        DXY                         = 1/2*(diff(VX(2:end-1,:),1,2)/dy + diff(VY(:,2:end-1),1,1)/dx); % Shear strain rate; two entries smaller
        % === TOTAL PRESSURE
        dPf_dt                      = (Pf - Pf_old)/dtp;
        div_bulk                    = 1 + dtp*kd./((1-PHI).*LAMBDA);
        para_bulk                   = kd*DIVV - alpha*dPf_dt - kd*Pf./((1-PHI).*LAMBDA);
        Ptot                        = (Ptot_old-dtp*(para_bulk))./div_bulk;           % Rheology for total pressure
        % === STRESSES & VISCOSITIES ===
        TAUXX                       = 2.*ETA.*DXX;                              % Horizontal deviatoric stress
        TAUYY                       = 2.*ETA.*DYY;                              % Vertical deviatoric stress
        ETAXY                       = (ETA(2:end,:)   + ETA(1:end-1,:))/2;      % Averaged viscosity for shear stress
        ETAXY                       = (ETAXY(:,2:end) + ETAXY(:,1:end-1))/2;    % Averaged viscosity for shear stress
        TAUXY                       = 2.*ETAXY.*DXY;                            % Shear stress
        TII                         = sqrt( 1/4.*(TAUXX-TAUYY).^2 + swell2(swell2(TAUXY,1),2).^2);
        % === FORCE BALANCE EQUATIONS ===
        RES_VX                      = -diff(Ptot(:,2:end-1),1,1)/dx + diff(TAUXX(:,2:end-1),1,1)/dx + diff(TAUXY,1,2)/dy; % HORIZONTAL FORCE BALANCE
        RES_VY                      = -diff(Ptot(2:end-1,:),1,2)/dy + diff(TAUYY(2:end-1,:),1,2)/dy + diff(TAUXY,1,1)/dx; % VERTICAL   FORCE BALANCE
        VX(2:end-1,2:end-1)         = VX(2:end-1,2:end-1) + dt_Stokes*RES_VX; % Pseudo-transient form of horizontal force balance
        VY(2:end-1,2:end-1)         = VY(2:end-1,2:end-1) + dt_Stokes*RES_VY; % Pseudo-transient form of vertical force balance
        %==================================================================
        if it_tstep>250
            err_Mx                      = max(abs(dt_Stokes*RES_VX(:)))/max(abs(VX(:)));    % Error horizontal velocitiy
            err_My                      = max(abs(dt_Stokes*RES_VY(:)))/max(abs(VY(:)));    % Error vertical velocity
            err_Pf                      = max(abs(dt_Pf*RES_Pf(:)))/max(abs(Pf(:)));        % Error fluid pressure
            err_Phi                     = max(abs(dtp*RES_PHI(:)));                         % Error porosity
            err_M                       = max([err_Pf, err_Mx, err_My, err_Phi]);           % Error total
            % === PLOT ERROR EVOLUTION ===
            if mod(it_tstep,1e4)==1 | it_tstep==251
                figure(1)
                plot(it,log10(err_Mx),'ko'),hold on;plot(it,log10(err_My),'rd');plot(it,log10(err_Pf),'ks');plot(it,log10(err_Phi),'b+');
                plot([0 it+200],[log10(tol) log10(tol)],'-r','linewidth',2)
                legend('Error Vx','Error Vy','Error Pf','Error \phi','Tolerance','Location','Northwest'); grid on;xlabel('Total iterations');ylabel('log_{10} Error')
                drawnow
            end
        end
    end % === END PSEUDO-TIME ITERATION LOOP ===
    
    % =========================================================================
    % === VISUALIZATION =======================================================
    % =========================================================================
    if mod(itp,2e1)==1 | itp==1
        figure(2)
        ax  = Lx_rad/2*[-1 1 -1 1];
        sp  = 5;
        colormap(jet)
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
        pcolor(X2D,Y2D,DIVV*Time_model_sec/Time_phys_sec);
        col=colorbar;  shading interp; hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['C) \nabla(v_s) [1/s]'])
        axis equal; axis(ax)
        
        subplot('position',[0.1 0.4 0.25 0.25])
        pcolor(X2D,Y2D,X_S);
        col=colorbar;  shading interp; hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['D) X_s'])
        ylabel('Height / radius [ ]')
        axis equal; axis(ax)
        
        subplot('position',[0.36 0.4 0.25 0.25])
        VX_C    = (VX(1:end-1,:)+VX(2:end,:))/2;
        VY_C    = (VY(:,1:end-1)+VY(:,2:end))/2;
        % pcolor(X2D,Y2D,sqrt(VX_C.^2+VY_C.^2)*Vel_phys_m_s);
        pcolor(X2D,Y2D,RHO_S*rho_0);
        col=colorbar;  shading interp; hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['E) \rho_s [kg/m^3]'])
        %         title(['E) [ (v_x^s)^2 + (v_y^s)^2 ]^{1/2} [m/s]'])
        axis equal; axis(ax)
        
        subplot('position',[0.62 0.4 0.25 0.25])
        pcolor(X2D,Y2D,PHI);
        col=colorbar;  shading interp;hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['F) \phi'])
        axis equal; axis(ax)
        
        X2D_C       = ( X2D(  1:end-1,:) + X2D(  2:end,:) )/2;
        X2D_C       = ( X2D_C(:,1:end-1) + X2D_C(:,2:end) )/2;
        Y2D_C       = ( Y2D(  1:end-1,:) + Y2D(  2:end,:) )/2;
        Y2D_C       = ( Y2D_C(:,1:end-1) + Y2D_C(:,2:end) )/2;
        
        subplot('position',[0.1 0.08 0.25 0.25])
        pcolor(X2D,Y2D,RHO_F*rho_0);
        col=colorbar;  shading interp; hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['G) \rho_f [kg/m^3]'])
        axis equal; axis(ax)
        ylabel('Height / radius [ ]')
        xlabel('Width / radius [ ]');
        
        subplot('position',[0.36 0.08 0.25 0.25])
        pcolor(X2D,Y2D,TII/Pini_Pappl/1e6);
        col=colorbar;  shading interp; hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        axis equal
        axis(ax)
        title(['H) \tau_{II} [MPa]'])
        xlabel('Width / radius [ ]'); %ylabel('Height / radius [ ]')
        
        subplot('position',[0.62 0.08 0.25 0.25])
        caxis([-0.1 0.05])
        pcolor(X2D,Y2D,log10(ETA*visc_char));
        col=colorbar;  shading interp;hold on
        plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
        title(['I) log_{10} \eta^s [Pas]'])
        axis equal; axis(ax)
        xlabel('Width / radius [ ]'); %ylabel('Height / radius [ ]')
        
        set(gcf,'position',[524.2000 79.8000 986.4000 698.4000])
        drawnow
    end
    
    % === SAVING RESULTS ===
    if  itp==1 | mod(itp-1,nsave)==0
        save_count      = save_count+1;
        string_number   = num2str(save_count,['%',num2str(5),'.',num2str(5),'d']);
        file_name       = {['RES_HMC_atg_', string_number,'.mat']};
        save(char(file_name))
    end
    % === SAVING DATA FOR RESTART OPTION ===
    if mod(itp-1,nrestart)==0
        save('RES_HMC_atg_restart.mat')
    end
    
end % === END TIME LOOP ===

% =========================================================================
% === ADDITIONAL FUNCTION
% =========================================================================
function [B] = swell2(A,ndim);
% Logic: assume 3 points x1 x2 x3 and that the "extended" array is xe
% we have 2 equations (assuming linear extrapolations)
% 1: xe1 + xe2 = 2*x1
% 2: xe2       = (x1+x2)/2
%Substitute the second into the first and solve for xe1 gives
% xe1 + x1/2 + x2/2 = (4/2)*x1 -> xe1 = 3/2*x1 -1/2*x2
if ndim == 1
    B = zeros(size(A,1)+1,size(A,2));
    %     B(2:end-1,:) = ava2(A,ndim);
    B(2:end-1,:)   = (A(2:end,:)   + A(1:end-1,:))/2;
    B(1,  :)       = 1.5*A(1,:)   -0.5*A(2,:);
    B(end,:)       = 1.5*A(end,:) -0.5*A(end-1,:);
elseif ndim ==2
    B = zeros(size(A,1),size(A,2)+1);
    %     B(:,2:end-1)   = ava2(A,ndim);
    B(:,2:end-1)   = (A(:,2:end) + A(:,1:end-1))/2;
    B(:,1  )       = 1.5*A(:,1)   -0.5*A(:,2);
    B(:,end)       = 1.5*A(:,end) -0.5*A(:,end-1);
end
return;