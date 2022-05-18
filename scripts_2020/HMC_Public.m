function [ ] = HMC_PT_DIV();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PSEUDO-TRANSIENT FINITE DIFFERENCE CODE FOR HYDRO-MECHANICAL-CHEMICAL SIMULATION OF BRUCITE-PERICLASE REACTIONS
% Stefan Schmalholz, May 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all, close all, clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMPORT DATA for DENSITIES AND MASS FRACTION OF MgO IN SOLID for T = 800 C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load LOOK_UP_HMC_Pub
RHO_s_LU        = Rho_s_07;     % Precalculated solid density as function of fluid pressure
Rho_f_LU        = Rho_f;        % Precalculated fluid density as function of fluid pressure
X_LU            = X_s_vec;      % Precalculated mass fraction of MgO as function of fluid pressure
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
ellipse_factor  = 3.0;          % Use 1 for circle
angle           = 30;           % Counterclockwise angle of long axis with respect to vertical direction
phi_ini         = 2e-3;         % Initial porosity
eta_incl_fac    = 1e-3;         % Factor, how much solid SHEAR viscosity of inclusion is larger (factor>1) or smaller than surrounding
lambda_incl_fac = 1;            % Factor, how much solid BULK viscosity of inclusion is larger (factor>1) or smaller than surrounding
n_exp           = 3;            % Stress exponent of matrix; n=1 means linear viscous
% -------------------------------------------------------------------------
lam_eta         = 1e0;          % lam_eta = lambda / eta_mat; []; Ratio of bulk to shear viscosity
Lc_rad2         = 1e8;          % Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
Da              = 0.0024;       % Da   = eb*eta_mat/P_ini; []; Ratio of viscous stress to initial stress
sig_yield       = 0.024;        % Stress_ref / P_ini; []; Reference stress used for power-law viscous flow law
Lx_rad          = 10;           % Model width divided by inclusion radius
Ly_Lx           = 1;            % Model height divided by model width
Pini_Pappl      = P_ini/8.5e8;  % Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
% Dependant parameters
beta_eff        = 1e-2/P_ini;               % Effective compressibility used only to determine PT time step [1/Pa]
k_etaf          = Lc_rad2*radius^2/eta_mat; % Permeability divided by fluid viscosity; [m^2/(Pa s)]
P_pert          = 0.2*P_ini;                % Pressure perturbation [Pa]
lambda          = lam_eta*eta_mat;          % Bulk viscosity [Pa s]
eb              = Da*P_ini/eta_mat;         % Background strain rate in matrix [1/s]
stress_ref      = sig_yield *P_ini;         % Stress reference for power-law viscosity
Lx              = Lx_rad*radius;            % Model width [m]
Ly              = Ly_Lx*Lx;                 % Model height [m]
P_LU            = P_LU*Pini_Pappl;          % Transform look-up table stress to PT stress scale
% Numerical resolution
nx              = 101;                              % Numerical resolution width
ny              = nx;                               % Numerical resolution height
tol             = 10^(-5);                          % Tolerance for pseudo-transient iterations
cfl             = 1/16.1;                           % CFL parameter for PT-Stokes solution
dtp             = 2e0*radius^2/(k_etaf/beta_eff);   % Time step physical
time_tot        = 1*dtp;                            % Total time of simulation
% === INPUT INPUT INPUT ===================================================
% Configuration of grid, matrices and numerical parameters
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
Pf(sqrt(X_ROT.^2./rad_a^2+Y_ROT.^2./rad_b^2)<1) = P_ini-P_pert;  % Fluid pressure petubation
X_Ellipse       =  rad_a*cos([0:0.01:2*pi])*cosd(angle)+rad_b*sin([0:0.01:2*pi])*sind(angle);
Y_Ellipse       = -rad_a*cos([0:0.01:2*pi])*sind(angle)+rad_b*sin([0:0.01:2*pi])*cosd(angle);
ELLIPSE_XY      = [-X_Ellipse; Y_Ellipse];
for smo=1:3 % Smooting of perturbation
    Ii               = [2:nx-1];
    Pf(Ii,:)         = Pf(Ii,:) + 0.4*(Pf(Ii+1,:)-2*Pf(Ii,:)+Pf(Ii-1,:));
    Pf(:,Ii)         = Pf(:,Ii) + 0.4*(Pf(:,Ii+1)-2*Pf(:,Ii)+Pf(:,Ii-1));
end
Pf_inip         = Pf;
% Density, compressibility and gamma from concentration and pressure from thermodynamic data base
RHO_S           = interp1(P_LU(:,1),RHO_s_LU   ,Pf, 'linear') /rho_0;
RHO_F           = interp1(P_LU(:,1),Rho_f_LU   ,Pf, 'linear') /rho_0;
X_S             = interp1(P_LU(:,1),X_LU       ,Pf, 'linear') ;
RHO_S_ini       = interp1(P_LU(:,1),RHO_s_LU   ,P_ini*ones(nx,  ny), 'linear') /rho_0;
X_S_ini         = interp1(P_LU(:,1),X_LU       ,P_ini*ones(nx,  ny), 'linear') ;
% Initialize required matrices (or arrays)
PHI_INI         = phi_ini*ones(nx,  ny);
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
VX              = -eb*X2D_vx;                   % Pure shear, shortening in x
VY              =  eb*Y2D_vy;                   % Pure shear, extension in y
% Parameters for time loop and pseudo-transient iterations
time            = 0;                            % Initial time
it              = 0;                            % Integer count for iteration loop
itp             = 0;                            % Integer count for time loop
save_count      = 0;
Time_vec        = [0];
% === TIME LOOP ===
while time < time_tot
    err_M                           = 1e10;     % Error at beginning of each PT interation
    itp                             = itp + 1;
    time                            = time + dtp;
    Time_vec                        = [Time_vec time];
    RHO_S_old                       = RHO_S;
    RHO_F_old                       = RHO_F;
    X_S_old                         = X_S;
    RHO_X_old                       = RHO_S_old.*X_S_old;
    PHI_old                         = PHI;
    Ptot_old                        = Ptot;
    if itp==1
        RHO_X_INI                   = RHO_S_ini.*X_S_ini;
        PHI_old                     = 1 - (RHO_X_INI).*(1-phi_ini)./RHO_X_old;
        PHI_ini                     = PHI_old;
        PHI                         = PHI_old;
    end
    RHO_T_old                       = RHO_F_old.*PHI_old + RHO_S_old.*(1-PHI_old);
    Ind_incl                        = find(PHI>(max(PHI(:))+min(PHI(:)))/2);
    ETA_MAT                         = eta_mat*ones(nx, ny);         % Ambient viscosity
    ETA_MAT(Ind_incl)               = eta_mat*eta_incl_fac;         % Inclusion viscosity
    ETA_PL                          = ETA_MAT;
    ETA                             = ETA_MAT;
    LAMBDA                          = lambda*ones(nx, ny);         % Viscosity
    LAMBDA(Ind_incl)                = lambda*lambda_incl_fac;  % Fluid pressure petubation
    % === PSEUDO-TIME LOOP ===
    it_tstep                        = 0;
    itmax                           = 3e4;
    while err_M > tol & it_tstep<itmax
        it                          = it+1;
        it_tstep                    = it_tstep+1;
        dt_Stokes                   = cfl*max(dx,dy)^2/max(max(ETA));                       % Pseudo time step for Stokes
        dt_Pf                       = cfl*max(dx,dy)^2/max(max(K_DARCY.*PHI.^3/beta_eff));  % Pseudo time step for fluid pressure
        %==================================================================
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
        % === LOOK-UP OF DENSITIES ===
        RHO_S                       = interp1(P_LU(:,1),RHO_s_LU   ,Pf, 'linear') /rho_0;
        RHO_F                       = interp1(P_LU(:,1),Rho_f_LU,   Pf, 'linear') /rho_0;
        X_S                         = interp1(P_LU(:,1),X_LU       ,Pf, 'linear') ;
        % === POROSITY EVOLUTION ===
        RHO_X                       = RHO_S.*X_S;
        RES_PHI                     = ( (1-PHI).*RHO_X - (1-PHI_old).*RHO_X_old )/dtp + DIVV_RHO_X;     % CONSERVATION OF MASS OF MgO EQUATION
        PHI                         = PHI + dtp.*RES_PHI;
        %==================================================================
        %SSSS   STOKES   SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        % === STRAIN RATES ===
        DIVV_RHO_X                  = (diff( swell2((1-PHI).*RHO_X,1).*VX,1,1)/dx + diff( swell2((1-PHI).*RHO_X,2).*VY,1,2)/dy);
        DIVV_RHO_T                  = (diff( swell2(RHO_T,1).*VX,1,1)/dx + diff( swell2(RHO_T,2).*VY,1,2)/dy);
        DIVV                        = (diff(VX,1,1)/dx + diff(VY,1,2)/dy);        % Divergence of velocity
        DXX                         = diff(VX,1,1)/dx -1/3*DIVV;                % Horizontal deviatoric strain rate
        DYY                         = diff(VY,1,2)/dy -1/3*DIVV;                % Vertical deviatoric strain rate
        DXY                         = 1/2*(diff(VX(2:end-1,:),1,2)/dy + diff(VY(:,2:end-1),1,1)/dx); % Shear strain rate; two smaller
        % === TOTAL PRESSURE, STRESSES & VISCOSITIES ===
        Ptot                        = Pf - DIVV .* ((1-PHI).*LAMBDA);           % Rheology for total pressure
        TAUXX                       = 2.*ETA.*DXX;                              % Horizontal deviatoric stress
        TAUYY                       = 2.*ETA.*DYY;                              % Vertical deviatoric stress
        ETAXY                       = (ETA(2:end,:)   + ETA(1:end-1,:))/2;      % Averaged viscosity for shear stress
        ETAXY                       = (ETAXY(:,2:end) + ETAXY(:,1:end-1))/2;    % Averaged viscosity for shear stress
        TAUXY                       = 2.*ETAXY.*DXY;                            % Shear stress
        TII                         = sqrt( 1/4.*(TAUXX-TAUYY).^2 + swell2(swell2(TAUXY,1),2).^2);
        % === POWER LAW VISCOSITY ===
        if n_exp>1
            ETA_PL_old_ptit             = ETA_PL;                                   % Previous PT viscosity
            ETA_PL                      = ETA_MAT.*(TII/stress_ref).^(1-n_exp);
            relax                       = 0.5;
            ETA_PL(TII<stress_ref)      = ETA_MAT(TII<stress_ref); %eta_mat;
            ETA_PL                      = exp(log(ETA_PL)*relax+log(ETA_PL_old_ptit)*(1-relax));
            ETA                         = 2./( 1./ETA_MAT + 1./ETA_PL );
        end
        % === FORCE BALANCE EQUATIONS ===
        RES_VX                      = -diff(Ptot(:,2:end-1),1,1)/dx + diff(TAUXX(:,2:end-1),1,1)/dx + diff(TAUXY,1,2)/dy; % HORIZONTAL FORCE BALANCE
        RES_VY                      = -diff(Ptot(2:end-1,:),1,2)/dy + diff(TAUYY(2:end-1,:),1,2)/dy + diff(TAUXY,1,1)/dx; % VERTICAL   FORCE BALANCE
        VX(2:end-1,2:end-1)         = VX(2:end-1,2:end-1) + dt_Stokes*RES_VX; % Pseudo-transient form of horizontal force balance
        VY(2:end-1,2:end-1)         = VY(2:end-1,2:end-1) + dt_Stokes*RES_VY; % Pseudo-transient form of vertical force balance
        %SSSS   STOKES   SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS
        %==================================================================
        if it_tstep>250
            err_Mx                      = max(abs(dt_Stokes*RES_VX(:)))/max(abs(VX(:)));    % Error horizontal velocitiy
            err_My                      = max(abs(dt_Stokes*RES_VY(:)))/max(abs(VY(:)));    % Error vertical velocity
            err_Pf                      = max(abs(dt_Pf*RES_Pf(:)))/max(abs(Pf(:)));        % Error fluid pressure
            err_Phi                     = max(abs(dtp*RES_PHI(:)));                         % Error porosity
            err_M                       = max([err_Pf, err_Mx, err_My, err_Phi]);           % Error total
            % === PLOT ERROR EVOLUTION ===
            if mod(it_tstep,2e3)==1 | it_tstep==251
                plot(it,log10(err_Mx),'ko'),hold on;plot(it,log10(err_My),'rd');plot(it,log10(err_Pf),'ks');plot(it,log10(err_Phi),'b+');
                plot([0 it+200],[log10(tol) log10(tol)],'-r','linewidth',2)
                legend('Error Vx','Error Vy','Error Pf','Error \phi','Tolerance','Location','Northwest'); grid on;xlabel('Total iterations');ylabel('log_{10} Error')
                drawnow
            end
        end
        % === END PSEUDO-TIME LOOP ===
    end
    
end

% =========================================================================
% === VISUALIZATION =======================================================
% =========================================================================
figure(2)
ax  = [-5 5 -5 5];
sp  = 5;
colormap(jet)
% Parameters for scaling results for specific values of parameters
Length_model_m      = radius;
Time_model_sec      = radius^2/(k_etaf/beta_eff);
Length_phys_meter   = 0.01;
Time_phys_sec       = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8));
Vel_phys_m_s        = (Length_phys_meter/Length_model_m) / (Time_phys_sec/Time_model_sec);

subplot('position',[0.1 0.72 0.25 0.25])
pcolor(X2D,Y2D,Pf/Pini_Pappl/1e8); 
col=colorbar;  shading interp; hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--k','linewidth',1)
title('A) p_f [kbar]')
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

Length_model_m      = radius;
Time_model_sec      = radius^2/(k_etaf/beta_eff);
Length_phys_meter   = 0.01;
Time_phys_sec       = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8));
Vel_phys_m_s        = (Length_phys_meter/Length_model_m) / (Time_phys_sec/Time_model_sec);

subplot('position',[0.1 0.4 0.25 0.25])
pcolor(X2D,Y2D,sqrt(VX_f.^2+VY_f.^2)*Vel_phys_m_s);
col=colorbar;  shading interp; hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
title(['D) [ (v_x^f)^2 + (v_y^f)^2 ]^{1/2} [m/s]'])
ylabel('Height / radius [ ]')
axis equal; axis(ax)

subplot('position',[0.36 0.4 0.25 0.25])
VX_C    = (VX(1:end-1,:)+VX(2:end,:))/2;
VY_C    = (VY(:,1:end-1)+VY(:,2:end))/2;
pcolor(X2D,Y2D,sqrt(VX_C.^2+VY_C.^2)*Vel_phys_m_s);
col=colorbar;  shading interp; hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
title(['E) [ (v_x^s)^2 + (v_y^s)^2 ]^{1/2} [m/s]'])
axis equal; axis(ax)

subplot('position',[0.62 0.4 0.25 0.25])
pcolor(X2D,Y2D,sqrt(VX_f_Ptot.^2+VY_f_Ptot.^2)*Vel_phys_m_s);
col=colorbar;  shading interp;hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
title(['F) [ (v_x^f^p)^2 + (v_y^f^p)^2 ]^{1/2} [m/s]'])
axis equal; axis(ax)

X2D_C       = ( X2D(  1:end-1,:) + X2D(  2:end,:) )/2;
X2D_C       = ( X2D_C(:,1:end-1) + X2D_C(:,2:end) )/2;
Y2D_C       = ( Y2D(  1:end-1,:) + Y2D(  2:end,:) )/2;
Y2D_C       = ( Y2D_C(:,1:end-1) + Y2D_C(:,2:end) )/2;

subplot('position',[0.1 0.08 0.25 0.25])
pcolor(X2D_C,Y2D_C,TAUXY/Pini_Pappl/1e6);
col=colorbar;  shading interp; hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
title(['G) \tau_{xy} [MPa]'])
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
pcolor(X2D,Y2D,ETA*1e20);
col=colorbar;  shading interp;hold on
plot(ELLIPSE_XY(1,:),ELLIPSE_XY(2,:),'--w','linewidth',1)
title(['I) \eta^s [Pas]'])
axis equal; axis(ax)
xlabel('Width / radius [ ]'); %ylabel('Height / radius [ ]')

set(gcf,'position',[524.2000 79.8000 986.4000 698.4000])

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
