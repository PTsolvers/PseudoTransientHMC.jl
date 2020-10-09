# const USE_GPU  = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
# const GPU_ID   = 0
# using ParallelStencil
# using ParallelStencil.FiniteDifferences2D
# @static if USE_GPU
#     @init_parallel_stencil(CUDA, Float64, 2)
#     CUDA.device!(GPU_ID) # select GPU
# else
#     @init_parallel_stencil(Threads, Float64, 2)
# end
using Plots, Printf, Statistics, LinearAlgebra
using MAT, Interpolations
################################ Macros from cuda_scientific
# import ParallelStencil: INDICES
# ix,  iy  = INDICES[1], INDICES[2]
# ixi, iyi = :($ix+1), :($iy+1)

# macro av_xya(A) esc(:( ($A[$ix, $iy] + $A[($ix+1), $iy] + $A[$ix,($iy+1)] + $A[($ix+1),($iy+1)])*0.25 )) end
# ################################
@views av_xy(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views function swell2!(B, A, ndim)
	# Logic: assume 3 points x1 x2 x3 and that the "extended" array is xe
	# we have 2 equations (assuming linear extrapolations)
	# 1: xe1 + xe2 = 2*x1
	# 2: xe2       = (x1+x2)/2
	# Substitute the second into the first and solve for xe1 gives
	# xe1 + x1/2 + x2/2 = (4/2)*x1 -> xe1 = 3/2*x1 -1/2*x2
	B .= B.*0.0
	if ndim==1
		B[2:end-1,:] .= (A[2:end,:]  .+ A[1:end-1,:])./2.0
		B[1      ,:] .= 1.5*A[1  ,:] .- 0.5*A[2    ,:]
		B[end    ,:] .= 1.5*A[end,:] .- 0.5*A[end-1,:]
	elseif ndim==2
		B[:,2:end-1] .= (A[:,2:end]  .+ A[:,1:end-1])./2.0
		B[:,1      ] .= 1.5*A[:,1  ] .- 0.5*A[:,2    ]
		B[:,end    ] .= 1.5*A[:,end] .- 0.5*A[:,end-1]
	end
	return B
end
# ################################
# @parallel_indices (ix,iy) function initialize!(xc, yc, lx::Data.Number, ly::Data.Number, μsi::Data.Number, ρgi::Data.Number, Mus::Data.Array, Rog::Data.Array)
    
#     if (ix<=size(xc, 1) && iy<=size(yc ,1)) radc = ((xc[ix]-lx*0.5))*((xc[ix]-lx*0.5)) + ((yc[iy]-ly*0.5))*((yc[iy]-ly*0.5)); end
#     if (ix<=size(Mus,1) && iy<=size(Mus,2)) if (radc<1.0)  Mus[ix,iy] = μsi; end; end
#     if (ix<=size(Rog,1) && iy<=size(Rog,2)) if (radc<1.0)  Rog[ix,iy] = ρgi; end; end
    
#     return
# end

# @parallel function timesteps!(Vsc::Data.Number, Ptsc::Data.Number, min_dxy2::Data.Number, max_nxy::Int, dτVx::Data.Array, dτVy::Data.Array, dτPt::Data.Array, Mus::Data.Array)

#     @all(dτVx) = Vsc*min_dxy2/@av_xi(Mus)/4.1
#     @all(dτVy) = Vsc*min_dxy2/@av_yi(Mus)/4.1
#     @all(dτPt) = Ptsc*4.1*@all(Mus)/max_nxy

#     return
# end

# @parallel function compute_PT!(divV::Data.Array, Vx::Data.Array, Vy::Data.Array, Pt::Data.Array, dτPt::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Mus::Data.Array, _dx::Data.Number, _dy::Data.Number)

#     @all(divV) = _dx*@d_xa(Vx) + _dy*@d_ya(Vy)
#     @all(Pt)   = @all(Pt)  -  @all(dτPt)*@all(divV) 

#     @all(τxx) = 2.0*@all(Mus)*(_dx*@d_xa(Vx)  - 0.333*@all(divV))  
#     @all(τyy) = 2.0*@all(Mus)*(_dy*@d_ya(Vy)  - 0.333*@all(divV))
#     @all(τxy) =  @av_xya(Mus)*(_dy*@d_yi(Vx) + _dx*@d_xi(Vy))  # Free slip BC; 2*0.5 = 1 DEBUG: free slip BC

#     return
# end

# @parallel function compute_dV!(dampX::Data.Number, dampY::Data.Number, Pt::Data.Array, Rog::Data.Array, τxx::Data.Array, τyy::Data.Array, τxy::Data.Array, Rx::Data.Array, Ry::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, _dx::Data.Number, _dy::Data.Number)

#     @all(Rx)    = _dx*@d_xi(τxx) + _dy*@d_ya(τxy) - _dx*@d_xi(Pt)
#     @all(Ry)    = _dy*@d_yi(τyy) + _dx*@d_xa(τxy) - _dy*@d_yi(Pt) + @av_yi(Rog)
#     @all(dVxdτ) = dampX*@all(dVxdτ) + @all(Rx) 
#     @all(dVydτ) = dampY*@all(dVydτ) + @all(Ry)

#     return
# end

# @parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, dVxdτ::Data.Array, dVydτ::Data.Array, dτVx::Data.Array, dτVy::Data.Array)

#     @inn(Vx) = @inn(Vx) + @all(dτVx)*@all(dVxdτ)
#     @inn(Vy) = @inn(Vy) + @all(dτVy)*@all(dVydτ)
    
#     return
# end

# @parallel_indices (ix,iy) function bc_X!(A::Data.Array)

#     if (ix==1          && iy<=size(A ,2)) A[ix, iy] = A[ix+1, iy] end
#     if (ix==size(A, 1) && iy<=size(A ,2)) A[ix, iy] = A[ix-1, iy] end

#     return
# end

# @parallel_indices (ix,iy) function bc_Y!(A::Data.Array)

#     if (ix<=size(A ,1) && iy==1         ) A[ix, iy] = A[ix, iy+1] end
#     if (ix<=size(A, 1) && iy==size(A ,2)) A[ix, iy] = A[ix, iy-1] end

#     return
# end

##################################################
@views function PT_HMC_v0()

    # read in mat file
    vars = matread("LOOK_UP_HMC_Pub.mat")
    RHO_s_LU = get(vars, "Rho_s_07",1)
    Rho_f_LU = get(vars, "Rho_f",1)
    X_LU     = get(vars, "X_s_vec",1)
    P_LU     = get(vars, "P_vec",1)*1e8

    # Independent parameters
    radius          = 1.0          # Radius of initial P-perturbation [m]
    eta_mat         = 1.0          # Viscosity scale [Pa s]
    P_ini           = 1.0          # Initial ambient pressure [Pa]
    rho_0           = 3000.0       # Density scale [kg/m^3]
    # Nondimensional parameters
    ellipse_factor  = 3.0          # Use 1 for circle
    angle           = 30.0         # Counterclockwise angle of long axis with respect to vertical direction
    phi_ini         = 2e-3         # Initial porosity
    eta_incl_fac    = 1e-3         # Factor, how much solid SHEAR viscosity of inclusion is larger (factor>1) or smaller than surrounding
    lambda_incl_fac = 1.0          # Factor, how much solid BULK viscosity of inclusion is larger (factor>1) or smaller than surrounding
    n_exp           = 3.0          # Stress exponent of matrix; n=1 means linear viscous
    lam_eta         = 1e0          # lam_eta = lambda / eta_mat; []; Ratio of bulk to shear viscosity
    Lc_rad2         = 1e8          # Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
    Da              = 0.0024       # Da   = eb*eta_mat/P_ini; []; Ratio of viscous stress to initial stress
    sig_yield       = 0.024        # Stress_ref / P_ini; []; Reference stress used for power-law viscous flow law
    Lx_rad          = 10.0         # Model width divided by inclusion radius
    Ly_Lx           = 1.0          # Model height divided by model width
    Pini_Pappl      = P_ini/8.5e8  # Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
    # Dependant parameters
    beta_eff        = 1e-2/P_ini               # Effective compressibility used only to determine PT time step [1/Pa]
    k_etaf          = Lc_rad2*radius^2/eta_mat # Permeability divided by fluid viscosity; [m^2/(Pa s)]
    P_pert          = 0.2*P_ini                # Pressure perturbation [Pa]
    lambda          = lam_eta*eta_mat          # Bulk viscosity [Pa s]
    eb              = Da*P_ini/eta_mat         # Background strain rate in matrix [1/s]
    stress_ref      = sig_yield *P_ini         # Stress reference for power-law viscosity
    Lx              = Lx_rad*radius            # Model width [m]
    Ly              = Ly_Lx*Lx                 # Model height [m]
    P_LU            = P_LU*Pini_Pappl          # Transform look-up table stress to PT stress scale
    # Numerical resolution
    nx              = 101                              # Numerical resolution width
    ny              = nx+1                             # Numerical resolution height
    tol             = 10^(-5)                          # Tolerance for pseudo-transient iterations
    cfl             = 1/16.1                           # CFL parameter for PT-Stokes solution
    dtp             = 2e0*radius^2/(k_etaf/beta_eff)   # Time step physical
    time_tot        = 1.0*dtp                          # Total time of simulation
    relax           = 0.5
    itmax           = 3e4
    nout            = 1e3
    # Configuration of grid, matrices and numerical parameters
    dx              = Lx/(nx-1)                     # Grid spacing
    dy              = Ly/(ny-1)                     # Grid spacing
    xc              = -Lx/2:dx:Lx/2                 # Coordinate vector
    yc              = -Ly/2:dy:Ly/2                 # Coordinate vector
    xv              = -Lx/2-dx/2:dx:Lx/2+dx/2  # Horizontal vector for Vx which is one more than basic grid
    yv              = -Ly/2-dy/2:dy:Ly/2+dy/2  # Verical    vector for Vy which is one more than basic grid
    (Xc2, Yc2)      = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    (Xc2vx, Yc2vx)  = ([x for x=xv, y=yc], [y for x=xv, y=yc])
    (Xc2vy, Yc2vy)  = ([x for x=xc, y=yv], [y for x=xc, y=yv])
    Pf              = P_ini*ones(nx,  ny)           # Initial ambient fluid pressure
    rad_a           = radius
    rad_b           = rad_a*ellipse_factor
    X_ROT           =  Xc2*cosd(angle)+Yc2*sind(angle)
    Y_ROT           = -Xc2*sind(angle)+Yc2*cosd(angle)
    Pf[sqrt.(X_ROT.^2.0./rad_a.^2.0 .+ Y_ROT.^2.0./rad_b.^2) .< 1.0] .= P_ini - P_pert  # Fluid pressure petubation
    X_Ellipse       =  rad_a.*cos.(0:0.01:2*pi).*cosd(angle).+rad_b.*sin.(0:0.01:2*pi).*sind(angle)
    Y_Ellipse       = -rad_a.*cos.(0:0.01:2*pi).*sind(angle).+rad_b.*sin.(0:0.01:2*pi).*cosd(angle)
    ELLIPSE_XY      = (-X_Ellipse, Y_Ellipse)
    for smo=1:3 # Smooting of perturbation
        Pf[2:end-1,:] .= Pf[2:end-1,:] .+ 0.4.*(Pf[3:end,:].-2.0.*Pf[2:end-1,:].+Pf[1:end-2,:])
        Pf[:,2:end-1] .= Pf[:,2:end-1] .+ 0.4.*(Pf[:,3:end].-2.0.*Pf[:,2:end-1].+Pf[:,1:end-2])
    end
    Pf_inip         = Pf
    # Density, compressibility and gamma from concentration and pressure from thermodynamic data base
    itp1      = interpolate( (P_LU[:,1],), RHO_s_LU[:,1], Gridded(Linear()))
    itp2      = interpolate( (P_LU[:,1],), Rho_f_LU[:,1], Gridded(Linear()))
    itp3      = interpolate( (P_LU[:,1],),     X_LU[:,1], Gridded(Linear()))
    RHO_S     = itp1.(Pf)./rho_0
    RHO_F     = itp2.(Pf)./rho_0
    X_S       = itp3.(Pf)
    RHO_S_ini = itp1.(P_ini*ones(nx, ny))./rho_0
    X_S_ini   = itp3.(P_ini*ones(nx, ny))
    # Initialize ALL arrays in Julia
    Ptot            = zeros(nx,  ny)              # Initial ambient fluid pressure
    Ptot           .= Pf                          # Initial total pressure
    PHI_INI         = phi_ini*ones(nx, ny)
    PHI             = PHI_INI
    K_DARCY         = k_etaf*ones(nx, ny)
    DIVV            = zeros(nx  , ny  )
    DIVV_RHO_X      = zeros(nx  , ny  )
    DIVV_RHO_T      = zeros(nx  , ny  )
    TAUXX           = zeros(nx  , ny  )               # Deviatoric stress
    TAUYY           = zeros(nx  , ny  )               # Deviatoric stress
    TAUXY           = zeros(nx-1, ny-1)            # Deviatoric stress
    RES_VX          = zeros(nx-1, ny-2)
    RES_VY          = zeros(nx-2, ny-1)
    RHO_S_old       = zeros(nx  , ny  )
    RHO_F_old       = zeros(nx  , ny  )
    X_S_old         = zeros(nx  , ny  )
    RHO_X_old       = zeros(nx  , ny  )
    PHI_old         = zeros(nx  , ny  )
    Ptot_old        = zeros(nx  , ny  )
    RHO_X_INI       = zeros(nx  , ny  )
    PHI_ini         = zeros(nx  , ny  )
    PHI             = zeros(nx  , ny  )
    RHO_T_old       = zeros(nx  , ny  )
    ETA_MAT         = eta_mat.*ones(nx, ny)
    ETA_PL          = eta_mat.*ones(nx, ny)
    ETA             = eta_mat.*ones(nx, ny)
    LAMBDA          =  lambda.*ones(nx, ny)         # Viscosity
    RHO_T           = zeros(nx  , ny  )
    para_cx         = zeros(nx-1, ny  )
    para_cy         = zeros(nx  , ny-1)
    q_f_X           = zeros(nx-1, ny  )
    q_f_Y           = zeros(nx  , ny-1)
    q_f_X_Ptot      = zeros(nx-1, ny  )
    q_f_Y_Ptot      = zeros(nx  , ny-1)
    DIV_q_f         = zeros(nx-2, ny-2)
    RES_Pf          = zeros(nx-2, ny-2)
    RHO_X           = zeros(nx  , ny  )
    RES_PHI         = zeros(nx  , ny  )
    DXX             = zeros(nx  , ny  )
    DYY             = zeros(nx  , ny  )
    DXY             = zeros(nx-1, ny-1)
    TII             = zeros(nx  , ny  )
    ETA_PL_old_ptit = zeros(nx  , ny  )

    TmpX            = zeros(nx+1, ny  )
    TmpY            = zeros(nx  , ny+1)
    TmpS1           = zeros(nx  , ny-1)
    TmpS2           = zeros(nx  , ny  )
    # arrays for visu
    VX_f            = zeros(nx  , ny  )
    VY_f            = zeros(nx  , ny  )
    VX_f_Ptot       = zeros(nx  , ny  )
    VY_f_Ptot       = zeros(nx  , ny  )
    # Centered coordinates needed for staggered grid for solid velocities
    VX              = -eb*Xc2vx                   # Pure shear, shortening in x
    VY              =  eb*Yc2vy                   # Pure shear, extension in y
    # Parameters for time loop and pseudo-transient iterations
    timeP           = 0.0                         # Initial time
    it              = 0                           # Integer count for iteration loop
    itp             = 0                           # Integer count for time loop
    save_count      = 0 
    Time_vec        = []
    # time loop
    while timeP < time_tot
    	err_M=2*tol; itp+=1.0
    	timeP=timeP+dtp; push!(Time_vec, timeP)
    	RHO_S_old      .= RHO_S
    	RHO_F_old      .= RHO_F
    	X_S_old        .= X_S
    	RHO_X_old      .= RHO_S_old.*X_S_old
    	PHI_old        .= PHI
    	Ptot_old       .= Ptot
    	if itp==1
    		RHO_X_INI  .= RHO_S_ini.*X_S_ini
    		PHI_old    .= 1.0 .- (RHO_X_INI).*(1.0 .- phi_ini)./RHO_X_old
    		PHI_ini    .= PHI_old
    		PHI        .= PHI_old
    	end
    	RHO_T_old      .= RHO_F_old.*PHI_old .+ RHO_S_old.*(1.0 .- PHI_old)
    	ETA_MAT        .= 0.0 .+ eta_mat             # Ambient viscosity
    	ETA_MAT[PHI.>(maximum(PHI)+minimum(PHI))./2.0] .= eta_mat*eta_incl_fac         # Inclusion viscosity
    	ETA_PL         .= ETA_MAT
    	ETA            .= ETA_MAT
    	LAMBDA         .= 0.0 .+ lambda
    	LAMBDA[PHI.>(maximum(PHI)+minimum(PHI))./2.0]  .= lambda*lambda_incl_fac;  # Fluid pressure petubation
    	# PT loop
    	it_tstep=0; err_evo1=[]; err_evo2=[]
    	while err_M>tol && it_tstep<itmax
        	it+=1; it_tstep+=1
        	dt_Stokes   = cfl*max(dx,dy).^2/maximum(ETA)                       # Pseudo time step for Stokes
        	dt_Pf       = cfl*max(dx,dy).^2/maximum(K_DARCY.*PHI.^3/beta_eff)  # Pseudo time step for fluid pressure
        	# Fluid pressure evolution
        	RHO_T      .= RHO_F.*PHI .+ RHO_S.*(1.0.-PHI)
        	para_cx    .= ( RHO_F[1:end-1,:].*K_DARCY[1:end-1,:].*PHI[1:end-1,:].^3. .+ RHO_F[2:end,:].*K_DARCY[2:end,:].*PHI[2:end,:].^3 )./2.0
        	para_cy    .= ( RHO_F[:,1:end-1].*K_DARCY[:,1:end-1].*PHI[:,1:end-1].^3. .+ RHO_F[:,2:end].*K_DARCY[:,2:end].*PHI[:,2:end].^3 )./2.0
        	q_f_X      .= -para_cx.*diff(Pf  ,dims=1)./dx        # Correct   Darcy flux with fluid pressure
        	q_f_Y      .= -para_cy.*diff(Pf  ,dims=2)./dy        # Correct   Darcy flux with fluid pressure
        	q_f_X_Ptot .= -para_cx.*diff(Ptot,dims=1)./dx        # Incorrect Darcy flux with total pressure
        	q_f_Y_Ptot .= -para_cy.*diff(Ptot,dims=2)./dy        # Incorrect Darcy flux with total pressure
        	DIV_q_f    .= 0*diff(q_f_X[:,2:end-1],dims=1)./dx .+ diff(q_f_Y[2:end-1,:],dims=2)./dy
        	RES_Pf     .= -DIV_q_f .-(RHO_T[2:end-1,2:end-1].-RHO_T_old[2:end-1,2:end-1])./dtp .- DIVV_RHO_T[2:end-1,2:end-1] # CONSERVATION OF TOTAL MASS EQUATION
        	Pf[2:end-1,2:end-1] .= Pf[2:end-1,2:end-1] .+ dt_Pf.*RES_Pf
        	# Look up for densities
        	RHO_S      .= itp1.(Pf)./rho_0
        	RHO_F      .= itp2.(Pf)./rho_0
        	X_S        .= itp3.(Pf)
        	# Porosity evolution
        	RHO_X      .= RHO_S.*X_S
        	RES_PHI    .= ( (1.0.-PHI).*RHO_X .- (1.0.-PHI_old).*RHO_X_old )./dtp .+ DIVV_RHO_X   # CONSERVATION OF MASS OF MgO EQUATION
        	PHI        .= PHI .+ dtp.*RES_PHI
        	# Stokes
        	DIVV_RHO_X .= diff(swell2!(TmpX, (1.0.-PHI).*RHO_X, 1).*VX, dims=1)./dx .+ diff(swell2!(TmpY, (1.0.-PHI).*RHO_X, 2).*VY, dims=2)./dy
        	DIVV_RHO_T .= diff(swell2!(TmpX, RHO_T, 1).*VX, dims=1)./dx .+ diff(swell2!(TmpY, RHO_T, 2).*VY, dims=2)./dy
        	DIVV       .= diff(VX, dims=1)./dx .+ diff(VY, dims=2)./dy                              # Divergence of velocity
        	DXX        .= diff(VX, dims=1)./dx .- 1/3 .*DIVV                                        # Horizontal deviatoric strain rate
        	DYY        .= diff(VY, dims=2)/dy .- 1/3 .*DIVV                                         # Vertical deviatoric strain rate
        	DXY        .= 0.5.*(diff(VX[2:end-1,:], dims=2)./dy .+ diff(VY[:,2:end-1], dims=1)./dx) # Shear strain rate; two smaller
        	Ptot       .= Pf .- DIVV.*(1.0.-PHI).*LAMBDA             # Rheology for total pressure
        	TAUXX      .= 2.0.*ETA.*DXX                              # Horizontal deviatoric stress
        	TAUYY      .= 2.0.*ETA.*DYY                              # Vertical deviatoric stress
        	TAUXY      .= 2.0.*av_xy(ETA).*DXY                       # Shear stress
        	TII        .= sqrt.( 0.25.*(TAUXX-TAUYY).^2 .+ swell2!(TmpS2, swell2!(TmpS1, TAUXY, 1), 2).^2)
        	# power-law
        	if n_exp>1
        		ETA_PL_old_ptit .= ETA_PL                                    # Previous PT viscosity
        		ETA_PL          .= ETA_MAT.*(TII./stress_ref).^(1-n_exp)
        		ETA_PL[TII.<stress_ref] .= ETA_MAT[TII.<stress_ref] #eta_mat
        		ETA_PL          .= exp.(log.(ETA_PL).*relax.+log.(ETA_PL_old_ptit).*(1-relax))
        		ETA             .= 2.0 ./( 1.0./ETA_MAT .+ 1.0./ETA_PL )
        	end
        	RES_VX     .= -diff(Ptot[:,2:end-1], dims=1)./dx .+ diff(TAUXX[:,2:end-1], dims=1)./dx .+ diff(TAUXY, dims=2)./dy  # HORIZONTAL FORCE BALANCE
        	RES_VY     .= -diff(Ptot[2:end-1,:], dims=2)./dy .+ diff(TAUYY[2:end-1,:], dims=2)./dy .+ diff(TAUXY, dims=1)./dx  # VERTICAL   FORCE BALANCE
        	VX[2:end-1,2:end-1] .= VX[2:end-1,2:end-1] .+ dt_Stokes.*RES_VX   # Pseudo-transient form of horizontal force balance
        	VY[2:end-1,2:end-1] .= VY[2:end-1,2:end-1] .+ dt_Stokes.*RES_VY   # Pseudo-transient form of vertical force balance
        	if mod(it_tstep, nout)==0 && it_tstep>250
        		err_Mx   = maximum(abs.(dt_Stokes.*RES_VX)/maximum(abs.(VX)))    # Error horizontal velocitiy
        		err_My   = maximum(abs.(dt_Stokes.*RES_VY)/maximum(abs.(VY)))    # Error vertical velocity
        		err_Pf   = maximum(abs.(dt_Pf.*RES_Pf)/maximum(abs.(Pf)))        # Error fluid pressure
        		err_Phi  = maximum(abs.(dtp.*RES_PHI))                           # Error porosity
        		err_M    = maximum([err_Pf, err_Mx, err_My, err_Phi])            # Error total
        		err_evo1 = push!(err_evo1, it_tstep); err_evo2 = push!(err_evo2, err_M)
        		# plot evol
        		p1 = plot(err_evo1, err_evo2, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        		display(p1)
        	end
        end # end PT loop
    end
    # Visu
    swell2!(VX_f, q_f_X./(RHO_F[1:end-1,:]./PHI[1:end-1,:].+RHO_F[2:end,:]./PHI[2:end,:]).*2.0, 1)
    swell2!(VY_f, q_f_Y./(RHO_F[:,1:end-1]./PHI[:,1:end-1].+RHO_F[:,2:end]./PHI[:,2:end]).*2.0, 2)
    swell2!(VX_f_Ptot, q_f_X_Ptot./(RHO_F[1:end-1,:]./PHI[1:end-1,:].+RHO_F[2:end,:]./PHI[2:end,:]).*2.0, 1)
    swell2!(VY_f_Ptot, q_f_Y_Ptot./(RHO_F[:,1:end-1]./PHI[:,1:end-1].+RHO_F[:,2:end]./PHI[:,2:end]).*2.0, 2)

    Length_model_m = radius
    Time_model_sec = radius^2/(k_etaf/beta_eff)
    Length_phys_m  = 0.01
    Time_phys_sec  = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8))
    Vel_phys_m_s   = (Length_phys_m/Length_model_m) / (Time_phys_sec/Time_model_sec)
    
    p2  = heatmap(xc, yc, Pf'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="A) p_f [kbar]")
    	  plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=2, linecolor="white", legend=false)
    p3  = heatmap(xc, yc, Ptot'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="B) p [kbar]")
    p4  = heatmap(xc, yc, DIVV'.*Time_model_sec./Time_phys_sec, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="C) ∇(v_s) [1/s]")
    p5  = heatmap(xc, yc, sqrt.(VX_f.^2 .+ VY_f.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="D) ||v_f|| [m/s]")
    p6  = heatmap(xc, yc, sqrt.(av_xa(VX).^2 .+ av_ya(VY).^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="E) ||v_s|| [m/s]")
    p7  = heatmap(xc, yc, sqrt.(VX_f_Ptot.^2 .+ VY_f_Ptot.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="F) ||v_f||p [m/s]")
    p8  = heatmap(xv[2:end-1], yv[2:end-1], TAUXY'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xv[2], xv[end-1]), ylims=(yv[2], yv[end-1]), title="G) τxy [MPa]")
    p9  = heatmap(xc, yc, TII'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="H) τII [MPa]")
    p10 = heatmap(xc, yc, ETA'*1e20, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="I) ηs [Pas]")
    display(plot(p2, p3, p4, p5, p6, p7, p8, p9, p10))

    # # Physics
    # lx, ly    = 10.0, 10.0
    # μs0       = 1.0
    # μsi       = 0.1
    # ρgi       = 1.0
    # # Numerics
    # nitr      = 10000
    # nout      = 200
    # BLOCK_X   = 16
    # BLOCK_Y   = 16 
    # GRID_X    = 8
    # GRID_Y    = 8
    # Vdmp      = 4.0
    # Vsc       = 1.0
    # Ptsc      = 1.0/4.0
    # ε         = 1e-5
    # nx        = GRID_X*BLOCK_X - 1 # -1 due to overlength of array nx+1
    # ny        = GRID_Y*BLOCK_Y - 1 # -1 due to overlength of array ny+1
    # cuthreads = (BLOCK_X, BLOCK_Y, 1)
    # cublocks  = (GRID_X , GRID_Y , 1)
    # dx, dy    = lx/nx, ly/ny
    # _dx, _dy  = 1.0/dx, 1.0/dy 
    # # Initialisation
    # Pt        = @zeros(nx  ,ny  )
    # dτPt      = @zeros(nx  ,ny  )
    # divV      = @zeros(nx  ,ny  )
    # Vx        = @zeros(nx+1,ny  )
    # Vy        = @zeros(nx  ,ny+1)
    # τxx       = @zeros(nx  ,ny  )
    # τyy       = @zeros(nx  ,ny  )
    # τxy       = @zeros(nx-1,ny-1)
    # Rx        = @zeros(nx-1,ny-2)
    # Ry        = @zeros(nx-2,ny-1)
    # dVxdτ     = @zeros(nx-1,ny-2)
    # dVydτ     = @zeros(nx-2,ny-1)
    # dτVx      = @zeros(nx-1,ny-2)
    # dτVy      = @zeros(nx-2,ny-1)
    # Rog       = @zeros(nx  ,ny  )
    # Mus       = μs0*@ones(nx,ny)
    # xc        = LinRange(dx/2, lx-dx/2, nx)
    # yc        = LinRange(dy/2, ly-dy/2, ny)
    # min_dxy2  = min(dx,dy)^2
    # max_nxy   = max(nx,ny)
    # dampX     = 1.0-Vdmp/nx
    # dampY     = 1.0-Vdmp/ny
    # # visualisation
    # gr() #pyplot()
    # ENV["GKSwstype"]="nul"; if isdir("viz2D_out")==false mkdir("viz2D_out") end; loadpath = "./viz2D_out/"; anim = Animation(loadpath,String[])
    # println("Animation directory: $(anim.dir)")
    # # Action
    # @parallel cublocks cuthreads initialize!(xc, yc, lx, ly, μsi, ρgi, Mus, Rog)

    # @parallel cublocks cuthreads timesteps!(Vsc, Ptsc, min_dxy2, max_nxy, dτVx, dτVy, dτPt, Mus)

    # err=2*ε; err_evo1=[]; err_evo2=[]; warmup=10
    # for itr = 1:nitr
    #     if (itr==warmup+1) global t0 = Base.time(); end

    #     @parallel cublocks cuthreads compute_PT!(divV, Vx, Vy, Pt, dτPt, τxx, τyy, τxy, Mus, _dx, _dy)

    #     @parallel cublocks cuthreads compute_dV!(dampX, dampY, Pt, Rog, τxx, τyy, τxy, Rx, Ry, dVxdτ, dVydτ, _dx, _dy)

    #     @parallel cublocks cuthreads compute_V!(Vx, Vy, dVxdτ, dVydτ, dτVx, dτVy)

    #     @parallel cublocks cuthreads bc_Y!(Vx)
    #     @parallel cublocks cuthreads bc_X!(Vy)

    #     # convergence check
    #     if mod(itr, nout)==0
    #         global mean_Rx, mean_Ry, mean_divV
    #         mean_Rx = mean(abs.(Rx[:])); mean_Ry = mean(abs.(Ry[:])); mean_divV = mean(abs.(divV[:]))
    #         err = maximum([mean_Rx, mean_Ry, mean_divV])
    #         push!(err_evo1, maximum([mean_Rx, mean_Ry, mean_divV])); push!(err_evo2,itr)
    #         @printf("Total steps = %d, err = %1.3e [mean_Rx=%1.3e, mean_Ry=%1.3e, mean_divV=%1.3e] \n", itc, err, mean_Rx, mean_Ry, mean_divV)
    #     end
    #     if (err<=ε) break; end
    #     global itc=itr
    # end
    # global time_s = Base.time() - t0
    # A_eff = (3*2)/1e9*nx*ny*sizeof(Data.Number)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    # t_it  = time_s/(itc-warmup)                  # Execution time per iteration [s]
    # T_eff = A_eff/t_it                           # Effective memory throughput [GB/s]
    # @printf("Total steps = %d, err = %1.3e, time = %1.3e sec (@ %1.2f GB/s) \n",itc,err,time_s,T_eff)
    # # Plotting
    # yv = LinRange(0, ly, ny+1)
    # p1 = heatmap(xc,yc,Pt', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:inferno, title="Pressure")
    # p2 = heatmap(xc,yv,Vy', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yv[1], yv[end]), c=:inferno, title="Vy")
    # p4 = heatmap(xc[2:end-1],yv[2:end-1],log10.(abs.(Ry')), aspect_ratio=1, xlims=(xc[2], xc[end-1]), ylims=(yc[2], yc[end-1]), c=:inferno, title="log10(Ry)");
    # p5 = plot(err_evo2,err_evo1, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
    # # display(plot( p1, p2, p4, p5 ))
    # plot( p1, p2, p4, p5 ); frame(anim)
    # gif(anim, "Stokes2D.gif", fps = 15)

    return
end

@time PT_HMC_v0()
