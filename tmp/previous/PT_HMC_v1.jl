using Plots, Printf, Statistics, LinearAlgebra
using MAT, Interpolations
##################################################
@views av_xy(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views function swell2!(B, A, ndim)
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
##################################################
@views function PT_HMC()
    # read in mat file
    vars            = matread("LOOK_UP_HMC_Pub.mat")
    RHO_s_LU        = get(vars, "Rho_s_07",1)
    Rho_f_LU        = get(vars, "Rho_f",1)
    X_LU            = get(vars, "X_s_vec",1)
    P_LU            = get(vars, "P_vec",1)*1e8
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
    Ptot            = zeros(nx  , ny  )              # Initial ambient fluid pressure
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
    # TMP arrays for swell2
    TmpX            = zeros(nx+1, ny  )
    TmpY            = zeros(nx  , ny+1)
    TmpS1           = zeros(nx  , ny-1)
    TmpS2           = zeros(nx  , ny  )
    # arrays for visu
    VX_f            = zeros(nx  , ny  )
    VY_f            = zeros(nx  , ny  )
    VX_f_Ptot       = zeros(nx  , ny  )
    VY_f_Ptot       = zeros(nx  , ny  )
    Ptot           .= Pf                          # Initial total pressure
    PHI_INI         = phi_ini*ones(nx, ny)
    PHI             = phi_ini*ones(nx, ny)
    K_DARCY         =  k_etaf*ones(nx, ny)
    ETA_MAT         = eta_mat*ones(nx, ny)
    ETA_PL          = eta_mat*ones(nx, ny)
    ETA             = eta_mat*ones(nx, ny)
    LAMBDA          =  lambda*ones(nx, ny)         # Viscosity
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
    lw = 1.2
    p2  = heatmap(xc, yc, Pf'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="A) p_f [kbar]")
    	    plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p3  = heatmap(xc, yc, Ptot'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="B) p [kbar]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p4  = heatmap(xc, yc, DIVV'.*Time_model_sec./Time_phys_sec, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="C) ∇(v_s) [1/s]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p5  = heatmap(xc, yc, sqrt.(VX_f.^2 .+ VY_f.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="D) ||v_f|| [m/s]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p6  = heatmap(xc, yc, sqrt.(av_xa(VX).^2 .+ av_ya(VY).^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="E) ||v_s|| [m/s]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p7  = heatmap(xc, yc, sqrt.(VX_f_Ptot.^2 .+ VY_f_Ptot.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="F) ||v_f||p [m/s]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p8  = heatmap(xv[2:end-1], yv[2:end-1], TAUXY'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xv[2], xv[end-1]), ylims=(yv[2], yv[end-1]), c=:viridis, title="G) τxy [MPa]")
    p9  = heatmap(xc, yc, TII'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="H) τII [MPa]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    p10 = heatmap(xc, yc, ETA'*1e20, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="I) ηs [Pas]")
            plot!(ELLIPSE_XY[1], ELLIPSE_XY[2], linewidth=lw, linecolor="white", legend=false)
    display(plot(p2, p3, p4, p5, p6, p7, p8, p9, p10))

    return
end

@time PT_HMC()
