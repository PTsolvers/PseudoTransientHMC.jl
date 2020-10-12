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
    Rho_s_LU        = get(vars, "Rho_s_07",1)
    Rho_f_LU        = get(vars, "Rho_f"   ,1)
    X_LU            = get(vars, "X_s_vec" ,1)
    P_LU            = get(vars, "P_vec"   ,1)*1e8
    # Independent parameters
    rad             = 1.0          # rad of initial P-perturbation [m]
    η_m             = 1.0          # Viscosity scale [Pa s]
    P_ini           = 1.0          # Initial ambient pressure [Pa]
    ρ_0             = 3000.0       # Density scale [kg/m^3]
    # Nondimensional parameters
    elli_fac        = 3.0          # Use 1 for circle
    α               = 30.0         # Counterclockwise α of long axis with respect to vertical direction
    ϕ_ini           = 2e-3         # Initial porosity
    η_i_fac         = 1e-3         # Factor, how much solid SHEAR viscosity of inclusion is larger (factor>1) or smaller than surrounding
    λ_i_fac         = 1.0          # Factor, how much solid BULK viscosity of inclusion is larger (factor>1) or smaller than surrounding
    n_exp           = 3.0          # Stress exponent of matrix; n=1 means linear viscous
    λ_η             = 1e0          # λ_η = λ / η_m; []; Ratio of bulk to shear viscosity
    lc_rad2         = 1e8          # lc_rad2 = k_ηf*η_m/rad^2; []; Ratio of hydraulic fluid extraction to compaction extraction
    Da              = 0.0024       # Da   = ε_bg*η_m/P_ini; []; Ratio of viscous stress to initial stress
    σ_y             = 0.024        # Stress_ref / P_ini; []; Reference stress used for power-law viscous flow law
    lx_rad          = 10.0         # Model width divided by inclusion rad
    ly_lx           = 1.0          # Model height divided by model width
    Pini_Pappl      = P_ini/8.5e8  # Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
    # Dependant parameters
    β_eff           = 1e-2/P_ini           # Effective compressibility used only to determine PT time step [1/Pa]
    k_ηf            = lc_rad2*rad^2/η_m    # Permeability divided by fluid viscosity; [m^2/(Pa s)]
    P_pert          = 0.2*P_ini            # Pressure perturbation [Pa]
    λ               = λ_η*η_m              # Bulk viscosity [Pa s]
    ε_bg            = Da*P_ini/η_m         # Background strain rate in matrix [1/s]
    σ_ref           = σ_y*P_ini            # Stress reference for power-law viscosity
    lx              = lx_rad*rad           # Model width [m]
    ly              = ly_lx*lx             # Model height [m]
    P_LU            = P_LU*Pini_Pappl      # Transform look-up table stress to PT stress scale
    # Numerical resolution
    nx              = 100                              # Numerical resolution width
    ny              = nx+1                             # Numerical resolution height
    tol             = 1e-4                             # Tolerance for pseudo-transient iterations
    cfl             = 1/16.1                           # CFL parameter for PT-Stokes solution
    dtp             = 2e0*rad^2/(k_ηf/β_eff)           # Time step physical
    time_tot        = 1.0*dtp                          # Total time of simulation
    relax           = 0.5
    itmax           = 3e4
    nout            = 1e3
    # Configuration of grid, matrices and numerical parameters
    dx              = lx/(nx-1)                     # Grid spacing
    dy              = ly/(ny-1)                     # Grid spacing
    xc              = -lx/2:dx:lx/2                 # Coordinate vector
    yc              = -ly/2:dy:ly/2                 # Coordinate vector
    xv              = -lx/2-dx/2:dx:lx/2+dx/2  # Horizontal vector for Vx which is one more than basic grid
    yv              = -ly/2-dy/2:dy:ly/2+dy/2  # Verical    vector for Vy which is one more than basic grid
    (Xc2, Yc2)      = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    (Xc2vx, Yc2vx)  = ([x for x=xv, y=yc], [y for x=xv, y=yc])
    (Xc2vy, Yc2vy)  = ([x for x=xc, y=yv], [y for x=xc, y=yv])
    Pf              = P_ini*ones(nx,  ny)           # Initial ambient fluid pressure
    rad_a           = rad
    rad_b           = rad_a*elli_fac
    X_rot           =  Xc2*cosd(α)+Yc2*sind(α)
    Y_rot           = -Xc2*sind(α)+Yc2*cosd(α)
    Pf[sqrt.(X_rot.^2.0./rad_a.^2.0 .+ Y_rot.^2.0./rad_b.^2) .< 1.0] .= P_ini - P_pert  # Fluid pressure petubation
    X_elli          =  rad_a.*cos.(0:0.01:2*pi).*cosd(α).+rad_b.*sin.(0:0.01:2*pi).*sind(α)
    Y_elli          = -rad_a.*cos.(0:0.01:2*pi).*sind(α).+rad_b.*sin.(0:0.01:2*pi).*cosd(α)
    XY_elli         = (-X_elli, Y_elli)
    for smo=1:3 # Smooting of perturbation
        Pf[2:end-1,:] .= Pf[2:end-1,:] .+ 0.4.*(Pf[3:end,:].-2.0.*Pf[2:end-1,:].+Pf[1:end-2,:])
        Pf[:,2:end-1] .= Pf[:,2:end-1] .+ 0.4.*(Pf[:,3:end].-2.0.*Pf[:,2:end-1].+Pf[:,1:end-2])
    end
    # Density, compressibility and gamma from concentration and pressure from thermodynamic data base
    itp1      = interpolate( (P_LU[:,1],), Rho_s_LU[:,1], Gridded(Linear()))
    itp2      = interpolate( (P_LU[:,1],), Rho_f_LU[:,1], Gridded(Linear()))
    itp3      = interpolate( (P_LU[:,1],),     X_LU[:,1], Gridded(Linear()))
    Rho_s     = itp1.(Pf)./ρ_0
    Rho_f     = itp2.(Pf)./ρ_0
    X_s       = itp3.(Pf)
    Rho_s_ini = itp1.(P_ini*ones(nx, ny))./ρ_0
    X_s_ini   = itp3.(P_ini*ones(nx, ny))
    # Initialize ALL arrays in Julia
    Ptot            = zeros(nx  , ny  )              # Initial ambient fluid pressure
    ∇V              = zeros(nx  , ny  )
    ∇V_ρ_x          = zeros(nx  , ny  )
    ∇V_ρ_t          = zeros(nx  , ny  )
    τ_xx            = zeros(nx  , ny  )               # Deviatoric stress
    τ_yy            = zeros(nx  , ny  )               # Deviatoric stress
    τ_xy            = zeros(nx-1, ny-1)            # Deviatoric stress
    Res_Vx          = zeros(nx-1, ny-2)
    Res_Vy          = zeros(nx-2, ny-1)
    Rho_s_old       = zeros(nx  , ny  )
    Rho_f_old       = zeros(nx  , ny  )
    X_s_old         = zeros(nx  , ny  )
    Rho_X_old       = zeros(nx  , ny  )
    Phi_old         = zeros(nx  , ny  )
    Ptot_old        = zeros(nx  , ny  )
    Rho_X_ini       = zeros(nx  , ny  )
    Rho_t_old       = zeros(nx  , ny  )
    Rho_t           = zeros(nx  , ny  )
    para_cx         = zeros(nx-1, ny  )
    para_cy         = zeros(nx  , ny-1)
    q_f_X           = zeros(nx-1, ny  )
    q_f_Y           = zeros(nx  , ny-1)
    q_f_X_Ptot      = zeros(nx-1, ny  )
    q_f_Y_Ptot      = zeros(nx  , ny-1)
    ∇q_f            = zeros(nx-2, ny-2)
    Res_Pf          = zeros(nx-2, ny-2)
    Rho_X           = zeros(nx  , ny  )
    Res_Phi         = zeros(nx  , ny  )
    ε_xx            = zeros(nx  , ny  )
    ε_yy            = zeros(nx  , ny  )
    ε_xy            = zeros(nx-1, ny-1)
    τII             = zeros(nx  , ny  )
    Eta_iter        = zeros(nx  , ny  )
    # TMP arrays for swell2
    TmpX            = zeros(nx+1, ny  )
    TmpY            = zeros(nx  , ny+1)
    TmpS1           = zeros(nx  , ny-1)
    TmpS2           = zeros(nx  , ny  )
    # arrays for visu
    Vx_f            = zeros(nx  , ny  )
    Vy_f            = zeros(nx  , ny  )
    Vx_f_Ptot       = zeros(nx  , ny  )
    VY_f_Ptot       = zeros(nx  , ny  )
    # init
    Phi_ini         = ϕ_ini*ones(nx, ny)
    Phi             = ϕ_ini*ones(nx, ny)
    K_ηf            =  k_ηf*ones(nx, ny)
    Eta_m           =   η_m*ones(nx, ny)
    Eta_pl          =   η_m*ones(nx, ny)
    Eta             =   η_m*ones(nx, ny)
    Lam             =     λ*ones(nx, ny)         # Viscosity
    Ptot           .= Pf                          # Initial total pressure
    Vx              = -ε_bg*Xc2vx                   # Pure shear, shortening in x
    Vy              =  ε_bg*Yc2vy                   # Pure shear, extension in y
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
    	Rho_s_old      .= Rho_s
    	Rho_f_old      .= Rho_f
    	X_s_old        .= X_s
    	Rho_X_old      .= Rho_s_old.*X_s_old
    	Phi_old        .= Phi
    	Ptot_old       .= Ptot
    	if itp==1
    		Rho_X_ini  .= Rho_s_ini.*X_s_ini
    		Phi_old    .= 1.0 .- (Rho_X_ini).*(1.0 .- ϕ_ini)./Rho_X_old
    		Phi_ini    .= Phi_old
    		Phi        .= Phi_old
    	end
    	Rho_t_old      .= Rho_f_old.*Phi_old .+ Rho_s_old.*(1.0 .- Phi_old)
    	Eta_m          .= 0.0 .+ η_m             # Ambient viscosity
    	Eta_m[Phi.>(maximum(Phi)+minimum(Phi))./2.0] .= η_m*η_i_fac         # Inclusion viscosity
    	Eta_pl         .= Eta_m
    	Eta            .= Eta_m
    	Lam            .= 0.0 .+ λ
    	Lam[Phi.>(maximum(Phi)+minimum(Phi))./2.0]  .= λ*λ_i_fac;  # Fluid pressure petubation
    	# PT loop
    	it_tstep=0; err_evo1=[]; err_evo2=[]
    	while err_M>tol && it_tstep<itmax
        	it+=1; it_tstep+=1
        	dt_Stokes   = cfl*max(dx,dy).^2/maximum(Eta)                       # Pseudo time step for Stokes
        	dt_Pf       = cfl*max(dx,dy).^2/maximum(K_ηf.*Phi.^3/β_eff)  # Pseudo time step for fluid pressure
        	# Fluid pressure evolution
        	Rho_t      .= Rho_f.*Phi .+ Rho_s.*(1.0.-Phi)
        	para_cx    .= ( Rho_f[1:end-1,:].*K_ηf[1:end-1,:].*Phi[1:end-1,:].^3. .+ Rho_f[2:end,:].*K_ηf[2:end,:].*Phi[2:end,:].^3 )./2.0
        	para_cy    .= ( Rho_f[:,1:end-1].*K_ηf[:,1:end-1].*Phi[:,1:end-1].^3. .+ Rho_f[:,2:end].*K_ηf[:,2:end].*Phi[:,2:end].^3 )./2.0
        	q_f_X      .= -para_cx.*diff(Pf  ,dims=1)./dx        # Correct   Darcy flux with fluid pressure
        	q_f_Y      .= -para_cy.*diff(Pf  ,dims=2)./dy        # Correct   Darcy flux with fluid pressure
        	q_f_X_Ptot .= -para_cx.*diff(Ptot,dims=1)./dx        # Incorrect Darcy flux with total pressure
        	q_f_Y_Ptot .= -para_cy.*diff(Ptot,dims=2)./dy        # Incorrect Darcy flux with total pressure
        	∇q_f       .= diff(q_f_X[:,2:end-1],dims=1)./dx .+ diff(q_f_Y[2:end-1,:],dims=2)./dy
        	Res_Pf     .= -∇q_f .-(Rho_t[2:end-1,2:end-1].-Rho_t_old[2:end-1,2:end-1])./dtp .- ∇V_ρ_t[2:end-1,2:end-1] # CONSERVATION OF TOTAL MASS EQUATION
        	Pf[2:end-1,2:end-1] .= Pf[2:end-1,2:end-1] .+ dt_Pf.*Res_Pf
        	# Look up for densities
        	Rho_s      .= itp1.(Pf)./ρ_0
        	Rho_f      .= itp2.(Pf)./ρ_0
        	X_s        .= itp3.(Pf)
        	# Porosity evolution
        	Rho_X      .= Rho_s.*X_s
        	Res_Phi    .= ( (1.0.-Phi).*Rho_X .- (1.0.-Phi_old).*Rho_X_old )./dtp .+ ∇V_ρ_x   # CONSERVATION OF MASS OF MgO EQUATION
        	Phi        .= Phi .+ dtp.*Res_Phi
        	# Stokes
        	∇V_ρ_x     .= diff(swell2!(TmpX, (1.0.-Phi).*Rho_X, 1).*Vx, dims=1)./dx .+ diff(swell2!(TmpY, (1.0.-Phi).*Rho_X, 2).*Vy, dims=2)./dy
        	∇V_ρ_t     .= diff(swell2!(TmpX, Rho_t, 1).*Vx, dims=1)./dx .+ diff(swell2!(TmpY, Rho_t, 2).*Vy, dims=2)./dy
        	∇V         .= diff(Vx, dims=1)./dx .+ diff(Vy, dims=2)./dy                              # Divergence of velocity
        	ε_xx       .= diff(Vx, dims=1)./dx .- 1/3 .*∇V                                        # Horizontal deviatoric strain rate
        	ε_yy       .= diff(Vy, dims=2)/dy .- 1/3 .*∇V                                         # Vertical deviatoric strain rate
        	ε_xy       .= 0.5.*(diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1], dims=1)./dx) # Shear strain rate; two smaller
        	Ptot       .= Pf .- ∇V.*(1.0.-Phi).*Lam             # Rheology for total pressure
        	τ_xx       .= 2.0.*Eta.*ε_xx                              # Horizontal deviatoric stress
        	τ_yy       .= 2.0.*Eta.*ε_yy                              # Vertical deviatoric stress
        	τ_xy       .= 2.0.*av_xy(Eta).*ε_xy                       # Shear stress
        	τII        .= sqrt.( 0.25.*(τ_xx-τ_yy).^2 .+ swell2!(TmpS2, swell2!(TmpS1, τ_xy, 1), 2).^2)
        	# power-law
        	if n_exp>1
        		Eta_iter  .= Eta_pl                                    # Previous PT viscosity
        		Eta_pl    .= Eta_m.*(τII./σ_ref).^(1-n_exp)
        		Eta_pl[τII.<σ_ref] .= Eta_m[τII.<σ_ref] #η_m
        		Eta_pl     .= exp.(log.(Eta_pl).*relax.+log.(Eta_iter).*(1-relax))
        		Eta        .= 2.0 ./( 1.0./Eta_m .+ 1.0./Eta_pl )
        	end
        	Res_Vx     .= -diff(Ptot[:,2:end-1], dims=1)./dx .+ diff(τ_xx[:,2:end-1], dims=1)./dx .+ diff(τ_xy, dims=2)./dy  # HORIZONTAL FORCE BALANCE
        	Res_Vy     .= -diff(Ptot[2:end-1,:], dims=2)./dy .+ diff(τ_yy[2:end-1,:], dims=2)./dy .+ diff(τ_xy, dims=1)./dx  # VERTICAL   FORCE BALANCE
        	Vx[2:end-1,2:end-1] .= Vx[2:end-1,2:end-1] .+ dt_Stokes.*Res_Vx   # Pseudo-transient form of horizontal force balance
        	Vy[2:end-1,2:end-1] .= Vy[2:end-1,2:end-1] .+ dt_Stokes.*Res_Vy   # Pseudo-transient form of vertical force balance
        	if mod(it_tstep, nout)==0 && it_tstep>250
        		err_Mx   = maximum(abs.(dt_Stokes.*Res_Vx)/maximum(abs.(Vx)))    # Error horizontal velocitiy
        		err_My   = maximum(abs.(dt_Stokes.*Res_Vy)/maximum(abs.(Vy)))    # Error vertical velocity
        		err_Pf   = maximum(abs.(dt_Pf.*Res_Pf)/maximum(abs.(Pf)))        # Error fluid pressure
        		err_Phi  = maximum(abs.(dtp.*Res_Phi))                           # Error porosity
        		err_M    = maximum([err_Pf, err_Mx, err_My, err_Phi])            # Error total
        		err_evo1 = push!(err_evo1, it_tstep); err_evo2 = push!(err_evo2, err_M)
        		# plot evol
        		p1 = plot(err_evo1, err_evo2, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
        		display(p1)
        	end
        end # end PT loop
    end
    # Visu
    swell2!(Vx_f, q_f_X./(Rho_f[1:end-1,:]./Phi[1:end-1,:].+Rho_f[2:end,:]./Phi[2:end,:]).*2.0, 1)
    swell2!(Vy_f, q_f_Y./(Rho_f[:,1:end-1]./Phi[:,1:end-1].+Rho_f[:,2:end]./Phi[:,2:end]).*2.0, 2)
    swell2!(Vx_f_Ptot, q_f_X_Ptot./(Rho_f[1:end-1,:]./Phi[1:end-1,:].+Rho_f[2:end,:]./Phi[2:end,:]).*2.0, 1)
    swell2!(VY_f_Ptot, q_f_Y_Ptot./(Rho_f[:,1:end-1]./Phi[:,1:end-1].+Rho_f[:,2:end]./Phi[:,2:end]).*2.0, 2)

    Length_model_m = rad
    Time_model_sec = rad^2/(k_ηf/β_eff)
    Length_phys_m  = 0.01
    Time_phys_sec  = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8))
    Vel_phys_m_s   = (Length_phys_m/Length_model_m) / (Time_phys_sec/Time_model_sec)
    lw = 1.2
    p2  = heatmap(xc, yc, Pf'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="A) p_f [kbar]")
    	    plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p3  = heatmap(xc, yc, Ptot'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="B) p [kbar]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p4  = heatmap(xc, yc, ∇V'.*Time_model_sec./Time_phys_sec, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="C) ∇(v_s) [1/s]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p5  = heatmap(xc, yc, sqrt.(Vx_f.^2 .+ Vy_f.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="D) ||v_f|| [m/s]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p6  = heatmap(xc, yc, sqrt.(av_xa(Vx).^2 .+ av_ya(Vy).^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="E) ||v_s|| [m/s]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p7  = heatmap(xc, yc, sqrt.(Vx_f_Ptot.^2 .+ VY_f_Ptot.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="F) ||v_f||p [m/s]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p8  = heatmap(xv[2:end-1], yv[2:end-1], τ_xy'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xv[2], xv[end-1]), ylims=(yv[2], yv[end-1]), c=:viridis, title="G) τxy [MPa]")
    p9  = heatmap(xc, yc, τII'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="H) τII [MPa]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    p10 = heatmap(xc, yc, Eta'*1e20, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="I) ηs [Pas]")
            plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
    display(plot(p2, p3, p4, p5, p6, p7, p8, p9, p10))
    return
end

@time PT_HMC()
