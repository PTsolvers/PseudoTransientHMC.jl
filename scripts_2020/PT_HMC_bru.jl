const run_test = haskey(ENV, "RUN_TEST") ? parse(Bool, ENV["RUN_TEST"]) : false
const USE_GPU  = haskey(ENV, "USE_GPU" ) ? parse(Bool, ENV["USE_GPU"] ) : true
const GPU_ID   = haskey(ENV, "GPU_ID"  ) ? parse(Int,  ENV["GPU_ID"]  ) : 0
const do_viz   = haskey(ENV, "DO_VIZ"  ) ? parse(Bool, ENV["DO_VIZ"]  ) : true
const nx       = haskey(ENV, "NX"      ) ? parse(Int , ENV["NX"]      ) : 383
const ny       = haskey(ENV, "NY"      ) ? parse(Int , ENV["NY"]      ) : 383
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT, Interpolations
##################################################
@views av_xy(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])
@views function swell2s!(B, A, ndim)
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

@parallel_indices (ix,iy) function swell2!(B::Data.Array, A::Data.Array, ndim::Int)
    if (ix<=size(B,1) && iy<=size(B,2)) B[ix,iy] = 0.0  end
    if ndim==1
        if (2<=ix<=size(B,1)-1 && iy<=size(B,2))  B[ix,iy] = 0.5*(A[ix  ,iy] +     A[ix-1,iy]) end
        if (   ix==1           && iy<=size(B,2))  B[ix,iy] =  1.5*A[ix  ,iy] - 0.5*A[ix+1,iy]  end
        if (   ix==size(B,1)   && iy<=size(B,2))  B[ix,iy] =  1.5*A[ix-1,iy] - 0.5*A[ix-2,iy]  end
    elseif ndim==2
        if (ix<=size(B,1) && 2<=iy<=size(B,2)-1)  B[ix,iy] = 0.5*(A[ix,iy  ] +     A[ix,iy-1]) end
        if (ix<=size(B,1) &&    iy==1          )  B[ix,iy] =  1.5*A[ix,iy  ] - 0.5*A[ix,iy+1]  end
        if (ix<=size(B,1) &&    iy==size(B,2)  )  B[ix,iy] =  1.5*A[ix,iy-1] - 0.5*A[ix,iy-2]  end
    end
    return 
end

@parallel function cum_mult2!(A1::Data.Array, A2::Data.Array, B1::Data.Array, B2::Data.Array)
    @all(A1) = @all(A1)*@all(B1)
    @all(A2) = @all(A2)*@all(B2)
    return
end

@parallel function laplace!(A::Data.Array, TmpX::Data.Array, TmpY::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(A) = @d_xa(TmpX)*_dx + @d_ya(TmpY)*_dy
    return
end
##################################################

@parallel function compute_1!(Rho_s_old::Data.Array, Rho_f_old::Data.Array, X_s_old::Data.Array, Rho_X_old::Data.Array, Phi_old::Data.Array, Ptot_old::Data.Array, Rho_s::Data.Array, Rho_f::Data.Array, X_s::Data.Array, Phi::Data.Array, Ptot::Data.Array)
    @all(Rho_s_old) = @all(Rho_s)
    @all(Rho_f_old) = @all(Rho_f)
    @all(X_s_old)   = @all(X_s)
    @all(Phi_old)   = @all(Phi)
    @all(Ptot_old)  = @all(Ptot)
    @all(Rho_X_old) = @all(Rho_s_old)*@all(X_s_old)
    return
end

@parallel function compute_2!(Rho_X_ini::Data.Array, Phi_old::Data.Array, Phi_ini::Data.Array, Phi::Data.Array, Rho_s_ini::Data.Array, X_s_ini::Data.Array, Rho_X_old::Data.Array, ϕ_ini::Data.Number)
    @all(Rho_X_ini) = @all(Rho_s_ini)*@all(X_s_ini)
    @all(Phi_old)   = 1.0 - @all(Rho_X_ini)*(1.0-ϕ_ini)/@all(Rho_X_old)
    @all(Phi_ini)   = @all(Phi_old)
    @all(Phi)       = @all(Phi_old)
    return
end

@parallel_indices (ix,iy) function compute_3!(Eta_m::Data.Array, Lam::Data.Array, Phi::Data.Array, η_m::Data.Number, η_i_fac::Data.Number, λ::Data.Number, λ_i_fac::Data.Number, max_min_ϕ_2::Data.Number)
    if (ix<=size(Eta_m,1) && iy<=size(Eta_m,2))
        if (Phi[ix,iy]>max_min_ϕ_2) Eta_m[ix,iy] = η_m*η_i_fac; else Eta_m[ix,iy] = η_m; end
    end
    if (ix<=size(Lam,1) && iy<=size(Lam,2))
        if (Phi[ix,iy]>max_min_ϕ_2) Lam[ix,iy] = λ*λ_i_fac; else Lam[ix,iy] = λ; end
    end
    return
end

@parallel function compute_4!(Eta_pl::Data.Array, Eta::Data.Array, Rho_t_old::Data.Array, Eta_m::Data.Array, Rho_f_old::Data.Array, Phi_old::Data.Array, Rho_s_old::Data.Array)
    @all(Eta_pl)    = @all(Eta_m)
    @all(Eta)       = @all(Eta_m)
    @all(Rho_t_old) = @all(Rho_f_old)*@all(Phi_old) + @all(Rho_s_old)*(1.0 - @all(Phi_old))
    return
end

@parallel_indices (ix,iy) function compute_5!(Rho_t::Data.Array, para_cx::Data.Array, para_cy::Data.Array, Rho_f::Data.Array, Phi::Data.Array, Rho_s::Data.Array, K_ηf::Data.Array)
    if (ix<=size(Rho_t,1)   && iy<=size(Rho_t,2))   Rho_t[ix,iy]   = Rho_f[ix,iy]*Phi[ix,iy] + Rho_s[ix,iy]*(1.0-Phi[ix,iy]) end
    if (ix<=size(para_cx,1) && iy<=size(para_cx,2)) para_cx[ix,iy] = 0.5*( Rho_f[ix,iy]*K_ηf[ix,iy]*Phi[ix,iy]^3 + Rho_f[ix+1,iy]*K_ηf[ix+1,iy]*Phi[ix+1,iy]^3 ) end
    if (ix<=size(para_cy,1) && iy<=size(para_cy,2)) para_cy[ix,iy] = 0.5*( Rho_f[ix,iy]*K_ηf[ix,iy]*Phi[ix,iy]^3 + Rho_f[ix,iy+1]*K_ηf[ix,iy+1]*Phi[ix,iy+1]^3 ) end
    return
end

@parallel function compute_6!(q_f_X::Data.Array, q_f_Y::Data.Array, q_f_X_Ptot::Data.Array, q_f_Y_Ptot::Data.Array, para_cx::Data.Array, para_cy::Data.Array, Pf::Data.Array, Ptot::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(q_f_X)      = -@all(para_cx)*@d_xa(Pf)*_dx        # Correct   Darcy flux with fluid pressure
    @all(q_f_Y)      = -@all(para_cy)*@d_ya(Pf)*_dy        # Correct   Darcy flux with fluid pressure
    @all(q_f_X_Ptot) = -@all(para_cx)*@d_xa(Ptot)*_dx      # Incorrect Darcy flux with total pressure
    @all(q_f_Y_Ptot) = -@all(para_cy)*@d_ya(Ptot)*_dy      # Incorrect Darcy flux with total pressure
    return
end

@parallel function compute_7!(∇q_f::Data.Array, Res_Pf::Data.Array, q_f_X::Data.Array, q_f_Y::Data.Array, Rho_t::Data.Array, Rho_t_old::Data.Array, ∇V_ρ_t::Data.Array, dtp::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @all(∇q_f)   = @d_xi(q_f_X)*_dx + @d_yi(q_f_Y)*_dy
    @all(Res_Pf) = -@all(∇q_f) -(@inn(Rho_t) - @inn(Rho_t_old))/dtp - @inn(∇V_ρ_t) # CONSERVATION OF TOTAL MASS EQUATION
    return
end

@parallel function compute_8!(Pf::Data.Array, Rho_X_ϕ::Data.Array, Res_Phi::Data.Array, Phi::Data.Array, Res_Pf::Data.Array, Rho_s::Data.Array, X_s::Data.Array, Phi_old::Data.Array, Rho_X_old::Data.Array, ∇V_ρ_x::Data.Array, dt_Pf::Data.Number, dtp::Data.Number, _dx::Data.Number, _dy::Data.Number)
    @inn(Pf)      = @inn(Pf) + dt_Pf*@all(Res_Pf)
    @all(Rho_X_ϕ) = (1.0-@all(Phi))*@all(Rho_s)*@all(X_s)
    @all(Res_Phi) = ( @all(Rho_X_ϕ) - (1.0-@all(Phi_old))*@all(Rho_X_old) )/dtp + @all(∇V_ρ_x)   # CONSERVATION OF MASS OF MgO EQUATION
    @all(Phi)     = @all(Phi) + dtp*@all(Res_Phi)
    return
end


@parallel function compute_9!(∇V::Data.Array, ε_xx::Data.Array, ε_yy::Data.Array, ε_xy::Data.Array, Ptot::Data.Array, Vx::Data.Array, Vy::Data.Array, Phi::Data.Array, Pf::Data.Array, Lam::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(∇V)   = _dx*@d_xa(Vx) + _dy*@d_ya(Vy)
    @all(ε_xx) = _dx*@d_xa(Vx) - 1/3*@all(∇V)
    @all(ε_yy) = _dy*@d_ya(Vy) - 1/3*@all(∇V)
    @all(ε_xy) = 0.5*(_dy*@d_yi(Vx) + _dx*@d_xi(Vy))
    @all(Ptot) = @all(Pf) - @all(∇V)*(1.0-@all(Phi))*@all(Lam)
    return 
end

@parallel function compute_10!(τ_xx::Data.Array, τ_yy::Data.Array, τ_xy::Data.Array, Eta::Data.Array, ε_xx::Data.Array, ε_yy::Data.Array, ε_xy::Data.Array)
    @all(τ_xx)  = 2.0*@all(Eta)  * @all(ε_xx)    
    @all(τ_yy)  = 2.0*@all(Eta)  * @all(ε_yy) 
    @all(τ_xy)  = 2.0*@av(Eta)   * @all(ε_xy)
    return 
end

@parallel function compute_11!(τII::Data.Array, Res_Vx::Data.Array, Res_Vy::Data.Array, τ_xx::Data.Array, τ_yy::Data.Array, τ_xyn::Data.Array, Ptot::Data.Array, τ_xy::Data.Array, _dx::Data.Number, _dy::Data.Number)
    @all(τII)    = sqrt( 0.25*(@all(τ_xx)-@all(τ_yy))^2 + @all(τ_xyn)^2 )
    @all(Res_Vx) = -@d_xi(Ptot)*_dx + @d_xi(τ_xx)*_dx + @d_ya(τ_xy)*_dy  # HORIZONTAL FORCE BALANCE
    @all(Res_Vy) = -@d_yi(Ptot)*_dy + @d_yi(τ_yy)*_dy + @d_xa(τ_xy)*_dx  # VERTICAL   FORCE BALANCE
    return 
end

@parallel_indices (ix,iy) function compute_12!(Eta_iter::Data.Array, Eta_pl::Data.Array, Eta::Data.Array, Eta_m::Data.Array, τII::Data.Array, σ_ref::Data.Number, n_exp::Data.Number, relax::Data.Number)
    if (ix<=size(Eta_iter,1) && iy<=size(Eta_iter,2))  Eta_iter[ix,iy] = Eta_pl[ix,iy] end                                    # Previous PT viscosity
    if (ix<=size(Eta_pl,1)   && iy<=size(Eta_pl,2))    Eta_pl[ix,iy]   = Eta_m[ix,iy]*(τII[ix,iy]/σ_ref)^(1-n_exp) end
    if (ix<=size(Eta_pl,1)   && iy<=size(Eta_pl,2))    if (τII[ix,iy]<σ_ref) Eta_pl[ix,iy] = Eta_m[ix,iy]; end; end #η_m
    if (ix<=size(Eta_pl,1)   && iy<=size(Eta_pl,2))    Eta_pl[ix,iy]    = exp(log(Eta_pl[ix,iy])*relax + log(Eta_iter[ix,iy])*(1-relax)) end
    if (ix<=size(Eta,1)      && iy<=size(Eta,2))       Eta[ix,iy]       = 2.0/( 1.0/Eta_m[ix,iy] + 1.0/Eta_pl[ix,iy] ) end
    return 
end

@parallel function compute_13!(Vx::Data.Array, Vy::Data.Array, Res_Vx::Data.Array, Res_Vy::Data.Array, dt_Stokes::Data.Number)
    @inn(Vx) = @inn(Vx) + dt_Stokes*@all(Res_Vx)   # Pseudo-transient form of horizontal force balance
    @inn(Vy) = @inn(Vy) + dt_Stokes*@all(Res_Vy)   # Pseudo-transient form of vertical force balance
    return 
end
##################################################
@views function PT_HMC_()
    runid           = "bru"
    # read in mat file
    vars            = matread(string(@__DIR__, "/LOOK_UP_HMC_Pub.mat"))
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
    tol             = 2e-5                             # Tolerance for pseudo-transient iterations
    cfl             = 1.0/16.1                         # CFL parameter for PT-Stokes solution
    dtp             = 2e0*rad^2/(k_ηf/β_eff)           # Time step physical
    time_tot        = 1.0*dtp                          # Total time of simulation
    relax           = 0.5
    itmax           = 5e4
    nout            = 1e3
    # Configuration of grid, matrices and numerical parameters
    dx, dy          = lx/(nx-1), ly/(ny-1)             # Grid spacing
    xc, yc          = -lx/2:dx:lx/2, -ly/2:dy:ly/2     # Coordinate vector
    xv, yv          = -lx/2-dx/2:dx:lx/2+dx/2, -ly/2-dy/2:dy:ly/2+dy/2  # Horizontal vector for Vx which is one more than basic grid
    (Xc2, Yc2)      = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    (Xc2vx, Yc2vx)  = ([x for x=xv, y=yc], [y for x=xv, y=yc])
    (Xc2vy, Yc2vy)  = ([x for x=xc, y=yv], [y for x=xc, y=yv])
    rad_a           = rad
    rad_b           = rad_a*elli_fac
    X_rot           =  Xc2*cosd(α)+Yc2*sind(α)
    Y_rot           = -Xc2*sind(α)+Yc2*cosd(α)
    X_elli          =  rad_a.*cos.(0:0.01:2*pi).*cosd(α).+rad_b.*sin.(0:0.01:2*pi).*sind(α)
    Y_elli          = -rad_a.*cos.(0:0.01:2*pi).*sind(α).+rad_b.*sin.(0:0.01:2*pi).*cosd(α)
    XY_elli         = (-X_elli, Y_elli)
    # Initial ambient fluid pressure
    Pf              = P_ini*ones(nx,  ny)           
    Pf[sqrt.(X_rot.^2.0./rad_a.^2.0 .+ Y_rot.^2.0./rad_b.^2) .< 1.0] .= P_ini - P_pert  # Fluid pressure petubation
    for smo=1:5 # Smooting of perturbation
        Pf[2:end-1,:] .= Pf[2:end-1,:] .+ 0.4.*(Pf[3:end,:].-2.0.*Pf[2:end-1,:].+Pf[1:end-2,:])
        Pf[:,2:end-1] .= Pf[:,2:end-1] .+ 0.4.*(Pf[:,3:end].-2.0.*Pf[:,2:end-1].+Pf[:,1:end-2])
    end
    # Density, compressibility and gamma from concentration and pressure from thermodynamic data base
    itp1            = interpolate( (P_LU[:,1],), Rho_s_LU[:,1], Gridded(Linear()))
    itp2            = interpolate( (P_LU[:,1],), Rho_f_LU[:,1], Gridded(Linear()))
    itp3            = interpolate( (P_LU[:,1],),     X_LU[:,1], Gridded(Linear()))
    Rho_s           = Data.Array( itp1.(Pf)./ρ_0 )
    Rho_f           = Data.Array( itp2.(Pf)./ρ_0 )
    X_s             = Data.Array( itp3.(Pf) )
    Rho_s_ini       = Data.Array( itp1.(P_ini*ones(nx, ny))./ρ_0 )
    X_s_ini         = Data.Array( itp3.(P_ini*ones(nx, ny)) )
    # Initialize ALL arrays in Julia
    Ptot            = @zeros(nx  , ny  )               # Initial ambient fluid pressure
    ∇V              = @zeros(nx  , ny  )
    ∇V_ρ_x          = @zeros(nx  , ny  )
    ∇V_ρ_t          = @zeros(nx  , ny  )
    τ_xx            = @zeros(nx  , ny  )               # Deviatoric stress
    τ_yy            = @zeros(nx  , ny  )               # Deviatoric stress
    τ_xy            = @zeros(nx-1, ny-1)               # Deviatoric stress
    τ_xyn           = @zeros(nx  , ny  )
    Res_Vx          = @zeros(nx-1, ny-2)
    Res_Vy          = @zeros(nx-2, ny-1)
    Rho_s_old       = @zeros(nx  , ny  )
    Rho_f_old       = @zeros(nx  , ny  )
    X_s_old         = @zeros(nx  , ny  )
    Rho_X_old       = @zeros(nx  , ny  )
    Phi_old         = @zeros(nx  , ny  )
    Ptot_old        = @zeros(nx  , ny  )
    Rho_X_ini       = @zeros(nx  , ny  )
    Rho_t_old       = @zeros(nx  , ny  )
    Rho_t           = @zeros(nx  , ny  )
    para_cx         = @zeros(nx-1, ny  )
    para_cy         = @zeros(nx  , ny-1)
    q_f_X           = @zeros(nx-1, ny  )
    q_f_Y           = @zeros(nx  , ny-1)
    q_f_X_Ptot      = @zeros(nx-1, ny  )
    q_f_Y_Ptot      = @zeros(nx  , ny-1)
    ∇q_f            = @zeros(nx-2, ny-2)
    Res_Pf          = @zeros(nx-2, ny-2)
    Rho_X_ϕ         = @zeros(nx  , ny  )
    Res_Phi         = @zeros(nx  , ny  )
    ε_xx            = @zeros(nx  , ny  )
    ε_yy            = @zeros(nx  , ny  )
    ε_xy            = @zeros(nx-1, ny-1)
    τII             = @zeros(nx  , ny  )
    Eta_iter        = @zeros(nx  , ny  )
    # TMP arrays for swell2
    TmpX            = @zeros(nx+1, ny  )
    TmpY            = @zeros(nx  , ny+1)
    TmpS1           = @zeros(nx  , ny-1)
    # arrays for visu
    Vx_f            = zeros(nx  , ny  )
    Vy_f            = zeros(nx  , ny  )
    Vx_f_Ptot       = zeros(nx  , ny  )
    VY_f_Ptot       = zeros(nx  , ny  )
    # init
    Phi_ini         = ϕ_ini*@ones(nx, ny)
    Phi             = ϕ_ini*@ones(nx, ny)
    K_ηf            =  k_ηf*@ones(nx, ny)
    Eta_m           =   η_m*@ones(nx, ny)
    Eta_pl          =   η_m*@ones(nx, ny)
    Eta             =   η_m*@ones(nx, ny)
    Lam             =     λ*@ones(nx, ny)         # Viscosity
    Pf_tmp          = zeros(nx, ny)
    # Data.Array for ParallelStencil
    Vx              = -ε_bg*Data.Array(Xc2vx)     # Pure shear, shortening in x
    Vy              =  ε_bg*Data.Array(Yc2vy)     # Pure shear, extension in y
    Pf              = Data.Array(Pf)
    Ptot           .= Pf                          # Initial total pressure
    # Parameters for time loop and pseudo-transient iterations
    max_dxdy2       = max(dx,dy).^2
    max_min_ϕ_2     = (maximum(Phi)+minimum(Phi))/2.0
    _dx, _dy        = 1.0/dx, 1.0/dy
    timeP           = 0.0                         # Initial time
    it              = 0                           # Integer count for iteration loop
    itp             = 0                           # Integer count for time loop
    save_count      = 0 
    Time_vec        = []
    if do_viz
        !ispath(joinpath(@__DIR__,"../output")) && mkdir(joinpath(@__DIR__,"../output"))
        dirname = joinpath(@__DIR__, "../output/output_$(runid)_$(nx)x$(ny)"); !ispath(dirname) && mkdir(dirname)
    end
    # time loop
    while timeP < time_tot
    	err_M=2*tol; itp+=1.0
        timeP=timeP+dtp; push!(Time_vec, timeP)
        @parallel compute_1!(Rho_s_old, Rho_f_old, X_s_old, Rho_X_old, Phi_old, Ptot_old, Rho_s, Rho_f, X_s, Phi, Ptot)
        if itp==1  @parallel compute_2!(Rho_X_ini, Phi_old, Phi_ini, Phi, Rho_s_ini, X_s_ini, Rho_X_old, ϕ_ini) end
    	max_min_ϕ_2 = (maximum(Phi)+minimum(Phi))/2.0
    	@parallel compute_3!(Eta_m, Lam, Phi, η_m, η_i_fac, λ, λ_i_fac, max_min_ϕ_2)
        @parallel compute_4!(Eta_pl, Eta, Rho_t_old, Eta_m, Rho_f_old, Phi_old, Rho_s_old)
    	# PT loop
    	it_tstep=0; err_evo1=[]; err_evo2=[]
        dt_Stokes, dt_Pf = cfl*max_dxdy2, cfl*max_dxdy2
    	while err_M>tol && it_tstep<itmax
            it+=1; it_tstep+=1
            if mod(it_tstep, 100)==0 || it_tstep==1
                dt_Stokes = cfl*max_dxdy2/maximum(Eta)                       # Pseudo time step for Stokes
                dt_Pf     = cfl*max_dxdy2/maximum(K_ηf.*Phi.^3/β_eff)  # Pseudo time step for fluid pressure
            end
            # Fluid pressure evolution
            @parallel compute_5!(Rho_t, para_cx, para_cy, Rho_f, Phi, Rho_s, K_ηf)
            @parallel compute_6!(q_f_X, q_f_Y, q_f_X_Ptot, q_f_Y_Ptot, para_cx, para_cy, Pf, Ptot, _dx, _dy)
            @parallel compute_7!(∇q_f, Res_Pf, q_f_X, q_f_Y, Rho_t, Rho_t_old, ∇V_ρ_t, dtp, _dx, _dy)
            if mod(it_tstep, 50)==0 || it_tstep==1 # Look up for densities
                Pf_tmp .= Array( Pf )
                Rho_s  .= Data.Array( itp1.(Pf_tmp)./ρ_0 )
                Rho_f  .= Data.Array( itp2.(Pf_tmp)./ρ_0 )
                X_s    .= Data.Array( itp3.(Pf_tmp)      )
            end
            # Porosity evolution
            @parallel compute_8!(Pf, Rho_X_ϕ, Res_Phi, Phi, Res_Pf, Rho_s, X_s, Phi_old, Rho_X_old, ∇V_ρ_x, dt_Pf, dtp, _dx, _dy)
            # Stokes
            @parallel swell2!(TmpX, Rho_X_ϕ, 1)
            @parallel swell2!(TmpY, Rho_X_ϕ, 2)
            @parallel cum_mult2!(TmpX, TmpY, Vx, Vy)
            @parallel laplace!(∇V_ρ_x, TmpX, TmpY, _dx, _dy)
            @parallel swell2!(TmpX, Rho_t, 1)
            @parallel swell2!(TmpY, Rho_t, 2)
            @parallel cum_mult2!(TmpX, TmpY, Vx, Vy)
            @parallel laplace!(∇V_ρ_t, TmpX, TmpY, _dx, _dy)
            @parallel compute_9!(∇V, ε_xx, ε_yy, ε_xy, Ptot, Vx, Vy, Phi, Pf, Lam, _dx, _dy)
            @parallel compute_10!(τ_xx, τ_yy, τ_xy, Eta, ε_xx, ε_yy, ε_xy)
            @parallel swell2!(TmpS1, τ_xy,  1)
            @parallel swell2!(τ_xyn, TmpS1, 2)
            @parallel compute_11!(τII, Res_Vx, Res_Vy, τ_xx, τ_yy, τ_xyn, Ptot, τ_xy, _dx, _dy)
            # power-law
            if n_exp>1 @parallel compute_12!(Eta_iter, Eta_pl, Eta, Eta_m, τII, σ_ref, n_exp, relax) end
            @parallel compute_13!(Vx, Vy, Res_Vx, Res_Vy, dt_Stokes)
            if mod(it_tstep, nout)==0 && it_tstep>250
                err_Mx   = dt_Stokes*maximum(abs.(Res_Vx)/maximum(abs.(Vx)))    # Error horizontal velocitiy
                err_My   = dt_Stokes*maximum(abs.(Res_Vy)/maximum(abs.(Vy)))    # Error vertical velocity
                err_Pf   = dt_Pf*maximum(abs.(Res_Pf)/maximum(abs.(Pf)))        # Error fluid pressure
                err_Phi  = dtp*maximum(abs.(Res_Phi))                           # Error porosity
                err_M    = maximum([err_Pf, err_Mx, err_My, err_Phi])           # Error total
                err_evo1 = push!(err_evo1, it_tstep); err_evo2 = push!(err_evo2, err_M)
                # plot evol
                # p1 = plot(err_evo1, err_evo2, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
                # display(p1)
                @printf("iter = %d, error = %1.3e \n", it_tstep, err_M)
            end
        end # end PT loop
    end
    # Visu
    if do_viz
        println("iter tot = $it")
        swell2s!(Vx_f, Array(q_f_X)./(Array(Rho_f[1:end-1,:])./Array(Phi[1:end-1,:]).+Array(Rho_f[2:end,:])./Array(Phi[2:end,:])).*2.0, 1)
        swell2s!(Vy_f, Array(q_f_Y)./(Array(Rho_f[:,1:end-1])./Array(Phi[:,1:end-1]).+Array(Rho_f[:,2:end])./Array(Phi[:,2:end])).*2.0, 2)
        swell2s!(Vx_f_Ptot, Array(q_f_X_Ptot)./(Array(Rho_f[1:end-1,:])./Array(Phi[1:end-1,:]).+Array(Rho_f[2:end,:])./Array(Phi[2:end,:])).*2.0, 1)
        swell2s!(VY_f_Ptot, Array(q_f_Y_Ptot)./(Array(Rho_f[:,1:end-1])./Array(Phi[:,1:end-1]).+Array(Rho_f[:,2:end])./Array(Phi[:,2:end])).*2.0, 2)
        Length_model_m = rad
        Time_model_sec = rad^2/(k_ηf/β_eff)
        Length_phys_m  = 0.01
        Time_phys_sec  = 0.01^2/(1e-19/1e-3/(1e-2/8.5e8))
        Vel_phys_m_s   = (Length_phys_m/Length_model_m) / (Time_phys_sec/Time_model_sec)
        lw = 1.2; TFS = 10
        p2  = heatmap(xc, yc, Array(Pf)'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="A) p_f [kbar]", titlefontsize=TFS)
        	    plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p3  = heatmap(xc, yc, Array(Ptot)'./Pini_Pappl./1e8, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="B) p [kbar]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p4  = heatmap(xc, yc, Array(∇V)'.*Time_model_sec./Time_phys_sec, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="C) ∇(v_s) [1/s]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p5  = heatmap(xc, yc, sqrt.(Vx_f.^2 .+ Vy_f.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="D) ||v_f|| [m/s]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p6  = heatmap(xc, yc, sqrt.(av_xa(Array(Vx)).^2 .+ av_ya(Array(Vy)).^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="E) ||v_s|| [m/s]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p7  = heatmap(xc, yc, sqrt.(Vx_f_Ptot.^2 .+ VY_f_Ptot.^2)'*Vel_phys_m_s, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="F) ||v_f||p [m/s]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p8  = heatmap(xv[2:end-1], yv[2:end-1], Array(τ_xy)'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xv[2], xv[end-1]), ylims=(yv[2], yv[end-1]), c=:hot, title="G) τxy [MPa]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p9  = heatmap(xc, yc, Array(τII)'/Pini_Pappl/1e6, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="H) τII [MPa]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        p10 = heatmap(xc, yc, Array(Eta)'*1e20, aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:hot, title="I) ηs [Pas]", titlefontsize=TFS)
                plot!(XY_elli[1], XY_elli[2], linewidth=lw, linecolor="white", legend=false)
        display(plot(p2, p3, p4, p5, p6, p7, p8, p9, p10, background_color=:transparent, foreground_color=:gray))
        savefig(joinpath(@__DIR__, dirname, "fig_pt_hmc_bru.png"))
    end
    return xc, yc, Pf, Phi
end

if run_test
    xc, yc, Pf, Phi = PT_HMC_();
else
    PT_HMC = begin PT_HMC_(); return; end
end
