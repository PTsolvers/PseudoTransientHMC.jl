const use_return = haskey(ENV, "USE_RETURN") ? parse(Bool, ENV["USE_RETURN"]) : false
const USE_GPU    = haskey(ENV, "USE_GPU"   ) ? parse(Bool, ENV["USE_GPU"]   ) : false
const GPU_ID     = haskey(ENV, "GPU_ID"    ) ? parse(Int,  ENV["GPU_ID"]    ) : 0
const do_viz     = haskey(ENV, "DO_VIZ"    ) ? parse(Bool, ENV["DO_VIZ"]    ) : false
const do_save    = haskey(ENV, "DO_SAVE"   ) ? parse(Bool, ENV["DO_SAVE"]   ) : false
const nx         = haskey(ENV, "NX"        ) ? parse(Int , ENV["NX"]        ) : 159 #959
const ny         = haskey(ENV, "NY"        ) ? parse(Int , ENV["NY"]        ) : 159 #959
###
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra, MAT

import ParallelStencil: INDICES
ix,iy   = INDICES[1], INDICES[2]
ixi,iyi = :($ix+1), :($iy+1)

"Average in x and y dimension"
@views    av(A) = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end])
"Average in x dimension"
@views av_xa(A) =  0.5*(A[1:end-1,:].+A[2:end,:])
"Average in y dimension"
@views av_ya(A) =  0.5*(A[:,1:end-1].+A[:,2:end])

@views function swell2h!(B, A, ndim)
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

@parallel_indices (ix,iy) function swell2_x!(B::Data.Array, A::Data.Array)
    if (ix<=size(B,1) && iy<=size(B,2)) B[ix,iy] = 0.0  end
    if (2<=ix<=size(B,1)-1 && iy<=size(B,2))  B[ix,iy] = 0.5*(A[ix  ,iy] +     A[ix-1,iy]) end
    if (   ix==1           && iy<=size(B,2))  B[ix,iy] =  1.5*A[ix  ,iy] - 0.5*A[ix+1,iy]  end
    if (   ix==size(B,1)   && iy<=size(B,2))  B[ix,iy] =  1.5*A[ix-1,iy] - 0.5*A[ix-2,iy]  end
    return
end

@parallel_indices (ix,iy) function swell2_y!(B::Data.Array, A::Data.Array)
    if (ix<=size(B,1) && iy<=size(B,2)) B[ix,iy] = 0.0  end
    if (ix<=size(B,1) && 2<=iy<=size(B,2)-1)  B[ix,iy] = 0.5*(A[ix,iy  ] +     A[ix,iy-1]) end
    if (ix<=size(B,1) &&    iy==1          )  B[ix,iy] =  1.5*A[ix,iy  ] - 0.5*A[ix,iy+1]  end
    if (ix<=size(B,1) &&    iy==size(B,2)  )  B[ix,iy] =  1.5*A[ix,iy-1] - 0.5*A[ix,iy-2]  end
    return
end

@parallel function cum_mult2!(A1::Data.Array, A2::Data.Array, B1::Data.Array, B2::Data.Array)
    @all(A1) = @all(A1)*@all(B1)
    @all(A2) = @all(A2)*@all(B2)
    return
end

@parallel function laplace!(A::Data.Array, TmpX::Data.Array, TmpY::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(A) = @d_xa(TmpX)/dx + @d_ya(TmpY)/dy
    return
end

@parallel_indices (iy) function bc_x!(A::Data.Array)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A::Data.Array)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

@parallel function compute_1!(Rho_s_old::Data.Array, Rho_f_old::Data.Array, X_s_old::Data.Array, Rho_X_old::Data.Array, Phi_old::Data.Array, Ptot_old::Data.Array, Pf_old::Data.Array, Rho_t_old::Data.Array, Eta::Data.Array, Lam::Data.Array, Rho_s::Data.Array, Rho_f::Data.Array, X_s::Data.Array, Phi::Data.Array, Ptot::Data.Array, Pf::Data.Array, η_m::Data.Number, ϕ_exp::Data.Number, ϕ_ini::Data.Number, λ_η::Data.Number)
    @all(Rho_s_old) = @all(Rho_s)
    @all(Rho_f_old) = @all(Rho_f)
    @all(X_s_old)   = @all(X_s)
    @all(Phi_old)   = @all(Phi)
    @all(Ptot_old)  = @all(Ptot)
    @all(Pf_old)    = @all(Pf)
    @all(Rho_X_old) = @all(Rho_s_old)*@all(X_s_old)
    @all(Rho_t_old) = @all(Rho_f_old)*@all(Phi_old) + @all(Rho_s_old)*(1.0 - @all(Phi_old))
    @all(Eta)       = η_m*exp(-ϕ_exp*(@all(Phi)-ϕ_ini))
    @all(Lam)       = λ_η*@all(Eta)
    return
end

@parallel_indices (ix,iy) function compute_2!(Rho_t::Data.Array, para_cx::Data.Array, para_cy::Data.Array, Rho_f::Data.Array, Phi::Data.Array, Rho_s::Data.Array, k_ηf::Data.Number)
    if (ix<=size(Rho_t,1)   && iy<=size(Rho_t,2))   Rho_t[ix,iy]   = Rho_f[ix,iy]*Phi[ix,iy] + Rho_s[ix,iy]*(1.0-Phi[ix,iy]) end
    if (ix<=size(para_cx,1) && iy<=size(para_cx,2)) para_cx[ix,iy] = 0.5*( Rho_f[ix,iy]*k_ηf*Phi[ix,iy]^3 + Rho_f[ix+1,iy]*k_ηf*Phi[ix+1,iy]^3 ) end
    if (ix<=size(para_cy,1) && iy<=size(para_cy,2)) para_cy[ix,iy] = 0.5*( Rho_f[ix,iy]*k_ηf*Phi[ix,iy]^3 + Rho_f[ix,iy+1]*k_ηf*Phi[ix,iy+1]^3 ) end
    return
end

@parallel function compute_3!(q_f_X::Data.Array, q_f_Y::Data.Array, para_cx::Data.Array, para_cy::Data.Array, Pf::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(q_f_X) = -@all(para_cx)*@d_xa(Pf)/dx        # Correct   Darcy flux with fluid pressure
    @all(q_f_Y) = -@all(para_cy)*@d_ya(Pf)/dy        # Correct   Darcy flux with fluid pressure
    return
end

@parallel function compute_4!(∇q_f::Data.Array, Res_Pf::Data.Array, Rho_f::Data.Array, Rho_s_eq::Data.Array, X_s_eq::Data.Array, q_f_X::Data.Array, q_f_Y::Data.Array, Rho_t::Data.Array, Rho_t_old::Data.Array, ∇V_ρ_t::Data.Array, Pf::Data.Array, SlopeA::Data.Array, 
                              rho_f_maxA::Data.Number, ρ_0::Data.Number, p_reactA::Data.Number, rho_s_difA::Data.Number, rho_s_up::Data.Number, rho_s_minA::Data.Number, x_difA::Data.Number, x_minA::Data.Number, dtp::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(∇q_f)     = @d_xi(q_f_X)/dx + @d_yi(q_f_Y)/dy
    @all(Res_Pf)   = -@all(∇q_f) - (@inn(Rho_t) - @inn(Rho_t_old))/dtp - @inn(∇V_ρ_t) # CONSERVATION OF TOTAL MASS EQUATION
    @all(Rho_f)    = (rho_f_maxA * log(@all(Pf) + 1.0)^(1.0/3.5))/ρ_0
    @all(Rho_s_eq) = (-tanh( 6e2*(@all(Pf)-p_reactA) )*(rho_s_difA/2.0 + rho_s_up/3.0) + (rho_s_difA/2.0 - rho_s_up/3.0) + rho_s_minA + @all(SlopeA))/ρ_0
    @all(X_s_eq)   =  -tanh( 6e2*(@all(Pf)-p_reactA) )*x_difA/2.0 + x_difA/2.0 + x_minA
    return
end

@parallel function compute_5!(X_s::Data.Array, Rho_s::Data.Array, X_s_old::Data.Array, X_s_eq::Data.Array, Rho_s_old::Data.Array, Rho_s_eq::Data.Array, dtp::Data.Number, kin_time::Data.Number)
    @all(X_s)   = @all(X_s_old)   + dtp*(@all(X_s_eq)   - @all(X_s)  )/kin_time
    @all(Rho_s) = @all(Rho_s_old) + dtp*(@all(Rho_s_eq) - @all(Rho_s))/kin_time
    return
end

@parallel function compute_6!(dPfdτ::Data.Array, Pf::Data.Array, Rho_X_ϕ::Data.Array, Res_Phi::Data.Array, Phi::Data.Array, Res_Pf::Data.Array, Rho_s::Data.Array, X_s::Data.Array, Phi_old::Data.Array, Rho_X_old::Data.Array, ∇V_ρ_X::Data.Array, dampPf::Data.Number, dt_Pf::Data.Number, dtp::Data.Number)
    @all(dPfdτ)   = dampPf*@all(dPfdτ) + @all(Res_Pf)
    @inn(Pf)      = @inn(Pf) + dt_Pf*@all(dPfdτ)
    @all(Rho_X_ϕ) = (1.0-@all(Phi))*@all(Rho_s)*@all(X_s)
    @all(Res_Phi) = ( @all(Rho_X_ϕ) - (1.0-@all(Phi_old))*@all(Rho_X_old) )/dtp + @all(∇V_ρ_X)   # CONSERVATION OF MASS OF MgO EQUATION
    @all(Phi)     = @all(Phi) + dtp*@all(Res_Phi)
    return
end

macro limit_min(A,min_val) esc(:( max($A[$ix,$iy],$min_val) )) end
@parallel function compute_7!(Eta::Data.Array, Lam::Data.Array, ∇V::Data.Array, ε_xx::Data.Array, ε_yy::Data.Array, ε_xy::Data.Array, Ptot::Data.Array, Vx::Data.Array, Vy::Data.Array, Phi::Data.Array, Ptot_old::Data.Array, Pf::Data.Array, Pf_old::Data.Array,
                              dtPt::Data.Number, η_m::Data.Number, ϕ_exp::Data.Number, ϕ_ini::Data.Number, λ_η::Data.Number, dtp::Data.Number, K_d::Data.Number, η_min::Data.Number, α::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(Eta)  = η_m*exp(-ϕ_exp*(@all(Phi)-ϕ_ini))
    @all(Eta)  = @limit_min(Eta,η_min)
    @all(Lam)  = λ_η*@all(Eta)
    @all(∇V)   = @d_xa(Vx)/dx + @d_ya(Vy)/dy
    @all(ε_xx) = @d_xa(Vx)/dx - 1.0/3.0*@all(∇V)
    @all(ε_yy) = @d_ya(Vy)/dy - 1.0/3.0*@all(∇V)
    @all(ε_xy) = 0.5*(@d_yi(Vx)/dy + @d_xi(Vy)/dx)
    @all(Ptot) = @all(Ptot) - dtPt*( @all(Ptot) - (@all(Ptot_old)-dtp*( K_d*@all(∇V) - α*( (@all(Pf) - @all(Pf_old))/dtp ) - K_d*@all(Pf) / ((1.0-@all(Phi))*@all(Lam)) )) / ( 1.0 + dtp*K_d/((1.0-@all(Phi))*@all(Lam)) ) )
    return
end

@parallel function compute_8!(τ_xx::Data.Array, τ_yy::Data.Array, τ_xy::Data.Array, Eta::Data.Array, ε_xx::Data.Array, ε_yy::Data.Array, ε_xy::Data.Array)
    @all(τ_xx) = 2.0*@all(Eta) * @all(ε_xx)    
    @all(τ_yy) = 2.0*@all(Eta) * @all(ε_yy) 
    @all(τ_xy) = 2.0*@av(Eta)  * @all(ε_xy)
    return
end

@parallel function compute_plast_1!(τII::Data.Array, LamP::Data.Array, τ_xx::Data.Array, τ_yy::Data.Array, τ_xyn::Data.Array, σ_y::Data.Number)
    @all(τII)  = sqrt( 0.5*(@all(τ_xx)*@all(τ_xx) + @all(τ_yy)*@all(τ_yy)) + @all(τ_xyn)*@all(τ_xyn) )
    @all(LamP) = max(0.0, (1.0 - σ_y/@all(τII)))
    return
end

@parallel function compute_plast_2!(τ_xx::Data.Array, τ_yy::Data.Array, τ_xy::Data.Array, LamP::Data.Array)
    @all(τ_xx) = (1.0-@all(LamP))*@all(τ_xx)
    @all(τ_yy) = (1.0-@all(LamP))*@all(τ_yy)
    @all(τ_xy) = (1.0- @av(LamP))*@all(τ_xy)
    return
end

@parallel function compute_plast_3!(τII::Data.Array, LamP2::Data.Array, τ_xx::Data.Array, τ_yy::Data.Array, τ_xyn::Data.Array, LamP::Data.Array)
    @all(τII)   = sqrt( 0.5*(@all(τ_xx)*@all(τ_xx) + @all(τ_yy)*@all(τ_yy)) + @all(τ_xyn)*@all(τ_xyn) )
    @all(LamP2) = @all(LamP2) + @all(LamP)
    return
end

@parallel function compute_9!(Res_Vx::Data.Array, Res_Vy::Data.Array, τ_xx::Data.Array, τ_yy::Data.Array, τ_xyn::Data.Array, Ptot::Data.Array, τ_xy::Data.Array, dx::Data.Number, dy::Data.Number)
    @all(Res_Vx) = -@d_xi(Ptot)/dx + @d_xi(τ_xx)/dx + @d_ya(τ_xy)/dy  # HORIZONTAL FORCE BALANCE
    @all(Res_Vy) = -@d_yi(Ptot)/dy + @d_yi(τ_yy)/dy + @d_xa(τ_xy)/dx  # VERTICAL   FORCE BALANCE
    return
end

@parallel function compute_10!(dVxdτ::Data.Array, dVydτ::Data.Array, Vx::Data.Array, Vy::Data.Array, Res_Vx::Data.Array, Res_Vy::Data.Array, dampV::Data.Number, dt_Stokes::Data.Number)
    @all(dVxdτ) = @all(dVxdτ)*dampV + @all(Res_Vx) # dVxdτ .= dVxdτ.*(1-Vdmp/nx) .+ Rx
    @all(dVydτ) = @all(dVydτ)*dampV + @all(Res_Vy)
    @inn(Vx)    = @inn(Vx) + dt_Stokes*@all(dVxdτ)  # Pseudo-transient form of horizontal force balance
    @inn(Vy)    = @inn(Vy) + dt_Stokes*@all(dVydτ)  # Pseudo-transient form of vertical force balance
    return
end

@parallel function postprocess!(dRhoT_dt::Data.Array, dRhosPhi_dt::Data.Array, dRhofPhi_dt::Data.Array, dRhoXPhi_dt::Data.Array, dPf_dt::Data.Array, dPt_dt::Data.Array, dPhi_dt::Data.Array, dRhos_dt::Data.Array, Rho_s::Data.Array, Rho_f::Data.Array, X_s::Data.Array, Phi::Data.Array, Rho_s_old::Data.Array, Rho_f_old::Data.Array, Rho_X_old::Data.Array, Phi_old::Data.Array, Rho_t::Data.Array, Rho_t_old::Data.Array, Pf::Data.Array, Pf_old::Data.Array, Ptot::Data.Array, Ptot_old::Data.Array, dtp::Data.Number)
    @all(dRhoT_dt)    = (@all(Rho_t)                         - @all(Rho_t_old))                     /dtp
    @all(dRhosPhi_dt) = (@all(Rho_s)*(1.0-@all(Phi))         - @all(Rho_s_old)*(1.0-@all(Phi_old))) /dtp
    @all(dRhofPhi_dt) = (@all(Rho_f)*@all(Phi)               - @all(Rho_f_old)*@all(Phi_old))       /dtp
    @all(dRhoXPhi_dt) = (@all(Rho_s)*@all(X_s)*(1-@all(Phi)) - @all(Rho_X_old)*(1-@all(Phi_old)))   /dtp
    @all(dPf_dt)      = (@all(Pf)                            - @all(Pf_old))                        /dtp
    @all(dPt_dt)      = (@all(Ptot)                          - @all(Ptot_old))                      /dtp
    @all(dPhi_dt)     = (@all(Phi)                           - @all(Phi_old))                       /dtp
    @all(dRhos_dt)    = (@all(Rho_s)                         - @all(Rho_s_old))                     /dtp
    return
end
##################################################
@views function PT_HMC_()
    runid      = "plast1"
    nsave      = 25
    nviz       = 20
    do_restart = false
    irestart   = 1750 # Step to restart from if do_reatart
    # read in mat file
    vars            = matread( string(@__DIR__, "/LOOK_UP_atg.mat") )
    Rho_s_LU        = get(vars, "Rho_s"  ,1)
    Rho_f_LU        = get(vars, "Rho_f"  ,1)
    X_LU            = get(vars, "X_s_vec",1)
    P_LU            = get(vars, "P_vec"  ,1)*1e8
    # Independent parameters
    rad             = 1.0          # Radius of initial P-perturbation [m]
    η_m             = 1.0          # Viscosity scale [Pa s]
    P_ini           = 1.0          # Initial ambient pressure [Pa]
    ρ_0             = 3000.0       # Density scale [kg/m^3]
    # Nondimensional parameters
    elli_fac        = 2.0          # Use 1 for circle
    α               = 0.0          # Counterclockwise angle of long axis with respect to vertical direction
    ϕ_ini           = 2e-2         # Initial porosity
    ϕ_exp           = 30.0         # Parameter controlling viscosity-porosity relation
    lx_rad          = 40.0         # LAMBDA_1 in equation (15) in the manuscript; Model height divided by inclusion radius;
    lc_rad2         = 1e1          # LAMBDA_2 in equation (15) in the manuscript; Lc_rad2 = k_etaf*eta_mat/radius^2; []; Ratio of hydraulic fluid extraction to compaction extraction
    λ_η             = 2.0          # LAMBDA_3 in equation (15) in the manuscript; lam_eta = lambda / eta_mat; []; Ratio of bulk to shear viscosity
    ly_lx           = 1.0          # Model height divided by model width
    Pini_Pappl      = P_ini/12.75e8 # Dimensionless ratio of abritrary model-P_ini to P_ini in applicable Pa-values; necessary for Look-up table
    σ_y             = 150.0*Pini_Pappl*1e6 # yield stress
    # Dependant parameters
    K_s             = 1e11*Pini_Pappl      # Solid elastic bulk modulus
    k_ηf            = lc_rad2*rad^2/η_m    # Permeability divided by fluid viscosity; [m^2/(Pa s)]
    lx              = lx_rad*rad           # Model width [m]
    ly              = ly_lx*lx             # Model height [m]
    P_LU            = P_LU*Pini_Pappl      # Transform look-up table stress to PT stress scale
    K_d             = K_s/2.0              # Elastic bulk modulus, drained
    α               = 1.0 - K_d/K_s        # Biot-Willis coefficient
    # Characteristic time scales
    τ_f_dif         = rad^2 / (k_ηf        *K_s)
    τ_f_dif_ϕ       = rad^2 / (k_ηf*ϕ_ini^3*K_s)
    τ_relax         = η_m   / K_s
    τ_kinetic       = 1.0/5.0/3.0/3.0/3.0 * 1/3*τ_f_dif_ϕ
    τ_deform        = 0.5/7.0 * 1.0*τ_f_dif_ϕ
    ε_bg            = 1.0 / τ_deform
    Da              = ε_bg/(P_ini/η_m)   # Re-scaling of Da (LAMBDA_4)
    # Numerics
    nt              = 1e4
    tol             = 1e-8                             # Tolerance for pseudo-transient iterations
    cfl             = 1.0/16.1                         # CFL parameter for PT-Stokes solution
    damping         = 1
    Re_Pf           = 60π
    Re_V            = 60π
    r               = 1.5
    dt_fact         = 1.0
    dtp             = τ_f_dif / 2.0 / dt_fact          # Time step physical
    time_tot        = dt_fact * 2*5e3*dtp              # Total time of simulation
    itmax           = 5e6
    nout            = 1e3
    nsm             = 5                                # number of porosity smoothing steps - explicit diffusion
    η_min           = 1e-6
    kin_time        = 1e1*τ_f_dif_ϕ
    kin_time_final  = τ_kinetic
    Kin_time_vec    = [1e25*τ_f_dif_ϕ; range(kin_time, stop=kin_time_final, length=20); kin_time_final*ones(Int(round(time_tot/dtp)))]
    # Configuration of grid, matrices and numerical parameters
    dx, dy          = lx/(nx-1), ly/(ny-1)             # Grid spacing
    xc, yc          = -lx/2:dx:lx/2, -ly/2:dy:ly/2     # Coordinate vector
    xv, yv          = -lx/2-dx/2:dx:lx/2+dx/2, -ly/2-dy/2:dy:ly/2+dy/2  # Horizontal vector for Vx which is one more than basic grid
    (Xc2, Yc2)      = ([x for x=xc, y=yc], [y for x=xc, y=yc])
    (Xc2vx, Yc2vx)  = ([x for x=xv, y=yc], [y for x=xv, y=yc])
    (Xc2vy, Yc2vy)  = ([x for x=xc, y=yv], [y for x=xc, y=yv])
    rad_a           = rad
    rad_b           = rad_a*elli_fac
    X_rot           =  Xc2*cosd(α)+Yc2*sind(α)
    Y_rot           = -Xc2*sind(α)+Yc2*cosd(α)
    X_elli          =  rad_a.*cos.(0:0.01:2*pi).*cosd(α).+rad_b.*sin.(0:0.01:2*pi).*sind(α)
    Y_elli          = -rad_a.*cos.(0:0.01:2*pi).*sind(α).+rad_b.*sin.(0:0.01:2*pi).*cosd(α)
    XY_elli         = (-X_elli, Y_elli)
    # Analytical fit of look-up table
    p_minA, p_maxA  = minimum(P_LU), maximum(P_LU)
    rho_s_up        = 50.0
    Pf              = P_ini*ones(nx, ny) 
    SlopeA          = (Pf.-p_minA)/p_maxA*rho_s_up
    rho_s_max       = Rho_s_LU[1]
    rho_s_minA      = minimum(Rho_s_LU)
    rho_s_difA      = rho_s_max-rho_s_minA
    p_reactA        = 12.65*1e8*Pini_Pappl
    rho_f_maxA      = maximum(Rho_f_LU) # Parameters for fluid density
    x_max           = maximum(X_LU)     # Parameters for mass fraction
    x_minA          = minimum(X_LU)
    x_difA          = x_max-x_minA
    # Density, compressibility and gamma from concentration and pressure from thermodynamic data base
    Rho_s           = Data.Array( (-tanh.( 6e2*(Pf.-p_reactA) )*(rho_s_difA/2.0 .+ rho_s_up/3.0) .+ (rho_s_difA/2.0 .- rho_s_up/3.0) .+ rho_s_minA .+ SlopeA)/ρ_0 )
    Rho_f           = Data.Array( (rho_f_maxA .* log.(Pf .+ 1.0).^(1.0/3.5))/ρ_0 )
    X_s             = Data.Array(  -tanh.( 6e2*(Pf.-p_reactA) )*x_difA/2.0 .+ x_difA/2.0 .+ x_minA )
    SlopeA          = Data.Array( SlopeA )
    # Initialize ALL arrays in Julia
    Ptot            = @zeros(nx  , ny  )               # Initial ambient fluid pressure
    ∇V              = @zeros(nx  , ny  )
    ∇V_ρ_X          = @zeros(nx  , ny  )
    ∇V_ρ_t          = @zeros(nx  , ny  )
    τ_xx            = @zeros(nx  , ny  )               # Deviatoric stress
    τ_yy            = @zeros(nx  , ny  )               # Deviatoric stress
    τ_xy            = @zeros(nx-1, ny-1)               # Deviatoric stress
    τ_xyn           = @zeros(nx  , ny  )
    Res_Vx          = @zeros(nx-1, ny-2)
    Res_Vy          = @zeros(nx-2, ny-1)
    dVxdτ           = @zeros(nx-1, ny-2) # damping
    dVydτ           = @zeros(nx-2, ny-1) # damping
    Rho_s_eq        = @zeros(nx  , ny  )
    X_s_eq          = @zeros(nx  , ny  )
    Rho_s_old       = @zeros(nx  , ny  )
    Rho_f_old       = @zeros(nx  , ny  )
    X_s_old         = @zeros(nx  , ny  )
    Rho_X_old       = @zeros(nx  , ny  )
    Phi_old         = @zeros(nx  , ny  )
    Ptot_old        = @zeros(nx  , ny  )
    Pf_old          = @zeros(nx  , ny  )
    Rho_t_old       = @zeros(nx  , ny  )
    Rho_t           = @zeros(nx  , ny  )
    para_cx         = @zeros(nx-1, ny  )
    para_cy         = @zeros(nx  , ny-1)
    q_f_X           = @zeros(nx-1, ny  )
    q_f_Y           = @zeros(nx  , ny-1)
    ∇q_f            = @zeros(nx-2, ny-2)
    Res_Pf          = @zeros(nx-2, ny-2)
    dPfdτ           = @zeros(nx-2, ny-2) # damping
    Rho_X_ϕ         = @zeros(nx  , ny  )
    Res_Phi         = @zeros(nx  , ny  )
    ε_xx            = @zeros(nx  , ny  )
    ε_yy            = @zeros(nx  , ny  )
    ε_xy            = @zeros(nx-1, ny-1)
    τII             = @zeros(nx  , ny  )
    LamP            = @zeros(nx  , ny  )
    LamP2           = @zeros(nx  , ny  )
    # Arrays for post-processing
    dRhoT_dt        = @zeros(nx  , ny  ) 
    dRhosPhi_dt     = @zeros(nx  , ny  ) 
    dRhofPhi_dt     = @zeros(nx  , ny  ) 
    dRhoXPhi_dt     = @zeros(nx  , ny  ) 
    dPf_dt          = @zeros(nx  , ny  ) 
    dPt_dt          = @zeros(nx  , ny  ) 
    dPhi_dt         = @zeros(nx  , ny  ) 
    dRhos_dt        = @zeros(nx  , ny  ) 
    # TMP arrays for swell2
    TmpX            = @zeros(nx+1, ny  )
    TmpY            = @zeros(nx  , ny+1)
    TmpS1           = @zeros(nx  , ny-1)
    # arrays for visu
    Vx_f            = zeros(nx  , ny  )
    Vy_f            = zeros(nx  , ny  )
    # init
    Phi_ini         = ϕ_ini*ones(nx, ny)
    # Phi_ini[sqrt.((X_rot .- 2.0).^2 ./ rad_a.^2 .+ (Y_rot .+ 6.0).^2 ./ rad_b.^2) .< 1.0] .= 2.5*ϕ_ini  # Porosity petubation
    # Phi_ini[sqrt.((X_rot .+ 2.0).^2 ./ rad_a.^2 .+ (Y_rot .- 6.0).^2 ./ rad_b.^2) .< 1.0] .= 2.5*ϕ_ini  # Porosity petubation
    # for smo=1:nsm # Smooting of perturbation
    #     Phi_ini[2:end-1,:] .= Phi_ini[2:end-1,:] .+ 0.4.*(Phi_ini[3:end,:].-2.0.*Phi_ini[2:end-1,:].+Phi_ini[1:end-2,:])
    #     Phi_ini[:,2:end-1] .= Phi_ini[:,2:end-1] .+ 0.4.*(Phi_ini[:,3:end].-2.0.*Phi_ini[:,2:end-1].+Phi_ini[:,1:end-2])
    # end
    Pert            = Data.Array((8 .* Phi_ini .- Phi_ini) .* exp.(.-(Xc2./rad_a).^2 .- (Yc2./rad_b).^2))
    Phi_ini         = Data.Array(Phi_ini)
    Phi             = Phi_ini .+ Pert
    # Phi             = copy(Phi_ini)
    Eta             =   η_m*@ones(nx, ny)         # Shear viscosity, alternative init:   # @all(Eta) = η_m*exp(-ϕ_exp*(@all(Phi)-ϕ_ini))
    Lam             =   λ_η*@ones(nx, ny)         # Bulk viscosity, alternative init:   # @all(Lam) = λ_η*@all(Eta)
    # Vx_ps           =  ε_bg*Data.Array(Xc2vx)     # Pure shear, shortening in x
    # Vy_ps           = -ε_bg*Data.Array(Yc2vy)     # Pure shear, extension in y
    # Vx_ss           =  ε_bg*Data.Array(Yc2vx)     # Simple shear, shearing in x
    # Vy_ss           =   0.0*Data.Array(Yc2vy)     # Simple shear, zero velocity in y
    # Vx              = 0.5.*(Vx_ps .+ Vx_ss)
    # Vy              = 0.5.*(Vy_ps .+ Vy_ss)
    Vx              = ε_bg*Data.Array(Yc2vx)     # Simple shear, shearing in x
    Vy              =  0.0*Data.Array(Yc2vy)     # Simple shear, zero velocity in y
    Pf              = Data.Array(Pf)
    Ptot           .= Pf                          # Initial total pressure
    # Parameters for time loop and pseudo-transient iterations
    max_dxdy2       = max(dx,dy).^2
    timeP           = 0.0                         # Initial time
    it              = 0                           # Integer count for iteration loop
    itp             = 0                           # Integer count for time loop
    save_count      = 0 
    Time_vec        = []
    ρ_i_Pf          = cfl*Re_Pf/nx
    ρ_i_V           = cfl*Re_V /nx
    dampPf          = damping.*(1.0 .- ρ_i_Pf)
    dampV           = damping.*(1.0 .- ρ_i_V )
    if do_save
        !ispath(joinpath(@__DIR__,"../output")) && mkdir(joinpath(@__DIR__,"../output"))
        dirname = joinpath(@__DIR__, "../output/output_$(runid)_$(nx)x$(ny)"); !ispath(dirname) && mkdir(dirname)
    end
    it_viz = 0
    # time loop
    while timeP < time_tot && itp < nt
        if do_restart
            restart_file = joinpath(@__DIR__, dirname, "pt_hmc_Atg_") * @sprintf("%04d", irestart) * ".mat"
            vars_restart = matread(restart_file)
            Ptot       .= Data.Array(get(vars_restart, "Ptot"     ,1))
            Pf         .= Data.Array(get(vars_restart, "Pf"       ,1))
            X_s        .= Data.Array(get(vars_restart, "X_s"      ,1))
            Phi        .= Data.Array(get(vars_restart, "Phi"      ,1))
            Rho_s      .= Data.Array(get(vars_restart, "Rho_s"    ,1))
            Rho_f      .= Data.Array(get(vars_restart, "Rho_f"    ,1))
            Vx         .= Data.Array(get(vars_restart, "Vx"       ,1))
            Vy         .= Data.Array(get(vars_restart, "Vy"       ,1))
            Vx_f       .= Data.Array(get(vars_restart, "Vx_f"     ,1))
            Vy_f       .= Data.Array(get(vars_restart, "Vy_f"     ,1)) 
            LamP2      .= Data.Array(get(vars_restart, "LamP2"    ,1))
            Rho_s_old  .= Data.Array(get(vars_restart, "Rho_s_old",1))
            Rho_f_old  .= Data.Array(get(vars_restart, "Rho_f_old",1))
            X_s_old    .= Data.Array(get(vars_restart, "X_s_old"  ,1))
            Rho_X_old  .= Data.Array(get(vars_restart, "Rho_X_old",1))
            Phi_old    .= Data.Array(get(vars_restart, "Phi_old"  ,1))
            Ptot_old   .= Data.Array(get(vars_restart, "Ptot_old" ,1))
            Pf_old     .= Data.Array(get(vars_restart, "Pf_old"   ,1))
            Rho_t_old  .= Data.Array(get(vars_restart, "Rho_t_old",1))
            Time_vec    =            get(vars_restart, "Time_vec" ,1)
            itp         =            get(vars_restart, "itp"      ,1)
            timeP       =            get(vars_restart, "timeP"    ,1)
            it          =            get(vars_restart, "it"       ,1)
            it_tstep    =            get(vars_restart, "it_tstep" ,1)
            do_restart  = false
        end
    	err_M = 2*tol; itp += 1
        timeP = timeP+dtp; push!(Time_vec, timeP)
        @parallel compute_1!(Rho_s_old, Rho_f_old, X_s_old, Rho_X_old, Phi_old, Ptot_old, Pf_old, Rho_t_old, Eta, Lam, Rho_s, Rho_f, X_s, Phi, Ptot, Pf, η_m, ϕ_exp, ϕ_ini, λ_η)
        kin_time = Kin_time_vec[itp]
    	# PT loop
    	it_tstep=0; err_evo1=[]; err_evo2=[]; err_pl=[]
        dt_Stokes, dt_Pf = cfl*max_dxdy2, cfl*max_dxdy2
        dt_Pt = maximum(Eta)/(lx/dx)
    	while err_M>tol && it_tstep<itmax
            it += 1; it_tstep += 1
            if it_tstep % 500 == 0 || it_tstep==1
                max_Eta   = maximum(Eta)
                dt_Stokes = cfl*max_dxdy2/max_Eta*(2-ρ_i_V)/1.5            # Pseudo time step for Stokes
                dt_Pf     = cfl*max_dxdy2/maximum(k_ηf.*Phi.^3*(4.0*K_s))  # Pseudo time step for fluid pressure
                dt_Pt     = r*Re_V*max_Eta*dx/lx
            end
            # Fluid pressure evolution
            @parallel compute_2!(Rho_t, para_cx, para_cy, Rho_f, Phi, Rho_s, k_ηf)
            @parallel compute_3!(q_f_X, q_f_Y, para_cx, para_cy, Pf, dx, dy)
            @parallel compute_4!(∇q_f, Res_Pf, Rho_f, Rho_s_eq, X_s_eq, q_f_X, q_f_Y, Rho_t, Rho_t_old, ∇V_ρ_t, Pf, SlopeA, rho_f_maxA, ρ_0, p_reactA, rho_s_difA, rho_s_up, rho_s_minA, x_difA, x_minA, dtp, dx, dy)
            if (itp > 1) @parallel compute_5!(X_s, Rho_s, X_s_old, X_s_eq, Rho_s_old, Rho_s_eq, dtp, kin_time) end
            # Porosity evolution
            @parallel compute_6!(dPfdτ, Pf, Rho_X_ϕ, Res_Phi, Phi, Res_Pf, Rho_s, X_s, Phi_old, Rho_X_old, ∇V_ρ_X, dampPf, dt_Pf, dtp)
            @parallel (1:size(Pf,1)) bc_y!(Pf)
            @parallel (1:size(Pf,2)) bc_x!(Pf)
            # Stokes
            @parallel swell2_x!(TmpX, Rho_X_ϕ)
            @parallel swell2_y!(TmpY, Rho_X_ϕ)
            @parallel cum_mult2!(TmpX, TmpY, Vx, Vy)
            @parallel laplace!(∇V_ρ_X, TmpX, TmpY, dx, dy)
            @parallel swell2_x!(TmpX, Rho_t)
            @parallel swell2_y!(TmpY, Rho_t)
            @parallel cum_mult2!(TmpX, TmpY, Vx, Vy)
            @parallel laplace!(∇V_ρ_t, TmpX, TmpY, dx, dy)
            @parallel compute_7!(Eta, Lam, ∇V, ε_xx, ε_yy, ε_xy, Ptot, Vx, Vy, Phi, Ptot_old, Pf, Pf_old, dt_Pt, η_m, ϕ_exp, ϕ_ini, λ_η, dtp, K_d, η_min, α, dx, dy)
            @parallel compute_8!(τ_xx, τ_yy, τ_xy, Eta, ε_xx, ε_yy, ε_xy)
            @parallel swell2_x!(TmpS1, τ_xy)
            @parallel swell2_y!(τ_xyn, TmpS1)
            # Plastic starts
            @parallel compute_plast_1!(τII, LamP, τ_xx, τ_yy, τ_xyn, σ_y)
            @parallel compute_plast_2!(τ_xx, τ_yy, τ_xy, LamP)
            @parallel swell2_x!(TmpS1, τ_xy)
            @parallel swell2_y!(τ_xyn, TmpS1)
            @parallel compute_plast_3!(τII, LamP2, τ_xx, τ_yy, τ_xyn, LamP)
            # Plastic ends
            @parallel compute_9!(Res_Vx, Res_Vy, τ_xx, τ_yy, τ_xyn, Ptot, τ_xy, dx, dy)
            @parallel compute_10!(dVxdτ, dVydτ, Vx, Vy, Res_Vx, Res_Vy, dampV, dt_Stokes)
            if it_tstep % nout == 0 && it_tstep > 250
                err_Mx   = dt_Stokes*maximum(abs.(Res_Vx)/maximum(abs.(Vx)))    # Error horizontal velocitiy
                err_R_Mx = maximum(abs.(Res_Vx))                                # Error horizontal force balance
                err_My   = dt_Stokes*maximum(abs.(Res_Vy)/maximum(abs.(Vy)))    # Error vertical velocity
                err_R_My = maximum(abs.(Res_Vy))                                # Error vertical force balance
                err_Pf   = dt_Pf*maximum(abs.(Res_Pf)/maximum(abs.(Pf)))        # Error fluid pressure
                err_R_Pf = maximum(abs.(Res_Pf))                                # Error total mass conservation
                err_Phi  = dtp*maximum(abs.(Res_Phi))                           # Error porosity
                err_R_Phi= maximum(abs.(Res_Phi))                               # Error MgO mass conservation
                err_M    = maximum([err_Pf, err_Mx, err_My, err_Phi, err_R_Pf, err_R_Mx, err_R_My, err_R_Phi]) # Error total
                if isnan(err_M) error("NoNs - stopping simulation.") end
                err_evo1 = push!(err_evo1, it_tstep); err_evo2 = push!(err_evo2, err_M)
                err_pl   = push!(err_pl, maximum(τII)/σ_y - 1.0)
                # plot evol
                # p1 = plot(err_evo1, err_evo2, legend=false, xlabel="# iterations", ylabel="log10(error)", linewidth=2, markershape=:circle, markersize=3, labels="max(error)", yaxis=:log10)
                # display(p1)
                @printf("iter = %d, error = %1.3e \n", it_tstep, err_M)
            end
        end # end PT loop
        println("it = $(itp), time = $(round(timeP, sigdigits=3)) (time_tot = $(round(time_tot, sigdigits=3)))")
        # Visu
        if do_viz && (itp % nviz == 0 || itp == 1)
            it_viz += 1
            swell2h!(Vx_f, Array(q_f_X)./(Array(Rho_f[1:end-1,:])./Array(Phi[1:end-1,:]).+Array(Rho_f[2:end,:])./Array(Phi[2:end,:])).*2.0, 1)
            swell2h!(Vy_f, Array(q_f_Y)./(Array(Rho_f[:,1:end-1])./Array(Phi[:,1:end-1]).+Array(Rho_f[:,2:end])./Array(Phi[:,2:end])).*2.0, 2)
            η_char         = 1e19
            Length_model_m = rad
            Time_model_sec = rad^2/(k_ηf*K_s)
            Length_phys_m  = 0.01
            Time_phys_sec  = 0.01^2 / (1e-19 / 1e-3 / (1e-2/8.5e8))
            Vel_phys_m_s   = (Length_phys_m/Length_model_m) / (Time_phys_sec/Time_model_sec)
            lw = 1.2; fontsize = 8
            opts1 = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), 
                     ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                     xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c = cgrad(:davos, rev = true) )
            opts2 = (linewidth=lw, linecolor="white", legend=false, framestyle=:box)
            p2  = heatmap(xc, yc, Array(Pf)'./Pini_Pappl./1e8; title="A) p_f [kbar]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2... )
            p3  = heatmap(xc, yc, Array(Ptot)'./Pini_Pappl./1e8; title="B) p [kbar]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p4  = heatmap(xc, yc, Array(∇V)'.*Time_model_sec./Time_phys_sec; title="C) ∇(v_s) [1/s]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p5  = heatmap(xc, yc, Array(X_s)'; title="D) X_s", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p6  = heatmap(xc, yc, Array(Rho_s)'.*ρ_0; title="E) ρ_s [kg/m^3]", opts1...)
            # p6  = heatmap(xc, yc, sqrt.(av_xa(Array(Vx)).^2 .+ av_ya(Array(Vy)).^2)'*Vel_phys_m_s; title="E) ||v_s|| [m/s]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p7  = heatmap(xc, yc, Array(Phi)'; title="F) ϕ", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p8  = heatmap(xc, yc, Array(Rho_f)'.*ρ_0; title="G) ρ_f [kg/m^3]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p9  = heatmap(xc, yc, Array(τII)'./Pini_Pappl./1e6; title="H) τII [MPa]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            p10 = heatmap(xc, yc, log10.(Array(Eta)'.*η_char); title="I) log10(ηs) [Pas]", opts1...)
            # plot!(XY_elli[1], XY_elli[2]; opts2...)
            display(plot(p2, p3, p4, p5, p6, p7, p8, p9, p10, background_color=:transparent, foreground_color=:gray, dpi=300))
            #savefig("output_$(runid)_$(nx)x$(ny)/PT_HMC_Atg_$(nx)x$(ny)_$(it_viz).png")
        end
        if do_save && (itp % nsave == 0 || itp==1)
            @parallel postprocess!(dRhoT_dt, dRhosPhi_dt, dRhofPhi_dt, dRhoXPhi_dt, dPf_dt, dPt_dt, dPhi_dt, dRhos_dt, Rho_s, Rho_f, X_s, Phi, Rho_s_old, Rho_f_old, Rho_X_old, Phi_old, Rho_t, Rho_t_old, Pf, Pf_old, Ptot, Ptot_old, dtp)
            matwrite(joinpath(@__DIR__, dirname, "pt_hmc_Atg_") * @sprintf("%04d", itp) * ".mat",
                      Dict("Ptot"=> Array(Ptot),
                           "Pf"=> Array(Pf),
                           "divV"=> Array(∇V),
                           "X_s"=> Array(X_s),
                           "Rho_s"=> Array(Rho_s),
                           "Phi"=> Array(Phi),
                           "Rho_f"=> Array(Rho_f),
                           "TauII"=> Array(τII),
                           "Eta"=> Array(Eta),
                           "Tauxx"=> Array(τ_xx),
                           "Tauyy"=> Array(τ_yy),
                           "Tauxy"=> Array(τ_xy),
                           "Vx"=> Array(Vx),
                           "Vy"=> Array(Vy),
                           "Vx_f"=> Array(Vx_f),
                           "Vy_f"=> Array(Vy_f),
                           "Ellipse_x"=> Array(X_elli),
                           "Ellipse_y"=> Array(Y_elli),
                           "Time_vec"=> Array(Time_vec),
                           "LamP2"=> Array(LamP2),
                           "Rho_s_old"=> Array(Rho_s_old),
                           "Rho_f_old"=> Array(Rho_f_old),
                           "X_s_old"=> Array(X_s_old),
                           "Rho_X_old"=> Array(Rho_X_old),
                           "Phi_old"=> Array(Phi_old),
                           "Ptot_old"=> Array(Ptot_old),
                           "Pf_old"=> Array(Pf_old),
                           "Rho_t_old"=> Array(Rho_t_old),
                           "Lam"=> Array(Lam),
                           "dRhoT_dt"=> Array(dRhoT_dt),
                           "dRhosPhi_dt"=> Array(dRhosPhi_dt),
                           "dRhofPhi_dt"=> Array(dRhofPhi_dt),
                           "dRhoXPhi_dt"=> Array(dRhoXPhi_dt),
                           "dPf_dt"=> Array(dPf_dt),
                           "dPt_dt"=> Array(dPt_dt),
                           "dPhi_dt"=> Array(dPhi_dt),
                           "dRhos_dt"=> Array(dRhos_dt),
                           "Res_pf"=> Array(Res_Pf),
                           "div_qf"=> Array(∇q_f),
                           "div_rhoTvs"=> Array(∇V_ρ_t),
                           "div_rhoXvs"=> Array(∇V_ρ_X),
                           "Ch_ti_fluid_dif_phi"=> τ_f_dif_ϕ,
                           "Ch_ti_deformation"=> τ_deform,
                           "Ch_ti_kinetic"=> τ_kinetic,
                           "Ch_ti_relaxation"=> τ_relax,
                           "itp"=> itp,
                           "timeP"=> timeP,
                           "it"=> it,
                           "it_tstep"=> it_tstep,
                           "xc"=> Array(xc), "yc"=> Array(yc)); compress = true)
        end
        if (use_return && itp==1) break; end
    end
    return xc, yc, Pf, Phi
end

if use_return
    xc, yc, Pf, Phi = PT_HMC_();
else
    PT_HMC = begin PT_HMC_(); return; end
end
