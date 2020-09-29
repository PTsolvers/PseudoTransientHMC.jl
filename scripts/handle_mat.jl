using Plots, Printf, Statistics, LinearAlgebra
# read in mat file
using MAT
vars = matread("LOOK_UP_HMC_Pub.mat")
RHO_s_LU = get(vars, "Rho_s_07",1)
Rho_f_LU = get(vars, "Rho_f",1)
X_LU     = get(vars, "X_s_vec",1)
P_LU     = get(vars, "P_vec",1)*1e8

plot(RHO_s_LU)
