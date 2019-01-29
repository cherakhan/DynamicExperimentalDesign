using Test, Revise
#include("C:\\Users\\Arno\\Dropbox\\git\\DynamicExperimentalDesign\\main.jl")
#include("/home/arno/Dropbox/git/DynamicExperimentalDesign/main.jl")
using DynamicExperimentalDesign
using LabelledArrays, StaticArrays
using ForwardDiff 
using Plots
using NLopt
@inline function mass_balance!(dx, x, p, t)
    rO2  = p.θ.VmO2 *x.O2 /((p.θ.KmO2 + x.O2)*(1.0+x.CO2/p.θ.KiO2))
    fCO2 = p.θ.VmCO2*x.CO2/(1.0 + x.O2/p.θ.KiCO2 + x.CO2/p.θ.Ki_CO2_CO2)
    rCO2 = p.θ.RQ*rO2 + fCO2
    dx.O2  = 1/p.c.Vj*(p.u.Q * (p.u.O2in  - x.O2)  - p.c.Mp*rO2)
    dx.CO2 = 1/p.c.Vj*(p.u.Q * (p.u.CO2in - x.CO2) + p.c.Mp*rCO2)
    return nothing
end
Rho_p = 1637.14780257; R = 8.3144598; T = 273.15+1.0; Tref = 273.15+20; Patm = 101325;
O2atm = 0.21; O20 = O2atm*Patm/(R*T)
CO2atm = 0.0041;CO20 = CO2atm*Patm/(R*T)
x0 =  LArray{(:O2,:CO2)}([O20,CO20]) 
Vj = 5.0/1000; Mp = 4.0
c = SLArray{Tuple{2},1,(:Vj, :Mp),Float64}([Vj, Mp]) 

θ = @SArray [  1.4925677111223096e-8/(exp(80.2*1000.0*(1/Tref - 1/T)/R))
0.4387101798598021
29.13035594269086
0.97
2.0588290772191242e-8/(exp(80.2*1000.0*(1/Tref - 1/T)/R))
0.12283885036074459];

θ = @SArray [  2.4303952944343485e-8
0.14924311888654018
48.382863341617664
0.6408943680301596
1.2101223453569529e-8
0.04178807328823125
7.109947385315544];

θ = SLArray{Tuple{7},1,(:VmO2, :KmO2, :KiO2, :RQ, :VmCO2, :KiCO2,:Ki_CO2_CO2),Float64}(θ);
tswitch = 0.0:2.0*3600.0:24.0*3600.0;
u_initial = @LVector Float64 (:Q,:O2in,:CO2in);
obj = DynamicExperimentalDesign.objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
sol = obj(zeros((length(tswitch)-1)*length(u_initial)),[]);

opt = Opt(:GN_ESCH, (length(tswitch)-1)*length(u_initial));
max_objective!(opt, obj);
maxeval!(opt,1000000);

lower_bound = zeros((length(tswitch)-1)*length(u_initial));
lower_bounds!(opt, lower_bound);

upper_bound_Q = ones((length(tswitch)-1))*10.0/3600.0/1000.0;  #l/h
upper_bound_O2 = ones((length(tswitch)-1))*10.0;
upper_bound_CO2 = ones((length(tswitch)-1))*10.0;
upper_bound = vcat(upper_bound_Q,upper_bound_O2,upper_bound_CO2);
upper_bounds!(opt, upper_bound);

(max_obj,max_u,ret) = NLopt.optimize(opt,lower_bound)
matrix_inputs = reshape(max_u,:,3)

plotter = DynamicExperimentalDesign.plot_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
sol_ini_val = plotter(reshape(max_u,:,3),[]);
time_plot = sol_ini_val.t
O2_plot = zeros(length(sol_ini_val.t));
CO2_plot = zeros(length(sol_ini_val.t));
for k in 1:length(sol_ini_val.t)
    O2_plot[k] = sol_ini_val.u[k][1].value
    CO2_plot[k] = sol_ini_val.u[k][2].value
end
plot(time_plot/3600,O2_plot);
plot!(time_plot/3600,CO2_plot)


function input_plotter_help(t,matrix_inputs,tswitch)
    for k in 1:length(tswitch)
        if t < tswitch[k]
            return matrix_inputs[k-1,:]
        end
        if t == tswitch[end]
            return matrix_inputs[end,:]
        end
    end
end
function input_plotter(t_plot,matrix_inputs,tswitch)
    input = zeros(length(t_plot),3)
    for k in 1:length(t_plot)
        input[k,:] = input_plotter_help(t_plot[k],matrix_inputs,tswitch)
    end
    return input
end
t_plot = 0:1:3600*24
input = input_plotter(t,matrix_inputs,tswitch)
t_plot_h = t_plot/3600
input_resize = copy(input)
input_resize[:,1] = input_resize[:,1]/maximum(input[:,1])
input_resize[:,2] = input_resize[:,2]/maximum(input[:,2])
input_resize[:,3] = input_resize[:,3]/maximum(input[:,3])
plot(t_plot_h,input_resize)










using JLD2
blah = @load "VarCoVar.jld2" VarCoVar
VarCoVar = eval(blah[1])
theta = [i for i in θ.__x.data]

using LinearAlgebra
using Distributions
using Random

θ_dist = MvNormal(theta, VarCoVar)
rand(θ_dist )
obj = DynamicExperimentalDesign.bay_objective_generator(mass_balance!,x0,c,θ_dist,u_initial,tswitch)
sol = obj(zeros((length(tswitch)-1)*length(u_initial)),[]);
opt = Opt(:GN_ESCH, (length(tswitch)-1)*length(u_initial));
max_objective!(opt, obj);
maxeval!(opt,10000);

lower_bound = zeros((length(tswitch)-1)*length(u_initial));
lower_bounds!(opt, lower_bound);

upper_bound_Q = ones((length(tswitch)-1))*10.0/3600.0/1000.0;  #l/h
upper_bound_O2 = ones((length(tswitch)-1))*10.0;
upper_bound_CO2 = ones((length(tswitch)-1))*10.0;
upper_bound = vcat(upper_bound_Q,upper_bound_O2,upper_bound_CO2);
upper_bounds!(opt, upper_bound);

(max_obj,max_u,ret) = NLopt.optimize(opt,lower_bound)
matrix_inputs = reshape(max_u,:,3)

plotter = DynamicExperimentalDesign.plot_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
sol_ini_val = plotter(reshape(max_u,:,3),[]);
time_plot = sol_ini_val.t
O2_plot = zeros(length(sol_ini_val.t));
CO2_plot = zeros(length(sol_ini_val.t));
for k in 1:length(sol_ini_val.t)
    O2_plot[k] = sol_ini_val.u[k][1].value
    CO2_plot[k] = sol_ini_val.u[k][2].value
end
plot(time_plot/3600,O2_plot);
plot!(time_plot/3600,CO2_plot)







opt_loc = Opt(:LN_SBPLX, (length(tswitch)-1)*length(u_initial));
max_objective!(opt_loc, obj);
maxeval!(opt_loc,100000);

lower_bounds!(opt_loc , lower_bound);
upper_bounds!(opt_loc , upper_bound);

(max_obj_loc,max_u_loc,ret) = NLopt.optimize(opt_loc,max_u)
matrix_inputs_loc = reshape(max_u_loc,:,3)

plotter = DynamicExperimentalDesign.plot_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
sol_ini_val = plotter(reshape(max_u_loc,:,3),[]);
time_plot = sol_ini_val.t
O2_plot = zeros(length(sol_ini_val.t));
CO2_plot = zeros(length(sol_ini_val.t));
for k in 1:length(sol_ini_val.t)
    O2_plot[k] = sol_ini_val.u[k][1].value
    CO2_plot[k] = sol_ini_val.u[k][2].value
end
plot(time_plot/3600,O2_plot);
plot!(time_plot/3600,CO2_plot)























plotter = DynamicExperimentalDesign.plot_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
sol_ini_val = plotter(reshape(zeros(length(max_u)),:,3),[]);
time_plot = sol_ini_val.t
O2_plot = zeros(length(sol_ini_val.t));
CO2_plot = zeros(length(sol_ini_val.t));
for k in 1:length(sol_ini_val.t)
    O2_plot[k] = sol_ini_val.u[k][1].value
    CO2_plot[k] = sol_ini_val.u[k][2].value
end
plot(time_plot/3600,O2_plot);
plot!(time_plot/3600,CO2_plot)

obj = DynamicExperimentalDesign.objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch);
obj(reshape(max_u,:,3),[])


#using Random
#Random.seed!(1234)
#randn()

# using Profile
# Profile.clear_malloc_data() 
# sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
# exit()

# @time obj(zeros(length(tswitch)-1,length(u_initial)),[]);
# @allocated obj(zeros(length(tswitch)-1,length(u_initial)),[])
# @code_warntype obj(zeros(length(tswitch)-1,length(u_initial)),[])
# @code_warntype objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)

# Profile.clear() 
# function prof_test(tswitch,u_initial)
#     for k in 1:10000
#         obj(zeros(length(tswitch)-1,length(u_initial)),[])
#     end
# end
# @profile prof_test(tswitch,u_initial)
# Profile.print()
# using ProfileView
# ProfileView.view()
# open(Profile.print, a, "w")