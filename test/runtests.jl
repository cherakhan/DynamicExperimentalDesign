using Test
using Revise
#include("C:\\Users\\Arno\\Dropbox\\git\\DynamicExperimentalDesign\\main.jl")
#include("/home/arno/Dropbox/git/DynamicExperimentalDesign/main.jl")
using DynamicExperimentalDesign
using LabelledArrays
using StaticArrays
using ForwardDiff 
using Plots
@inline function mass_balance!(dx, x, p, t)
    rO2  = p.θ.VmO2 *x.O2 /((p.θ.KmO2 + x.O2)*(1.0+x.CO2/p.θ.KiO2))
    fCO2 = p.θ.VmCO2*x.CO2/(1.0 + x.O2/p.θ.KiCO2)
    rCO2 = p.θ.RQ*rO2 + fCO2
    dx.O2  = 1/p.c.Vj*(p.u.Q * (p.u.O2in  - x.O2)  - p.c.Mp*rO2)
    dx.CO2 = 1/p.c.Vj*(p.u.Q * (p.u.CO2in - x.CO2) + p.c.Mp*rCO2)
    return nothing
end
Rho_p = 1637.14780257
R = 8.3144598
T = 273.15+1.0
Tref = 273.15+20
Patm = 101325
O2atm = 0.21
O20 = O2atm*Patm/(R*T)
CO2atm = 0.0041
CO20 = CO2atm*Patm/(R*T)
x0 =  LArray{(:O2,:CO2)}([O20,CO20]) 
Vj = 5.0/1000; Mp = 4.0
c = SLArray{Tuple{2},1,(:Vj, :Mp),Float64}([Vj, Mp]) 


θ = @SArray [  1.4925677111223096e-8/(exp(80.2*1000.0*(1/Tref - 1/T)/R))
0.4387101798598021
29.13035594269086
0.97
2.0588290772191242e-8/(exp(80.2*1000.0*(1/Tref - 1/T)/R))
0.12283885036074459]

θ = SLArray{Tuple{6},1,(:VmO2, :KmO2, :KiO2, :RQ, :VmCO2, :KiCO2),Float64}(θ)
tswitch = 0.0:2.0*3600.0:24*3600.0
u_initial = @LVector Float64 (:Q,:O2in,:CO2in)
obj = DynamicExperimentalDesign.objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)
sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
timeplot = sol.t
O2_plot = zeros(length(sol.u))
for k in 1:length(sol.u)
    O2_plot[k] = sol.u[k][1].value
end
CO2_plot = zeros(length(sol.u))
for k in 1:length(sol.u)
    CO2_plot[k] = sol.u[k][2].value
end
plot(timeplot,O2_plot)
plot!(timeplot,CO2_plot)
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