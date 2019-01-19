include("C:\\Users\\Arno\\Dropbox\\git\\dynamic\\main.jl")
function mass_balance!(dx, x, p, t)
rO2  = p.θ.VmO2  * x.O2 /((p.θ.KmO2 + x.O2)*(1.0+x.CO2/p.θ.KiO2))
rCO2 = p.θ.VmCO2 * x.CO2/(1.0 + x.O2/p.θ.KiCO2) + p.θ.RQ*rO2
dx.O2  = 1/p.c.Vj*(p.u.Q * (p.u.O2in  - x.O2)  - p.c.Mp * rO2)
dx.CO2 = 1/p.c.Vj*(p.u.Q * (p.u.CO2in - x.CO2) + p.c.Mp * rCO2) 
end
O20 = 0.008729954310467655
CO20 = 0.0004000725171963509
x0 =  LArray{(:O2,:CO2)}([O20,CO20]) 
Vj = 5.0
Mp = 4.0
c = SLArray{Tuple{2},1,(:Vj, :Mp),Float64}([Vj, Mp]) 
using StaticArrays
v3 = @SVector [1, 2, 3]
θ = @SArray [4.416797788362816e-6,
1.097375874222653,
4.389503496274951,
0.5506105422024243,
3.653543911566884e-6,
0.4245176907364542]
θ = SLArray{Tuple{6},1,(:VmO2, :KmO2, :KiO2, :RQ, :VmCO2, :KiCO2),Float64}(θ)
testo = SLArray
tswitch = 0.0:2.0*3600.0:24*3600.0
u_initial = @LVector Float64 (:Q,:O2in,:CO2in)
obj = objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)
sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
#@time sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
#@code_warntype obj(zeros(length(tswitch)-1,length(u_initial)),[])
#@code_warntype objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)

