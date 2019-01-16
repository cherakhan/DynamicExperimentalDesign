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
c = LArray{(:Vj, :Mp)}([Vj, Mp]) 
θ = [4.416797788362816e-6
1.097375874222653
4.389503496274951
0.5506105422024243
3.653543911566884e-6
0.4245176907364542]
θ = LArray{(:VmO2, :KmO2, :KiO2, :RQ, :VmCO2, :KiCO2)}(θ)
tswitch = 0.0:2.0*3600.0:24*3600.0
u_initial = @LVector Float64 (:Q,:O2in,:CO2in)
obj = objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)
sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
@time sol = obj(zeros(length(tswitch)-1,length(u_initial)),[]);
@code_warntype obj(zeros(length(tswitch)-1,length(u_initial)),[])
@code_warntype objective_generator(mass_balance!,x0,c,θ,u_initial,tswitch)

function bl(θ)
n_θ = length(θ)
syms_θ = propertynames(θ)
dual_θ = Vector{Dual{ParSen,eltype(θ),n_θ}}(undef, n_θ)
end
@code_warntype bl(θ)
testo = bl(θ)


self_sen = Matrix{Float64}(I, n_θ, n_θ)
for i in 1:n_θ
    dual_θ[i] = Dual{ParSen}(θ[i], self_sen[i,:]...)
end
dual_θ= LArray{syms_θ}(dual_θ)