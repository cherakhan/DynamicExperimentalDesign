using Revise
using DifferentialEquations
using ForwardDiff: Dual
using LabelledArrays
using LinearAlgebra
using Plots

function triangular_index(linear_index,square_size)
    i = linear_index
    j = 1
    for col_length in 1:square_size
        if i <= col_length
            return i,j
        end
        j += 1
        i -= col_length
    end
end

function dynamic_system!(dx, x, p, t)
    dx[1] = 1 / p.c.Vj * (p.u[1] * (p.c.Cin - x[1]) - p.c.mp * p.θ.Vmax * x[1] / (p.θ.Km + x[1]))
 #   dx[2] = 0
    n_θ = length(p.θ)
    n_states = 1
#    n_states = 1
    for linear_index in 1:div(n_θ*(n_θ +1),2)
        dx[n_states+linear_index] = 0.0
        i,j = triangular_index(linear_index,n_θ )
        for s in 1:n_states
            dx[n_states+linear_index] = x[s].partials.values[i]*x[s].partials.values[j]
        end
    end
end

abstract type ParSen end
struct Parameters{T, V<:AbstractVector{T}, W<:AbstractVector{<:Dual{ParSen,T}},U<:AbstractVector{T}}
    c::V
    θ::W
    u::U
end


include("parameters_raw.jl")
constants = LArray{(:Vj, :Cin, :mp)}([Vj, Cin, mp]) 
# unknown parameters
θ = LArray{(:Vmax, :Km)}([Vmax,Km])
n_θ = length(θ)

dual_θ = Vector{Dual{ParSen,eltype(θ),n_θ}}(undef, n_θ)
self_sen = Matrix{Float64}(I, n_θ, n_θ)
for i in 1:n_θ
    dual_θ[i] = Dual{ParSen}(θ[i], self_sen[i,:]...)
end
syms_θ = propertynames(θ)
dual_θ= LArray{syms_θ}(dual_θ)

#control parameter definition
#u = [Qmax]
u = [0.0]
parameters = Parameters(constants, dual_θ, u)

#states ini
#n_x_ini = 2
n_x_ini = 1
n_x = n_x_ini +(div(n_θ*(n_θ +1),2))
states0 = zeros(n_x)
states0[1] = C0

dual_states0 = Vector{Dual{ParSen,eltype(states0),n_θ}}(undef, n_x)
for i in 1:length(states0)
    dual_states0[i] = Dual{ParSen}(states0[i], zeros(n_θ)...)
end
function condition_generator(tstops,parametrized_input::Array{Float64,2})
    count::Int64 = 1
    function condition_closure(x,t,integrator)
        println(count)
        if  count < length(tstops) && t == tstops[count] 
            integrator.p.u[:] = parametrized_input[count,:]
            count += 1
            return true
        end
        return false
    end
end
function affect(integrator)
    return true
end
flowrates = [0.0, 0.0, Qmax, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prob_control = ODEProblem(dynamic_system!, dual_states0, (0, te), parameters)

tstops = tswitch:tswitch:te-tswitch
parametrized_input = flowrates[:,:]
condition = condition_generator(tstops,parametrized_input)

cb = DiscreteCallback(condition, affect,save_positions = (false,false))
cbs = CallbackSet(cb)

sol_control = solve(prob_control,Tsit5(), callback = cbs, tstops = tstops)

plot_states = zeros(length(sol_control.u), length(sol_control.u[1]))
for i = 1:length(sol_control.u)
    for j = 1:length(sol_control.u[1])
        plot_states[i,j] = sol_control.u[i][j].value
    end
end
plot(sol_control.t/3600, plot_states[:,1]*R*T*100000/Pmeng, seriestype=:scatter)
plot(sol_control.t/3600, plot_states[:,2]*R*T*100000/Pmeng, seriestype=:scatter)
sol_control.u[end][2].value * sol_control.u[end][4].value - (sol_control.u[end][3].value)^2
sol_control.u[end][2].value * sol_control.u[end][4].value - (sol_control.u[end][3].value)^2


