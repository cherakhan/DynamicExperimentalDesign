using OrdinaryDiffEq
using ForwardDiff: Dual
using LabelledArrays
using LinearAlgebra
using Plots
using NLopt
@inline function triangular_index(linear_index,square_size)
    for col_length in 1:square_size
        if linear_index <= col_length
            return linear_index,col_length
        end
        linear_index -= col_length
    end
end
@inline function labels(x::LArray{T,1,Labels}) where {T, Labels}
    return Labels
end
@inline function add_labels(state_labels,n_θ)
    for k in 1:(div(n_θ*(n_θ +1),2))
        label_index = triangular_index(k,n_θ)   
        new_label = :(Fim,$label_index)
        state_labels = (state_labels...,new_label)
    end
    return state_labels
end
function dynamic_system!(dx, x, p, t)
    p.dual_state_f!(dx, x, p, t)
    n_θ = length(p.θ)
    n_FIM = div(n_θ*(n_θ +1),2)
    n_dual_x = length(x) - n_FIM
    for linear_index in 1:n_FIM
        dx[n_dual_x+linear_index] = 0.0
        i,j = triangular_index(linear_index,n_θ )
        for s in 1:n_dual_x
            dx[n_dual_x+linear_index] += x[s].partials.values[i]*x[s].partials.values[j]
        end
    end
end
function condition_generator(tstops,parametrized_input)
    count::Int64 = 1
    function condition_closure(x,t,integrator)
        if  count < length(tstops) && t == tstops[count] 
            integrator.p.u[:] .= @view parametrized_input[count,:]
            count += 1
            return false
        end
        return false
    end
end
function affect(integrator)
    return nothing
end
abstract type ParSen end
struct Parameters{T, V<:AbstractVector{T}, W<:AbstractVector{<:Dual{ParSen,T}},U<:AbstractVector{T}}
    c::V
    θ::W
    u::U
    dual_state_f!::Function
end
function objective_generator(f!,x0,c,θ,u_example,tswitch)
    #time
    t0 = tswitch[1]
    te = tswitch[end]
    #dynamic system 

    #unknown parameters
    n_θ = length(θ)
    syms_θ = propertynames(θ)
    dual_θ = Vector{Dual{ParSen,eltype(θ),n_θ}}(undef, n_θ)
    self_sen = Matrix{Float64}(I, n_θ, n_θ)
    for i in 1:n_θ
        dual_θ[i] = Dual{ParSen}(θ[i], self_sen[i,:]...)
    end
    dual_θ= LArray{syms_θ}(dual_θ)
    #input parameters
    n_u = length(u_example)
    parameters = Parameters(c, dual_θ, u_example, f!)
    #state 
    n_dual_x = length(x0)
    n_x = n_dual_x +(div(n_θ*(n_θ +1),2))
    state_labels = labels(x0)
    state_labesl = add_labels(state_labels,n_θ)
    x0 = [x0; zeros(div(n_θ*(n_θ +1),2))]
    dual_x0 = Vector{Dual{ParSen,eltype(x0),n_θ}}(undef, n_x)
    for i in 1:length(x0)
        dual_x0[i] = Dual{ParSen}(x0[i], zeros(n_θ)...)
    end
    dual_x0 = LArray{state_labels}(dual_x0)
    
    function matrix_objective(inputs)
        parameters.u[:] .= @view inputs[1,:]
        ini_val_prob = ODEProblem(dynamic_system!, dual_x0, (t0, te), parameters)
        tswitch_view = @view tswitch[2:end-1]
        input_view = @view inputs[2:end,:]
        condition = condition_generator(tswitch_view,input_view)
        cb = DiscreteCallback(condition, affect,save_positions = (false,false))
        cbs = CallbackSet(cb)
        sol_ini_val = solve(ini_val_prob,Tsit5(), callback = cbs, tstops = tswitch)
    end    
    return function vector_objective(vector_inputs,grad)
        matrix_inputs = reshape(vector_inputs,:,n_u)
        return matrix_objective(matrix_inputs)
    end
end

