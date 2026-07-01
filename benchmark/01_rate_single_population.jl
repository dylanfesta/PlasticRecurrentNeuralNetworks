#= 
📄 benchmark/01_rate_single_population.jl
⏳ 2026-07-1

Simple benchmark for a single excitatory population weakly connected

This file intentionally redefines each algorithm locally instead of calling
`PNN.forward_signal!`. That keeps the comparison readable even after the package
implementation changes.
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))
using PlasticRecurrentNeuralNetworks ; global const PNN=PlasticRecurrentNeuralNetworks
using Distributions
using Random
using Statistics
using LinearAlgebra

using BenchmarkTools
using Profile



##  ========== Parameters =========== ##
const n_exc = 4000
const dt = 1E-3
const Ttot = 5.0
const T_warmup = Ttot/2
const τ_exc = 100E-3
const dt_recorder = 50E-3
const μ_input = 5.0
const w_self_sum = 0.7
const w_noise_ratio = 0.1
const t_benchmark = 0.0

Random.seed!(1234)

##

const r_fix_expected = μ_input / (1.0-w_self_sum)

##

# excitatory type
e_type = PNN.ExcitatoryRateNeuron(τ_exc;rate_saturation=100.0)
# excitatory population
e_pop = PNN.LinearRateNeuralPopulation(e_type,n_exc;initial_rates=μ_input)
# recorder
rec = PNN.RCRate(e_pop,Ttot,dt_recorder)
# input (no noise)
inp = PNN.RateFixedInput(n_exc,μ_input)
# connection of input
connection_input = (e_pop,inp)

# now weight matrix, random
w_ee = fill(w_self_sum/(n_exc-1),n_exc,n_exc)
w_ee[diagind(w_ee)] .= 0.0
# add noise to weights
noise_std = w_noise_ratio * w_self_sum / (n_exc-1)
w_ee .+= rand(Normal(0.0,noise_std),n_exc,n_exc)
# make sure no negative weights
w_ee[w_ee .< 0.0] .= 0.0
synapse_self = PNN.RateLinearSynapses(w_ee)
connection_self = (e_pop, synapse_self, e_pop)
# network
ntw = PNN.RecurrentNetwork(populations=(e_pop,), 
  connections=(connection_input,connection_self), recorders=(rec,))

##

function reset_input!(rnp::PNN.RateNeuralPopulation)
  fill!(rnp.input_alloc,0.0)
  return nothing
end

# Naive row-copy dot product algorithm:
#   For each postsynaptic neuron i, take the row `weights[i, :]` and compute its
#   dot product with the presynaptic rates. In Julia this row slice allocates a
#   fresh vector, so the algorithm performs one matrix row copy per postsynaptic
#   neuron before doing the dot product.
function forward_signal_row_dot!(
    t_now::Float64,
    dt::Float64,
    rnp_post::PNN.RateNeuralPopulation,
    sy::PNN.RateLinearSynapses,
    rnp_pre::PNN.RateNeuralPopulation,
  )
  if isa(rnp_pre.neuron_type,PNN.ExcitatoryRateNeuron)
    @inbounds @simd for i in 1:sy.n_post
      rnp_post.input_alloc[i] += dot(sy.weights[i,:],rnp_pre.rates_now)
    end
  elseif isa(rnp_pre.neuron_type,PNN.InhibitoryRateNeuron)
    @inbounds @simd for i in 1:sy.n_post
      rnp_post.input_alloc[i] -= dot(sy.weights[i,:],rnp_pre.rates_now)
    end
  end

  return nothing
end

# SELECTED FOR PACKAGE:
#   Use the 5-argument `mul!` as the synaptic accumulator:
#     input_alloc = alpha * weights * rates_now + input_alloc
#   This preserves `forward_signal!` accumulation semantics and dispatches dense
#   matrix-vector multiplication to the optimized BLAS backend without a scratch
#   vector or per-row allocations.
function forward_signal_direct_mul!(
    t_now::Float64,
    dt::Float64,
    rnp_post::PNN.RateNeuralPopulation,
    sy::PNN.RateLinearSynapses,
    rnp_pre::PNN.RateNeuralPopulation,
  )
  if isa(rnp_pre.neuron_type,PNN.ExcitatoryRateNeuron)
    mul!(rnp_post.input_alloc,sy.weights,rnp_pre.rates_now,1.0,1.0)
  elseif isa(rnp_pre.neuron_type,PNN.InhibitoryRateNeuron)
    mul!(rnp_post.input_alloc,sy.weights,rnp_pre.rates_now,-1.0,1.0)
  end

  return nothing
end

reset_input!(e_pop)
forward_signal_row_dot!(t_benchmark,dt,e_pop,synapse_self,e_pop)
input_row_dot = copy(e_pop.input_alloc)
reset_input!(e_pop)
forward_signal_direct_mul!(t_benchmark,dt,e_pop,synapse_self,e_pop)
input_direct_mul = copy(e_pop.input_alloc)
@assert isapprox(input_direct_mul,input_row_dot;rtol=1e-12,atol=1e-12)

println("Benchmarking RateLinearSynapses forward_signal! with n_exc = ",n_exc)
println()

println("Naive row-copy dot product: one allocated row slice and dot per postsynaptic neuron")
row_dot_trial = @benchmark forward_signal_row_dot!($t_benchmark,$dt,$e_pop,$synapse_self,$e_pop) setup=(reset_input!($e_pop))
display(row_dot_trial)
println()

println("SELECTED FOR PACKAGE - direct mul! accumulator: BLAS matrix-vector multiply into input_alloc")
mul_trial = @benchmark forward_signal_direct_mul!($t_benchmark,$dt,$e_pop,$synapse_self,$e_pop) setup=(reset_input!($e_pop))
display(mul_trial)
println()

# function do_simulation()
#   PNN.reset!(rec)
#   global t_now = 0.0
#   while t_now <= Ttot
#     global t_now = PNN.dynamic_step!(t_now,dt,ntw)
#   end
# end

# @profview do_simulation()
##
