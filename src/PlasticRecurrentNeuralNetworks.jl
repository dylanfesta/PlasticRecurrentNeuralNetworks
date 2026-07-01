module PlasticRecurrentNeuralNetworks

# dependencies
using LinearAlgebra,Statistics,StatsBase,Random,Distributions
using SparseArrays

#=

## Composition of the network:
 - Neuron type : parameters of single neuron
 - Neuron population : population of neurons of same type, allocations of internal states
 - Inputs : can be seen as a special kind of neural population, 
            that only sends signal and does not have an internal state
 - Connections : tuples of (population_post,synapse_post_pre,population_pre), or (population_post,input)
 - Recorders: stores population internal states, or synaptic weights and parameters
 

## Simulation loop
- 1. clean up internal states
- 2. forward signals
- 3. local updates
- 4. plasticity
- 5. recorders

=#

# abstract type declarations

# type is for E/I, conductance/current based, internal parameters, etc.
abstract type NeuronType end  # NT

# populations contain number of neurons and allocations
abstract type NeuralPopulation end  #  NP

# input goes directly into neuron and is not plastic
# (but of course can be changed explicitly)
# therefore there only input type which incorporates population, and interacts directly to neural population
abstract type Input end  # IN

# synapses connect populations, have weights and support plasticity rules
abstract type Synapses end # SY

# Weights are sparse or dense matrices
abstract type Weights end # WG

# plasticity rules
abstract type Plasticity end # PR

# recorders
abstract type Recorder end # RC

include("topology_utility_functions.jl")

include("rate_models.jl")

# WARNING: currently focusing on rate models and rate plasticity only
# IF neurons will be implemented later

# integrate and fire neuron and synapse models
# include("if_neuron.jl")

# include("simple_inputs.jl")

# # poisson input neurons
# include("poisson_inputs.jl")

# # recorders
# include("recorders.jl")

# big ugly type
# population is a tuple of all populations with internal states that need a clean up and a local update
# connections is a tuple of tuples in the form (population_post,synapse_post_pre,population_pre),
# or (population_post,input)
# where I call forward_signal!(t_now,dt,population_post,synapse_post_pre,population_pre)
# recorders is a tuple of all recorders, where I call record!(t_now,recorder)

Base.@kwdef struct RecurrentNetwork
  populations::Tuple        = ()
  connections::Tuple        = ()
  plasticity_rules::Tuple   = ()
  recorders::Tuple          = ()
end

# (you may not like it, but it is what peak code optimization looks like :-P )
# clean up: sets inputs to zero when needed
function call_clean_up!(populations)
  clean_up!(first(populations))
  call_clean_up!(Base.tail(populations))
  return nothing
end
function call_clean_up!(::Tuple{})
  return nothing
end
# forward signals
function call_forward_signal!(t_now,dt,connections)
  forward_signal!(t_now,dt,first(connections)...)
  call_forward_signal!(t_now,dt,Base.tail(connections))
  return nothing
end
function call_forward_signal!(::Float64,::Float64,::Tuple{},args...)
  return nothing
end

# local updates
function call_local_update!(t_now,dt,populations)
  local_update!(t_now,dt,first(populations))
  call_local_update!(t_now,dt,Base.tail(populations))
  return nothing
end
function call_local_update!(::Float64,::Float64,::Tuple{},args...)
  return nothing
end

# plasticity rules
# like for recorders, I assume plasticity rules have pointers to populations and synapses inside of them
function call_plasticity!(t_now,dt,plasticity_rules)
  plasticity!(t_now,dt,first(plasticity_rules))
  call_plasticity!(t_now,dt,Base.tail(plasticity_rules))
  return nothing
end
function call_plasticity!(::Float64,::Float64,::Tuple{},args...)
  return nothing
end


# recorders
function call_recorders!(t_now,recorders)
  record!(t_now,first(recorders))
  call_recorders!(t_now,Base.tail(recorders))
  return nothing
end
function call_recorders!(::Float64,::Tuple{},args...)
  return nothing
end 


function dynamic_step!(t_now::Float64,dt::Float64,rn::RecurrentNetwork)
  call_clean_up!(rn.populations) # clean up inputs
  call_forward_signal!(t_now,dt,rn.connections) # forward signals
  call_local_update!(t_now,dt,rn.populations) # local updates
  call_plasticity!(t_now,dt,rn.plasticity_rules) # plasticity
  call_recorders!(t_now,rn.recorders) # recorders
  return t_now + dt
end


end # of module
