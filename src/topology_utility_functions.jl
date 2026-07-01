#=
Component of PlasticRecurrentNeuralNetworks.jl package with utility functions for generating topologies of neural populations and connections.

=#

"""
  place_neurons_on_ring(n_neurons; offset=0.0)

  Generate locations in radians for `n_neurons` uniformly distributed on a ring.
  The first neuron is placed at `offset`, and the rest are placed at equal
  angular intervals around the ring.
"""
function place_neurons_on_ring(n_neurons::Int64;offset::Float64=0.0)
  locations = mod.(offset .+ collect(range(0,2π,length=n_neurons+1)[1:end-1]),2π)
  return locations
end


"""
  generate_ring_topology(n_post, n_pre; scale_factor=1.0, kappa=1.0)

Generate a distance-based scaling factor matrix for a one-dimensional ring topology.

Neurons are arranged uniformly on a ring. Returns
`(locations_post, locations_pre, A)`, where `A` has dimensions `(n_post, n_pre)`
and follows the package convention `post <- pre`.

The scale factor uses a Von Mises kernel,
`exp(kappa * cos(θ_post - θ_pre))`, normalized by its peak value so the maximum
possible scaling factor is `scale_factor`.
Higher kappa values result in a narrower kernel, while lower kappa values result in a wider kernel.
"""
function generate_ring_topology(n_post::Int64,n_pre::Int64;
    scale_factor::Float64=1.0,kappa::Float64=1.0,
    zero_cutoff_value::Float64=1E-6)
  A = zeros(Float64,n_post,n_pre)
  locations_pre = place_neurons_on_ring(n_pre)
  locations_post = place_neurons_on_ring(n_post)
  for i in 1:n_post
    θ_post = locations_post[i]
    for j in 1:n_pre
      θ_pre = locations_pre[j]
      A[i,j] = scale_factor * exp(kappa * (cos(θ_post - θ_pre) - 1))
    end
  end
  # if near zero, set to zero
  A[A .< zero_cutoff_value] .= 0.0
  return locations_post, locations_pre, A
end

# utility function for self-connected population
generate_ring_topology(n_neurons::Int64;scale_factor::Float64=1.0,kappa::Float64=1.0) = generate_ring_topology(n_neurons,n_neurons;scale_factor=scale_factor,kappa=kappa)
