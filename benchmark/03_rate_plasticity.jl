#=
📄 benchmark/03_rate_plasticity.jl
⏳ 2026-07-1

Simple benchmark for rate plasticity rules

This file intentionally redefines each algorithm locally instead of calling the
package's `plasticity!`. That keeps the benchmark readable and prevents the
current package implementation from hiding what is being measured.
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))
using PlasticRecurrentNeuralNetworks ; global const PNN=PlasticRecurrentNeuralNetworks
using Random

using BenchmarkTools



##  ========== Parameters =========== ##
const n_post = 500
const n_pre = 500
const dt_rate = 1E-3
const Δt_plasticity = 10E-3
const learning_rate = 0.2
const α_homeostatic = 0.8
const α_covariance = 0.05
const α1_quadratic = -0.4
const α2_quadratic = 0.1
const w_min = 1E-8
const w_max = 5.0
const t_benchmark = 0.0

Random.seed!(1234)

##

post_type = PNN.ExcitatoryRateNeuron(100E-3;rate_saturation=100.0)
pre_type = PNN.ExcitatoryRateNeuron(100E-3;rate_saturation=100.0)

post_population = PNN.LinearRateNeuralPopulation(
  post_type,
  n_post;
  initial_rates=rand(n_post),
)
pre_population = PNN.LinearRateNeuralPopulation(
  pre_type,
  n_pre;
  initial_rates=rand(n_pre),
)

post_mean_estimator = PNN.RateMeanEstimator(
  post_population,
  100E-3,
  Δt_plasticity;
  initial_mean=rand(n_post),
)
pre_mean_estimator = PNN.RateMeanEstimator(pre_population,100E-3,Δt_plasticity)
covariance_estimator = PNN.RateCovarianceEstimator(post_mean_estimator,pre_mean_estimator)
covariance_estimator.covariance_now .= 0.2 .* randn(n_post,n_pre)

initial_weights = 0.1 .+ rand(n_post,n_pre)
initial_weights[rand(n_post,n_pre) .< 0.1] .= 0.0
scale_matrix = rand(n_post,n_pre)
scale_matrix[rand(n_post,n_pre) .< 0.1] .= 0.0

homeostatic_synapse = PNN.RateLinearSynapses(copy(initial_weights))
covariance_synapse = PNN.RateLinearSynapses(copy(initial_weights))
scaled_covariance_synapse = PNN.RateLinearSynapses(copy(initial_weights))
quadratic_synapse = PNN.RateLinearSynapses(copy(initial_weights))

homeostatic_rule = PNN.RatePlasticityHomeostaticScaling(
  post_population,
  homeostatic_synapse,
  pre_population,
  α_homeostatic,
  Δt_plasticity,
  learning_rate,
  post_mean_estimator;
  w_min=w_min,
  w_max=w_max,
)

covariance_rule = PNN.RatePlasticityCovariance(
  post_population,
  covariance_synapse,
  pre_population,
  α_covariance,
  Δt_plasticity,
  learning_rate,
  covariance_estimator;
  w_min=w_min,
  w_max=w_max,
)

scaled_covariance_rule = PNN.RatePlasticityScaledCovariance(
  post_population,
  scaled_covariance_synapse,
  pre_population,
  scale_matrix,
  α_covariance,
  Δt_plasticity,
  learning_rate,
  covariance_estimator;
  w_min=w_min,
  w_max=w_max,
)

quadratic_rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
  post_population,
  quadratic_synapse,
  pre_population,
  Δt_plasticity,
  α1_quadratic,
  α2_quadratic,
  learning_rate,
  covariance_estimator;
  w_min=w_min,
  w_max=w_max,
)

##

function reset_rule_state!(rule,initial_weights::Matrix{Float64})
  copy!(rule.synapses_post_pre.weights,initial_weights)
  rule.t_last_update = -Inf
  return nothing
end

# Naive homeostatic scaling:
#   For every weight W[post, pre], repeatedly read through the rule object to get
#   postsynaptic rate and mean state. Only positive weights are plastic. The
#   update is multiplicative in the old weight:
#     W <- clamp(W + ηΔt * W * r_post * (α - mean_post)).
function plasticity_homeostatic_naive!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityHomeostaticScaling,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      if w_old == 0.0
        continue
      end
      r_post = rule.pop_post.rates_now[i]
      r_post_mean = rule.rate_estimator_post.mean_now[i]
      w_new = w_old + effective_learning_rate * w_old * r_post * (rule.α - r_post_mean)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier homeostatic kernel. The wrapper extracts concrete arrays
#   once, then this kernel performs one column-major pass over the weight matrix.
function update_homeostatic_kernel!(
    weights::Matrix{Float64},
    post_rates::Vector{Float64},
    post_means::Vector{Float64},
    α::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64,
  )
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * w_old * post_rates[i] * (α - post_means[i])
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function plasticity_homeostatic_kernel!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityHomeostaticScaling,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  rule.t_last_update = t_now
  update_homeostatic_kernel!(
    rule.synapses_post_pre.weights,
    rule.pop_post.rates_now,
    rule.rate_estimator_post.mean_now,
    rule.α,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

# Naive covariance rule:
#   Add an affine function of covariance to each positive weight:
#     W <- clamp(W + ηΔt * (C - α)).
#   This version reads the weights and covariance through the rule object inside
#   the nested loop.
function plasticity_covariance_naive!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityCovariance,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  covariance_now = rule.covariance_estimator.covariance_now
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (covariance_now[i,j] - rule.α)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier covariance kernel. One pass over concrete weight and
#   covariance matrices, with no scratch matrix and no package helper calls.
function update_covariance_plasticity_kernel!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    α::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64,
  )
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (covariance_now[i,j] - α)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function plasticity_covariance_kernel!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityCovariance,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  rule.t_last_update = t_now
  update_covariance_plasticity_kernel!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator.covariance_now,
    rule.α,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

# Naive scaled covariance rule:
#   Same as covariance plasticity, but the update is multiplied entrywise by a
#   scale matrix A[post, pre]. Entries with zero weight or zero scale are skipped.
function plasticity_scaled_covariance_naive!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityScaledCovariance,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  covariance_now = rule.covariance_estimator.covariance_now
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      scale = rule.scale_matrix[i,j]
      if (w_old == 0.0) || (scale == 0.0)
        continue
      end
      w_new = w_old + effective_learning_rate * scale * (covariance_now[i,j] - rule.α)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier scaled covariance kernel. The hot loop receives concrete
#   weights, covariance, and scale matrices and performs one fused pass.
function update_scaled_covariance_plasticity_kernel!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    scale_matrix::Matrix{Float64},
    α::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64,
  )
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      scale = scale_matrix[i,j]
      if (w_old == 0.0) || (scale == 0.0)
        continue
      end
      w_new = w_old + effective_learning_rate * scale * (covariance_now[i,j] - α)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function plasticity_scaled_covariance_kernel!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityScaledCovariance,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  rule.t_last_update = t_now
  update_scaled_covariance_plasticity_kernel!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator.covariance_now,
    rule.scale_matrix,
    rule.α,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

# Naive quadratically stabilized covariance rule:
#   Updates positive weights with a covariance term plus a quadratic weight term:
#     W <- clamp(W + ηΔt * (α1 * C + α2 * W^2)).
function plasticity_quadratic_naive!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityCovarianceQuadraticallyStabilized,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  covariance_now = rule.covariance_estimator.covariance_now
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate *
        (rule.α1 * covariance_now[i,j] + rule.α2 * w_old^2)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier quadratic kernel. Keeps the nonlinear update fused in one
#   concrete-array pass and avoids repeated abstract field access.
function update_quadratic_plasticity_kernel!(
    weights::Matrix{Float64},
    covariance_now::Matrix{Float64},
    α1::Float64,
    α2::Float64,
    effective_learning_rate::Float64,
    w_min::Float64,
    w_max::Float64,
  )
  n_post,n_pre = size(weights)
  @inbounds for j in 1:n_pre
    for i in 1:n_post
      w_old = weights[i,j]
      if w_old == 0.0
        continue
      end
      w_new = w_old + effective_learning_rate * (α1 * covariance_now[i,j] + α2 * w_old^2)
      weights[i,j] = clamp(w_new,w_min,w_max)
    end
  end
  return nothing
end

function plasticity_quadratic_kernel!(
    t_now::Float64,
    dt::Float64,
    rule::PNN.RatePlasticityCovarianceQuadraticallyStabilized,
  )
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  rule.t_last_update = t_now
  update_quadratic_plasticity_kernel!(
    rule.synapses_post_pre.weights,
    rule.covariance_estimator.covariance_now,
    rule.α1,
    rule.α2,
    rule.learning_rate * rule.Δt,
    rule.w_min,
    rule.w_max,
  )
  return nothing
end

##

reset_rule_state!(homeostatic_rule,initial_weights)
plasticity_homeostatic_naive!(t_benchmark,dt_rate,homeostatic_rule)
homeostatic_naive_weights = copy(homeostatic_rule.synapses_post_pre.weights)
reset_rule_state!(homeostatic_rule,initial_weights)
plasticity_homeostatic_kernel!(t_benchmark,dt_rate,homeostatic_rule)
@assert isapprox(homeostatic_rule.synapses_post_pre.weights,homeostatic_naive_weights;rtol=1e-12,atol=1e-12)

reset_rule_state!(covariance_rule,initial_weights)
plasticity_covariance_naive!(t_benchmark,dt_rate,covariance_rule)
covariance_naive_weights = copy(covariance_rule.synapses_post_pre.weights)
reset_rule_state!(covariance_rule,initial_weights)
plasticity_covariance_kernel!(t_benchmark,dt_rate,covariance_rule)
@assert isapprox(covariance_rule.synapses_post_pre.weights,covariance_naive_weights;rtol=1e-12,atol=1e-12)

reset_rule_state!(scaled_covariance_rule,initial_weights)
plasticity_scaled_covariance_naive!(t_benchmark,dt_rate,scaled_covariance_rule)
scaled_covariance_naive_weights = copy(scaled_covariance_rule.synapses_post_pre.weights)
reset_rule_state!(scaled_covariance_rule,initial_weights)
plasticity_scaled_covariance_kernel!(t_benchmark,dt_rate,scaled_covariance_rule)
@assert isapprox(scaled_covariance_rule.synapses_post_pre.weights,scaled_covariance_naive_weights;rtol=1e-12,atol=1e-12)

reset_rule_state!(quadratic_rule,initial_weights)
plasticity_quadratic_naive!(t_benchmark,dt_rate,quadratic_rule)
quadratic_naive_weights = copy(quadratic_rule.synapses_post_pre.weights)
reset_rule_state!(quadratic_rule,initial_weights)
plasticity_quadratic_kernel!(t_benchmark,dt_rate,quadratic_rule)
@assert isapprox(quadratic_rule.synapses_post_pre.weights,quadratic_naive_weights;rtol=1e-12,atol=1e-12)

println("Benchmarking rate plasticity rules with n_post = ",n_post,", n_pre = ",n_pre)
println()

println("Homeostatic scaling - naive object-field loop")
homeostatic_naive_trial = @benchmark plasticity_homeostatic_naive!($t_benchmark,$dt_rate,$homeostatic_rule) setup=(reset_rule_state!($homeostatic_rule,$initial_weights)) evals=1
display(homeostatic_naive_trial)
println()

println("SELECTED FOR PACKAGE - homeostatic scaling function-barrier kernel")
homeostatic_kernel_trial = @benchmark plasticity_homeostatic_kernel!($t_benchmark,$dt_rate,$homeostatic_rule) setup=(reset_rule_state!($homeostatic_rule,$initial_weights)) evals=1
display(homeostatic_kernel_trial)
println()

println("Covariance plasticity - naive object-field loop")
covariance_naive_trial = @benchmark plasticity_covariance_naive!($t_benchmark,$dt_rate,$covariance_rule) setup=(reset_rule_state!($covariance_rule,$initial_weights)) evals=1
display(covariance_naive_trial)
println()

println("SELECTED FOR PACKAGE - covariance plasticity function-barrier kernel")
covariance_kernel_trial = @benchmark plasticity_covariance_kernel!($t_benchmark,$dt_rate,$covariance_rule) setup=(reset_rule_state!($covariance_rule,$initial_weights)) evals=1
display(covariance_kernel_trial)
println()

println("Scaled covariance plasticity - naive object-field loop")
scaled_covariance_naive_trial = @benchmark plasticity_scaled_covariance_naive!($t_benchmark,$dt_rate,$scaled_covariance_rule) setup=(reset_rule_state!($scaled_covariance_rule,$initial_weights)) evals=1
display(scaled_covariance_naive_trial)
println()

println("SELECTED FOR PACKAGE - scaled covariance plasticity function-barrier kernel")
scaled_covariance_kernel_trial = @benchmark plasticity_scaled_covariance_kernel!($t_benchmark,$dt_rate,$scaled_covariance_rule) setup=(reset_rule_state!($scaled_covariance_rule,$initial_weights)) evals=1
display(scaled_covariance_kernel_trial)
println()

println("Quadratically stabilized covariance plasticity - naive object-field loop")
quadratic_naive_trial = @benchmark plasticity_quadratic_naive!($t_benchmark,$dt_rate,$quadratic_rule) setup=(reset_rule_state!($quadratic_rule,$initial_weights)) evals=1
display(quadratic_naive_trial)
println()

println("SELECTED FOR PACKAGE - quadratically stabilized covariance plasticity function-barrier kernel")
quadratic_kernel_trial = @benchmark plasticity_quadratic_kernel!($t_benchmark,$dt_rate,$quadratic_rule) setup=(reset_rule_state!($quadratic_rule,$initial_weights)) evals=1
display(quadratic_kernel_trial)
println()
