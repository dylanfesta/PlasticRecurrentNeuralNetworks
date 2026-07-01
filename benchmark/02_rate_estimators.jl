#=
📄 benchmark/02_rate_estimators.jl
⏳ 2026-07-1

Simple benchmark for rate mean and covariance estimators

This file intentionally redefines each algorithm locally instead of calling the
package's `local_update!`. That keeps the benchmark readable and prevents the
current package implementation from hiding what is being measured.
=#

push!(LOAD_PATH, abspath(@__DIR__,".."))
using PlasticRecurrentNeuralNetworks ; global const PNN=PlasticRecurrentNeuralNetworks
using Random
using LinearAlgebra

using BenchmarkTools



##  ========== Parameters =========== ##
const n_post = 1000
const n_pre = 1000
const dt_rate = 1E-3
const dt_trace = 10E-3
const τ_trace = 100E-3
const t_benchmark = 0.0

Random.seed!(1234)

##

post_type = PNN.ExcitatoryRateNeuron(τ_trace;rate_saturation=100.0)
pre_type = PNN.ExcitatoryRateNeuron(τ_trace;rate_saturation=100.0)

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

post_mean_estimator = PNN.RateMeanEstimator(post_population,τ_trace,dt_trace)
pre_mean_estimator = PNN.RateMeanEstimator(pre_population,τ_trace,dt_trace)
covariance_estimator = PNN.RateCovarianceEstimator(post_mean_estimator,pre_mean_estimator)

initial_post_mean = rand(n_post)
initial_pre_mean = rand(n_pre)
initial_second_moment = rand(n_post,n_pre)
initial_covariance = rand(n_post,n_pre)

##

function reset_mean_state!(est::PNN.RateMeanEstimator,initial_mean::Vector{Float64})
  copy!(est.mean_now,initial_mean)
  est.t_last_update = -Inf
  return nothing
end

function reset_covariance_state!(
    est::PNN.RateCovarianceEstimator,
    initial_second_moment::Matrix{Float64},
    initial_covariance::Matrix{Float64},
  )
  copy!(est.second_moment_now,initial_second_moment)
  copy!(est.covariance_now,initial_covariance)
  est.t_last_update = -Inf
  return nothing
end

# Naive mean-estimator loop:
#   This is the direct scalar expression of
#     mean = a * mean + (1 - a) * rate
#   inside the estimator object. It is simple, but accessing rate arrays through
#   abstractly typed population fields in the hot loop can force dynamic dispatch
#   and many small allocations.
function local_update_mean_naive_loop!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateMeanEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  @inbounds @simd for i in 1:est.n
    est.mean_now[i] = a * est.mean_now[i] + (1.0 - a) * est.pop.rates_now[i]
  end
  return nothing
end

# Broadcast mean update:
#   Expresses the same elementwise recurrence as a fused broadcast. This can
#   avoid the type-instability of the naive loop, but it is less explicit and
#   still creates a small amount of broadcast machinery.
function local_update_mean_broadcast!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateMeanEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a
  @. est.mean_now = a * est.mean_now + input_factor * est.pop.rates_now
  return nothing
end

# Cached-loop mean update:
#   Pulls arrays and constants into local variables before the scalar loop. This
#   is clearer than the naive loop, but by itself it does not fully solve the
#   type-instability caused by the abstract population field.
function local_update_mean_cached_loop!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateMeanEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a
  mean_now = est.mean_now
  rates_now = est.pop.rates_now
  @inbounds @simd for i in 1:est.n
    mean_now[i] = a * mean_now[i] + input_factor * rates_now[i]
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier kernel for the mean estimator. The wrapper extracts concrete
#   arrays once, then this kernel receives `Vector{Float64}` arguments and runs a
#   zero-allocation tight loop.
function update_mean_kernel!(
    mean_now::Vector{Float64},
    rates_now::Vector{Float64},
    a::Float64,
  )
  input_factor = 1.0 - a
  @inbounds @simd for i in eachindex(mean_now,rates_now)
    mean_now[i] = a * mean_now[i] + input_factor * rates_now[i]
  end
  return nothing
end

# Wrapper used by the selected mean-estimator algorithm. It mirrors the public
# estimator update policy, but the numerical work is delegated to the concrete
# array kernel above.
function local_update_mean_kernel!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateMeanEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  update_mean_kernel!(est.mean_now,est.pop.rates_now,est.propagation_factor)
  return nothing
end

# Naive covariance-estimator loop:
#   Fuses the second-moment update and covariance update in one pass over the
#   column-major matrices. The math is good, but repeated abstract field access
#   inside the nested loop can dominate runtime and allocation behavior.
function local_update_covariance_naive_loop!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateCovarianceEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a

  @inbounds for j in 1:est.n_pre
    pre_rate = est.pop_pre.rates_now[j]
    pre_mean = est.mean_pre_estimator.mean_now[j]
    for i in 1:est.n_post
      second_moment = a * est.second_moment_now[i,j] +
        input_factor * est.pop_post.rates_now[i] * pre_rate
      est.second_moment_now[i,j] = second_moment
      est.covariance_now[i,j] = second_moment - est.mean_post_estimator.mean_now[i] * pre_mean
    end
  end
  return nothing
end

# BLAS outer-product covariance update:
#   Updates Q with `rmul!` plus `BLAS.ger!`, then copies Q into C and subtracts
#   the mean outer product with another `ger!`. It is allocation-light and uses
#   BLAS, but requires multiple full matrix passes.
function local_update_covariance_blas!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateCovarianceEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a

  rmul!(est.second_moment_now,a)
  BLAS.ger!(
    input_factor,
    est.pop_post.rates_now,
    est.pop_pre.rates_now,
    est.second_moment_now,
  )

  copy!(est.covariance_now,est.second_moment_now)
  BLAS.ger!(
    -1.0,
    est.mean_post_estimator.mean_now,
    est.mean_pre_estimator.mean_now,
    est.covariance_now,
  )

  return nothing
end

# Cached-loop covariance update:
#   Same fused scalar algorithm as the naive version, with arrays cached in
#   local variables. This improves readability, but does not fully remove the
#   inference problem at the benchmark wrapper boundary.
function local_update_covariance_cached_loop!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateCovarianceEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a
  second_moment_now = est.second_moment_now
  covariance_now = est.covariance_now
  post_rates = est.pop_post.rates_now
  pre_rates = est.pop_pre.rates_now
  post_means = est.mean_post_estimator.mean_now
  pre_means = est.mean_pre_estimator.mean_now

  @inbounds for j in 1:est.n_pre
    pre_rate = pre_rates[j]
    pre_mean = pre_means[j]
    for i in 1:est.n_post
      second_moment = a * second_moment_now[i,j] +
        input_factor * post_rates[i] * pre_rate
      second_moment_now[i,j] = second_moment
      covariance_now[i,j] = second_moment - post_means[i] * pre_mean
    end
  end
  return nothing
end

# SELECTED FOR PACKAGE:
#   Function-barrier kernel for covariance. It keeps the best algorithmic shape:
#   one fused pass over `second_moment_now` and `covariance_now`, column-major
#   traversal, no scratch matrices, and no BLAS copy/update split.
function update_covariance_kernel!(
    second_moment_now::Matrix{Float64},
    covariance_now::Matrix{Float64},
    post_rates::Vector{Float64},
    pre_rates::Vector{Float64},
    post_means::Vector{Float64},
    pre_means::Vector{Float64},
    a::Float64,
  )
  input_factor = 1.0 - a
  n_post,n_pre = size(second_moment_now)
  @inbounds for j in 1:n_pre
    pre_rate = pre_rates[j]
    pre_mean = pre_means[j]
    for i in 1:n_post
      second_moment = a * second_moment_now[i,j] +
        input_factor * post_rates[i] * pre_rate
      second_moment_now[i,j] = second_moment
      covariance_now[i,j] = second_moment - post_means[i] * pre_mean
    end
  end
  return nothing
end

# Wrapper used by the selected covariance-estimator algorithm. It performs the
# estimator time gate, then passes concrete arrays to the selected kernel.
function local_update_covariance_kernel!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateCovarianceEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  update_covariance_kernel!(
    est.second_moment_now,
    est.covariance_now,
    est.pop_post.rates_now,
    est.pop_pre.rates_now,
    est.mean_post_estimator.mean_now,
    est.mean_pre_estimator.mean_now,
    est.propagation_factor,
  )
  return nothing
end

# Broadcast covariance update:
#   Uses reshaped vector views to express two outer-product broadcasts. This is
#   compact and allocation-light, but it performs separate full-matrix passes for
#   Q and C and is typically slower than the selected fused kernel.
function local_update_covariance_broadcast!(
    t_now::Float64,
    dt::Float64,
    est::PNN.RateCovarianceEstimator,
  )
  if t_now - est.t_last_update < est.dt
    return nothing
  end

  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1.0 - a
  post_rates = reshape(est.pop_post.rates_now,:,1)
  pre_rates = reshape(est.pop_pre.rates_now,1,:)
  post_means = reshape(est.mean_post_estimator.mean_now,:,1)
  pre_means = reshape(est.mean_pre_estimator.mean_now,1,:)

  @. est.second_moment_now =
    a * est.second_moment_now +
    input_factor * post_rates * pre_rates
  @. est.covariance_now =
    est.second_moment_now -
    post_means * pre_means
  return nothing
end

##

reset_mean_state!(post_mean_estimator,initial_post_mean)
local_update_mean_naive_loop!(t_benchmark,dt_rate,post_mean_estimator)
mean_normal = copy(post_mean_estimator.mean_now)

reset_mean_state!(post_mean_estimator,initial_post_mean)
local_update_mean_broadcast!(t_benchmark,dt_rate,post_mean_estimator)
mean_broadcast = copy(post_mean_estimator.mean_now)
@assert isapprox(mean_broadcast,mean_normal;rtol=1e-12,atol=1e-12)

reset_mean_state!(post_mean_estimator,initial_post_mean)
local_update_mean_cached_loop!(t_benchmark,dt_rate,post_mean_estimator)
mean_cached_loop = copy(post_mean_estimator.mean_now)
@assert isapprox(mean_cached_loop,mean_normal;rtol=1e-12,atol=1e-12)

reset_mean_state!(post_mean_estimator,initial_post_mean)
local_update_mean_kernel!(t_benchmark,dt_rate,post_mean_estimator)
mean_kernel = copy(post_mean_estimator.mean_now)
@assert isapprox(mean_kernel,mean_normal;rtol=1e-12,atol=1e-12)

reset_mean_state!(post_mean_estimator,initial_post_mean)
reset_mean_state!(pre_mean_estimator,initial_pre_mean)
reset_covariance_state!(covariance_estimator,initial_second_moment,initial_covariance)
local_update_covariance_naive_loop!(t_benchmark,dt_rate,covariance_estimator)
second_moment_normal = copy(covariance_estimator.second_moment_now)
covariance_normal = copy(covariance_estimator.covariance_now)

reset_covariance_state!(covariance_estimator,initial_second_moment,initial_covariance)
local_update_covariance_blas!(t_benchmark,dt_rate,covariance_estimator)
second_moment_blas = copy(covariance_estimator.second_moment_now)
covariance_blas = copy(covariance_estimator.covariance_now)
@assert isapprox(second_moment_blas,second_moment_normal;rtol=1e-12,atol=1e-12)
@assert isapprox(covariance_blas,covariance_normal;rtol=1e-12,atol=1e-12)

reset_covariance_state!(covariance_estimator,initial_second_moment,initial_covariance)
local_update_covariance_cached_loop!(t_benchmark,dt_rate,covariance_estimator)
second_moment_cached_loop = copy(covariance_estimator.second_moment_now)
covariance_cached_loop = copy(covariance_estimator.covariance_now)
@assert isapprox(second_moment_cached_loop,second_moment_normal;rtol=1e-12,atol=1e-12)
@assert isapprox(covariance_cached_loop,covariance_normal;rtol=1e-12,atol=1e-12)

reset_covariance_state!(covariance_estimator,initial_second_moment,initial_covariance)
local_update_covariance_kernel!(t_benchmark,dt_rate,covariance_estimator)
second_moment_kernel = copy(covariance_estimator.second_moment_now)
covariance_kernel = copy(covariance_estimator.covariance_now)
@assert isapprox(second_moment_kernel,second_moment_normal;rtol=1e-12,atol=1e-12)
@assert isapprox(covariance_kernel,covariance_normal;rtol=1e-12,atol=1e-12)

reset_covariance_state!(covariance_estimator,initial_second_moment,initial_covariance)
local_update_covariance_broadcast!(t_benchmark,dt_rate,covariance_estimator)
second_moment_broadcast = copy(covariance_estimator.second_moment_now)
covariance_broadcast = copy(covariance_estimator.covariance_now)
@assert isapprox(second_moment_broadcast,second_moment_normal;rtol=1e-12,atol=1e-12)
@assert isapprox(covariance_broadcast,covariance_normal;rtol=1e-12,atol=1e-12)

println("Benchmarking RateMeanEstimator with n_post = ",n_post)
println()

println("Naive scalar loop: direct field access inside the hot loop")
mean_normal_trial = @benchmark local_update_mean_naive_loop!($t_benchmark,$dt_rate,$post_mean_estimator) setup=(reset_mean_state!($post_mean_estimator,$initial_post_mean)) evals=1
display(mean_normal_trial)
println()

println("Broadcast expression: fused elementwise recurrence")
mean_broadcast_trial = @benchmark local_update_mean_broadcast!($t_benchmark,$dt_rate,$post_mean_estimator) setup=(reset_mean_state!($post_mean_estimator,$initial_post_mean)) evals=1
display(mean_broadcast_trial)
println()

println("Cached scalar loop: local array bindings before the loop")
mean_cached_loop_trial = @benchmark local_update_mean_cached_loop!($t_benchmark,$dt_rate,$post_mean_estimator) setup=(reset_mean_state!($post_mean_estimator,$initial_post_mean)) evals=1
display(mean_cached_loop_trial)
println()

println("SELECTED FOR PACKAGE - function-barrier loop: concrete Vector{Float64} kernel")
mean_kernel_trial = @benchmark local_update_mean_kernel!($t_benchmark,$dt_rate,$post_mean_estimator) setup=(reset_mean_state!($post_mean_estimator,$initial_post_mean)) evals=1
display(mean_kernel_trial)
println()

println("Benchmarking RateCovarianceEstimator with n_post = ",n_post,", n_pre = ",n_pre)
println()

println("Naive fused loop: direct field access inside nested loops")
covariance_normal_trial = @benchmark local_update_covariance_naive_loop!($t_benchmark,$dt_rate,$covariance_estimator) setup=(reset_covariance_state!($covariance_estimator,$initial_second_moment,$initial_covariance)) evals=1
display(covariance_normal_trial)
println()

println("BLAS ger!/copy: two outer-product updates plus a full matrix copy")
covariance_blas_trial = @benchmark local_update_covariance_blas!($t_benchmark,$dt_rate,$covariance_estimator) setup=(reset_covariance_state!($covariance_estimator,$initial_second_moment,$initial_covariance)) evals=1
display(covariance_blas_trial)
println()

println("Cached fused loop: local array bindings before nested loops")
covariance_cached_loop_trial = @benchmark local_update_covariance_cached_loop!($t_benchmark,$dt_rate,$covariance_estimator) setup=(reset_covariance_state!($covariance_estimator,$initial_second_moment,$initial_covariance)) evals=1
display(covariance_cached_loop_trial)
println()

println("SELECTED FOR PACKAGE - function-barrier fused loop: one concrete-array pass, no scratch matrix")
covariance_kernel_trial = @benchmark local_update_covariance_kernel!($t_benchmark,$dt_rate,$covariance_estimator) setup=(reset_covariance_state!($covariance_estimator,$initial_second_moment,$initial_covariance)) evals=1
display(covariance_kernel_trial)
println()

println("Broadcast outer products: allocation-light but separate matrix passes")
covariance_broadcast_trial = @benchmark local_update_covariance_broadcast!($t_benchmark,$dt_rate,$covariance_estimator) setup=(reset_covariance_state!($covariance_estimator,$initial_second_moment,$initial_covariance)) evals=1
display(covariance_broadcast_trial)
println()
