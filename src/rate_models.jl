#=
Component of PlasticRecurrentNeuralNetworks.jl package for the simulation of rate-based ReLu models

=#


"""
    RateNeuronType

Abstract parent for rate-neuron parameter types.

Concrete subtypes store intrinsic neuron parameters, such as the membrane/rate
time constant `τ`, and are carried by `RateNeuralPopulation`s.
"""
abstract type RateNeuronType <: NeuronType end  # RNT

"""
    RateNeuralPopulation

Abstract parent for rate populations with continuous-valued activity.

Concrete populations store the current rate vector and per-step work buffers.
"""
abstract type RateNeuralPopulation <: NeuralPopulation end  # RNP

"""
    RateInput

Abstract parent for external inputs that add current or drive to rate
populations during `forward_signal!`.
"""
abstract type RateInput <: Input end  #  RIN

"""
    RateSynapses

Abstract parent for synaptic operators between rate populations.

Connectivity follows the package convention `post <- pre`: weights and
statistics are indexed as `[post, pre]`.
"""
abstract type RateSynapses <: Synapses end  #  RSY

"""
    RatePlasticity

Abstract parent for plasticity rules acting on rate-model state.
"""
abstract type RatePlasticity <: Plasticity end  #  RP

"""
    RateWeights

Abstract parent for rate-model weight containers.
"""
abstract type RateWeights <: Weights end  #  RW

"""
    RateRecorder

Abstract parent for recorders that sample rate-model state.
"""
abstract type RateRecorder <: Recorder end  #  RR


"""
    ExcitatoryRateNeuron(τ; rate_saturation=200.0)

Rate-neuron parameters for an excitatory population.

The field `τ` is the rate time constant used by `local_update!`, and
`rate_saturation` is the upper bound applied after each rate update. When this
population is the presynaptic side of a `RateLinearSynapses`, its synaptic drive
is added to the postsynaptic input buffer.
"""
struct ExcitatoryRateNeuron <: RateNeuronType
  τ::Float64
  rate_saturation::Float64
end

function ExcitatoryRateNeuron(τ::Float64; rate_saturation::Float64=200.0)
  return ExcitatoryRateNeuron(τ,rate_saturation)
end

"""
    InhibitoryRateNeuron(τ; rate_saturation=200.0)

Rate-neuron parameters for an inhibitory population.

The field `τ` is the rate time constant used by `local_update!`, and
`rate_saturation` is the upper bound applied after each rate update. When this
population is the presynaptic side of a `RateLinearSynapses`, its synaptic drive
is subtracted from the postsynaptic input buffer.
"""
struct InhibitoryRateNeuron <: RateNeuronType
  τ::Float64
  rate_saturation::Float64
end

function InhibitoryRateNeuron(τ::Float64; rate_saturation::Float64=200.0)
  return InhibitoryRateNeuron(τ,rate_saturation)
end

"""
    LinearRateNeuralPopulation

Population of rectified linear rate neurons.

Fields:
- `neuron_type`: rate-neuron parameter object, including the time constant `τ`.
- `n`: number of neurons.
- `rates_now`: current rates, length `n`.
- `input_alloc`: per-step input buffer accumulated by inputs and synapses.
- `utility_alloc`: scratch buffer reserved for algorithms that need temporary storage.

The rate update is Euler integration of `τ dr/dt = -r + input`, followed by
rectification at zero.
"""
struct LinearRateNeuralPopulation <: RateNeuralPopulation
  neuron_type::RateNeuronType
  n::Int64
  rates_now::Vector{Float64}
  input_alloc::Vector{Float64}
  utility_alloc::Vector{Float64}
end

"""
    LinearRateNeuralPopulation(neuron_type, n; initial_rates=nothing)

Create a `LinearRateNeuralPopulation` with zeroed input and utility buffers.

`initial_rates` may be:
- `nothing`: initialize all rates to `0.0`.
- a `Float64`: fill all rates with that value.
- a `Vector{Float64}` of length `n`: use that vector as `rates_now`.

The vector case keeps the supplied vector rather than copying it.
"""
function LinearRateNeuralPopulation(neuron_type::RateNeuronType,n::Int64;
    initial_rates::Union{Nothing,Float64,Vector{Float64}}=nothing)
  if isnothing(initial_rates)
    rates_now = fill(0.0,n)
  elseif isa(initial_rates,Float64)
    rates_now = fill(initial_rates,n)
  elseif isa(initial_rates,Vector{Float64})
    @assert length(initial_rates) == n "Length of initial_rates must be equal to n"
    rates_now = initial_rates
  else 
    error("Invalid type for initial_rates")
  end

  return LinearRateNeuralPopulation(
        neuron_type,
        n,
        rates_now,
        zeros(Float64,n),
        zeros(Float64,n)
    )
end

"""
    clean_up!(rnp::LinearRateNeuralPopulation) -> nothing

Reset the population input buffer to zero before the next simulation step.
"""
function clean_up!(rnp::LinearRateNeuralPopulation)
  rnp.input_alloc .= 0.0
  return nothing
end


"""
    RateNoisyHomogeneousInput

Homogeneous Gaussian input source for rate populations.

Each call to `forward_signal!` samples one standard normal variate per neuron
into `rand_alloc`, then adds a drive with mean `μ` and noise scale chosen so the
linear Euler update has stationary standard deviation approximately `σ`.
"""
struct RateNoisyHomogeneousInput <: RateInput
  n::Int64
  μ::Float64
  σ::Float64
  rand_alloc::Vector{Float64}
end

"""
    RateNoisyHomogeneousInput(n, μ, σ)

Create a noisy homogeneous input for `n` target neurons.

The random buffer is allocated once and reused on every `forward_signal!` call.
"""
function RateNoisyHomogeneousInput(n::Int64,μ::Float64,σ::Float64)
    return RateNoisyHomogeneousInput(
        n,
        μ,
        σ,
        fill(NaN,n)
    )
end 

# forward signal, input into the rate population


"""
    forward_signal!(t_now, dt, rnp, inp::RateNoisyHomogeneousInput) -> nothing

Add one noisy homogeneous input sample to `rnp.input_alloc`.

The time argument is currently unused. The noise scale depends on the target
population time constant and on the neural integration step `dt`.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateNoisyHomogeneousInput)
  # sample Gaussian noise for each neuron N(0,1)
  randn!(inp.rand_alloc)
  
  # Scale the input noise so the stationary rate variance after the Euler
  # update is inp.σ^2.
  _noise_scale_factor = sqrt(2 * rnp.neuron_type.τ / dt - 1) * inp.σ
  @inbounds @simd for i in 1:inp.n
    rnp.input_alloc[i] += inp.μ + _noise_scale_factor * inp.rand_alloc[i]
  end
  return nothing
end
 

"""
    local_update!(t_now, dt, rnp::LinearRateNeuralPopulation) -> nothing

Advance the population rates by one Euler step.

For each neuron, this applies
`r_new = r_old + (dt / τ) * (-r_old + input)` and then clamps values to
`[0, neuron_type.rate_saturation]`. `input_alloc` is read but not cleared; call
`clean_up!` at the beginning of each simulation step to clear accumulated input.
"""
function local_update!(t_now::Float64,dt::Float64,rnp::LinearRateNeuralPopulation)
  # linear rate neuron dynamics
  # dr/dt = -r + input(t)
  # r_new = r_old + dt * (-r_old + input(t))
  τ = rnp.neuron_type.τ
  rate_saturation = rnp.neuron_type.rate_saturation
  @inbounds @simd for i in 1:rnp.n
    _new_rate = rnp.rates_now[i] + (dt/τ) * (-rnp.rates_now[i] + rnp.input_alloc[i])
    if _new_rate > rate_saturation
      rnp.rates_now[i] = rate_saturation
    elseif _new_rate >= 0.0
      rnp.rates_now[i] = _new_rate
    else 
      rnp.rates_now[i] = 0.0
    end
  end
  return nothing
end


"""
    RateFixedInput

Deterministic external input for rate populations.

The `input` vector is added elementwise to a population's `input_alloc` on each
`forward_signal!` call.
"""
struct RateFixedInput <: RateInput
  n::Int64
  input::Vector{Float64}
end

"""
    RateFixedInput(input_vals)

Create a fixed input from a vector of per-neuron values.

The supplied vector is stored directly.
"""
function RateFixedInput(input_vals::Vector{Float64})
  n = length(input_vals)
  return RateFixedInput(
    n,
    input_vals
  )
end 

"""
    RateFixedInput(n, input)

Create a fixed input of length `n` whose entries are all `input`.
"""
function RateFixedInput(n::Int64,input::Float64)
  input_vec = fill(input,n)
  return RateFixedInput(
    n,
    input_vec
  )
end 


"""
    forward_signal!(t_now, dt, rnp, inp::RateFixedInput) -> nothing

Add the fixed input vector to `rnp.input_alloc`.

The time and step-size arguments are accepted for the common `forward_signal!`
interface and are not otherwise used.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp::RateNeuralPopulation,inp::RateFixedInput)
  @inbounds @simd for i in 1:inp.n
    rnp.input_alloc[i] += inp.input[i]
  end
  return nothing
end
  
#############################
# ====== Synapses ========= #
#############################

"""
    RateLinearSynapses

Dense linear synapse between two rate populations.

`weights` has shape `(n_post, n_pre)` and follows `post <- pre`: `weights[i,j]`
is the magnitude of the connection from presynaptic neuron `j` to postsynaptic
neuron `i`. The sign of the contribution is determined by the presynaptic
population type: excitatory sources add drive, inhibitory sources subtract it.
"""
struct RateLinearSynapses <: RateSynapses
  n_pre::Int64
  n_post::Int64
  weights::Matrix{Float64}

  function RateLinearSynapses(weights::Matrix{Float64})
    n_post,n_pre = size(weights)
    return new(n_pre,n_post,weights)
  end
end

"""
    forward_signal!(t_now, dt, rnp_post, sy::RateLinearSynapses, rnp_pre) -> nothing

Accumulate synaptic drive from `rnp_pre` into `rnp_post.input_alloc`.

The synapse computes `weights * r_pre` with the package's `post <- pre`
orientation. Excitatory presynaptic populations add this drive; inhibitory
presynaptic populations subtract it.
"""
function forward_signal!(t_now::Float64,dt::Float64,rnp_post::RateNeuralPopulation,sy::RateLinearSynapses,rnp_pre::RateNeuralPopulation)
  # compute synaptic current input, depending on neuron type
  if isa(rnp_pre.neuron_type,ExcitatoryRateNeuron)
    @inbounds @simd for i in 1:sy.n_post
      rnp_post.input_alloc[i] += dot(sy.weights[i,:],rnp_pre.rates_now)
    end
  elseif isa(rnp_pre.neuron_type,InhibitoryRateNeuron)
    @inbounds @simd for i in 1:sy.n_post
      rnp_post.input_alloc[i] -= dot(sy.weights[i,:],rnp_pre.rates_now)
    end
  end
  return nothing
end


############################
# ===== Recorders ======== #
############################

"""
    RCRate

Recorder for sampled rate trajectories from one rate population.

`times` stores recording times and `rates` stores one row per recorded sample.
Rows that have not been written are initialized to `NaN`; `krec` is the index of
the last written row.
"""
mutable struct RCRate <: RateRecorder
  population::RateNeuralPopulation
  dt::Float64
  t_end::Float64
  t_start::Float64
  times::Vector{Float64}
  rates::Matrix{Float64}
  krec::Int64
end


"""
    RCRate(population, t_end, dt; t_start=0.0)

Allocate a rate recorder for `population`.

The recorder samples at multiples of `dt` over `[t_start, t_end]`, inclusive up
to floating-point tolerance. Storage is allocated eagerly from the requested
time window.
"""
function RCRate(population::RateNeuralPopulation,t_end::Float64,dt::Float64;t_start=0.0)
  _nrec = ceil(Int,(t_end - t_start) / dt) + 1
  return RCRate(
    population,
    dt,
    t_end,
    t_start,
    fill(NaN,_nrec),
    fill(NaN,_nrec,population.n),
    0
  )
end


"""
    reset!(rec::RCRate) -> nothing

Clear all recorded samples and reset the write index.
"""
function reset!(rec::RCRate)
  rec.krec = 0
  fill!(rec.times,NaN)
  fill!(rec.rates,NaN)
  return nothing
end

"""
    record!(t_now, rec::RCRate) -> nothing

Record the population's current rates if `t_now` reaches the next recorder
sample time.

Calls outside `[t_start, t_end]` or repeated calls within the same recorder bin
are ignored. If storage is exhausted, a warning is emitted and no sample is
written.
"""
function record!(t_now::Float64,rec::RCRate)
  # if out of bonds, do nothing
  if (t_now<rec.t_start) || (t_now>rec.t_end)
    return nothing
  end
  _elapsed = t_now - rec.t_start
  _time_tol = 8 * eps(max(abs(_elapsed),abs(rec.dt),1.0))
  _krec_now = floor(Int,(_elapsed + _time_tol) / rec.dt) + 1
  # no discrete step in recorder dt, nothing to do
  if _krec_now <= rec.krec
    return nothing
  end
  # simulation longer than expected?
  if _krec_now > length(rec.times)
    @warn "Recorder full! Increase nrec to avoid data loss."
    return nothing
  end
  # store the rate, update the step
  rec.krec = _krec_now
  rec.times[rec.krec] = t_now
  copy!(view(rec.rates,rec.krec,:),rec.population.rates_now)
  return nothing
end

"""
    RCRateContent

Trimmed, read-oriented view of an `RCRate` recording.

`times` and `rates` include only written samples. `Tstart`, `Tend`, and `dt`
copy the recorder configuration.
"""
struct RCRateContent
  times::Vector{Float64}
  rates::Matrix{Float64}
  Tstart::Float64
  Tend::Float64
  dt::Float64
end

"""
    RCRateContent(rec::RCRate)

Return recorded rate data with unwritten trailing `NaN` rows removed.
"""
function RCRateContent(r::RCRate)
  _times = r.times
  _rates = r.rates
  idx_last = findfirst(isnan,_times)
  if isnothing(idx_last)
    return RCRateContent(_times,_rates,r.t_start,r.t_end,r.dt)
  else
    return RCRateContent(_times[1:idx_last-1],_rates[1:idx_last-1,:],r.t_start,r.t_end,r.dt)
  end
end

"""
    get_content(rec::RCRate)

Return an `RCRateContent` object for `rec`.
"""
function get_content(rec::RCRate)
  return RCRateContent(rec)
end



"""
    RCWeights

Recorder for synaptic weights between two populations in a rate model, just stores weight matrices.

`times` stores recording times 
`weights` is an array with dimensions `(n_rec,n_post, n_pre)` where `n_rec` is the number of recorded samples.

"""
mutable struct RCWeights <: RateRecorder
  weight_matrix_now::Matrix{Float64}
  dt::Float64
  t_end::Float64
  t_start::Float64
  times::Vector{Float64}
  weights::Array{Float64,3}
  krec::Int64
end

"""
    RCWeights(weight_matrix_now, t_end, dt; t_start=0.0)
    RCWeights(synapses, t_end, dt; t_start=0.0)

Allocate a weight recorder.

The matrix constructor stores `weight_matrix_now` directly, so later
modifications to the same matrix are recorded. The synapse constructor records
`synapses.weights`. Stored weights have dimensions `(n_rec, n_post, n_pre)`.
"""
function RCWeights(weight_matrix_now::Matrix{Float64},t_end::Float64,dt::Float64;t_start=0.0)
  _nrec = ceil(Int,(t_end - t_start) / dt) + 1
  return RCWeights(
    weight_matrix_now,
    dt,
    t_end,
    t_start,
    fill(NaN,_nrec),
    fill(NaN,_nrec,size(weight_matrix_now,1),size(weight_matrix_now,2)),
    0
  )
end

function RCWeights(synapses::RateSynapses,t_end::Float64,dt::Float64;t_start=0.0)
  return RCWeights(synapses.weights,t_end,dt;t_start=t_start)
end

function RCWeights(weight_matrix_now::Matrix{Float64},dt::Float64,t_end::Float64,t_start::Float64)
  return RCWeights(weight_matrix_now,t_end,dt;t_start=t_start)
end

"""
    reset!(rec::RCWeights) -> nothing

Clear all recorded weight samples and reset the write index.
"""
function reset!(rec::RCWeights)
  rec.krec = 0
  fill!(rec.times,NaN)
  fill!(rec.weights,NaN)
  return nothing
end

"""
    record!(t_now, rec::RCWeights) -> nothing

Record the current weight matrix if `t_now` reaches the next recorder sample
time.

Calls outside `[t_start, t_end]` or repeated calls within the same recorder bin
are ignored. If storage is exhausted, a warning is emitted and no sample is
written.
"""
function record!(t_now::Float64,rec::RCWeights)
  if (t_now<rec.t_start) || (t_now>rec.t_end)
    return nothing
  end
  _elapsed = t_now - rec.t_start
  _time_tol = 8 * eps(max(abs(_elapsed),abs(rec.dt),1.0))
  _krec_now = floor(Int,(_elapsed + _time_tol) / rec.dt) + 1
  if _krec_now <= rec.krec
    return nothing
  end
  if _krec_now > length(rec.times)
    @warn "Recorder full! Increase nrec to avoid data loss."
    return nothing
  end
  rec.krec = _krec_now
  rec.times[rec.krec] = t_now
  copy!(view(rec.weights,rec.krec,:,:),rec.weight_matrix_now)
  return nothing
end

"""
    RCWeightsContent

Trimmed, read-oriented view of an `RCWeights` recording.
"""
struct RCWeightsContent
  times::Vector{Float64}
  weights::Array{Float64,3}
  Tstart::Float64
  Tend::Float64
  dt::Float64
end

"""
    RCWeightsContent(rec::RCWeights)

Return recorded weight data with unwritten trailing rows removed.
"""
function RCWeightsContent(rec::RCWeights)
  return RCWeightsContent(
    rec.times[1:rec.krec],
    rec.weights[1:rec.krec,:,:],
    rec.t_start,
    rec.t_end,
    rec.dt
  )
end

"""
    get_content(rec::RCWeights)

Return an `RCWeightsContent` object for `rec`.
"""
function get_content(rec::RCWeights)
  return RCWeightsContent(rec)
end


###############################################
# ===== Mean and variance estimators ======== #
###############################################

# included in plasticity rules, updated by plasticity updates


"""
    RateMeanEstimator

Online exponentially weighted mean estimator for one rate population.

The estimator updates at interval `dt`, not necessarily every neural integration
step. It stores `propagation_factor = exp(-dt / τ)` and updates
`mean_now = a * mean_now + (1 - a) * rates_now` whenever the estimator interval
has elapsed.
"""
mutable struct RateMeanEstimator
  pop::RateNeuralPopulation
  n::Int64
  τ::Float64
  dt::Float64
  propagation_factor::Float64
  mean_now::Vector{Float64}
  t_last_update::Float64
end

"""
    RateMeanEstimator(pop, τ, dt)

Create a mean estimator for `pop` with memory time constant `τ` and update
interval `dt`.

The initial mean is zero and `t_last_update` is `-Inf`, so the first
`local_update!` call updates immediately.
"""
function RateMeanEstimator(pop::RateNeuralPopulation,τ::Float64,dt::Float64;
    initial_mean::Union{Vector{Float64},Float64}=0.0)
  if isa(initial_mean,Vector{Float64})
    @assert length(initial_mean) == pop.n "Length of initial_mean must be equal to pop.n"
    mean_now = initial_mean
  elseif isa(initial_mean,Float64)
    mean_now = fill(initial_mean,pop.n)
  else
    error("Invalid type for initial_mean")
  end 
  return RateMeanEstimator(
    pop,
    pop.n,
    τ,
    dt,
    exp(-dt / τ),
    mean_now,
    -Inf
  )
end


"""
    reset!(est::RateMeanEstimator) -> nothing

Reset the estimated mean to zero and allow the next `local_update!` call to
update immediately.
"""
function reset!(est::RateMeanEstimator)
  fill!(est.mean_now,0.0)
  est.t_last_update = -Inf
  return nothing
end

"""
    local_update!(t_now, dt, est::RateMeanEstimator) -> nothing

Update the mean trace if at least `est.dt` time has elapsed since the previous
estimator update.

The `dt` argument is accepted for the shared local-update interface; the
estimator's own interval `est.dt` determines the exponential propagation factor.
"""
function local_update!(t_now::Float64,dt::Float64,est::RateMeanEstimator)
  # update only if enough time has passed
  if t_now - est.t_last_update < est.dt
    return nothing
  end
  est.t_last_update = t_now
  a = est.propagation_factor
  @inbounds @simd for i in 1:est.n
    est.mean_now[i] = a * est.mean_now[i] + (1 - a) * est.pop.rates_now[i]
  end
  return nothing
end


abstract type RateCovarianceAccumulator end

"""
    RateCovarianceEstimator

Online second-moment and covariance estimator for a pair of rate populations.
Matrices use the package convention `post <- pre`, so entry `[i,j]` corresponds
to post neuron `i` and pre neuron `j`.

The estimator stores the exponentially weighted second moment `Q` in
`second_moment_now` and the covariance `C` in `covariance_now`. Means are read
from externally supplied `RateMeanEstimator`s to keep first- and second-order
statistics modular.
"""
mutable struct RateCovarianceEstimator <: RateCovarianceAccumulator
  pop_post::RateNeuralPopulation
  pop_pre::RateNeuralPopulation
  n_post::Int64
  n_pre::Int64
  τ::Float64
  dt::Float64
  propagation_factor::Float64
  mean_post_estimator::RateMeanEstimator
  mean_pre_estimator::RateMeanEstimator
  second_moment_now::Matrix{Float64}
  covariance_now::Matrix{Float64}
  t_last_update::Float64
end

"""
    RateCovarianceEstimator(pop_post, pop_pre, mean_post_estimator, mean_pre_estimator, τ, dt)
    RateCovarianceEstimator(mean_post_estimator, mean_pre_estimator; τ=nothing, dt=nothing)

Construct a covariance estimator that tracks
`Q = E[r_post * r_pre']` and `C = Q - μ_post * μ_pre'`.

The shorter constructor extracts the populations from the supplied mean
estimators. If `τ` or `dt` is omitted, that value is inferred from the mean
estimators and the post/pre mean estimators must agree.
"""
function RateCovarianceEstimator(
    pop_post::RateNeuralPopulation,
    pop_pre::RateNeuralPopulation,
    mean_post_estimator::RateMeanEstimator,
    mean_pre_estimator::RateMeanEstimator,
    τ::Float64,
    dt::Float64)
  @assert mean_post_estimator.pop === pop_post "Post mean estimator must track pop_post"
  @assert mean_pre_estimator.pop === pop_pre "Pre mean estimator must track pop_pre"
  return RateCovarianceEstimator(
    pop_post,
    pop_pre,
    pop_post.n,
    pop_pre.n,
    τ,
    dt,
    exp(-dt / τ),
    mean_post_estimator,
    mean_pre_estimator,
    fill(0.0,pop_post.n,pop_pre.n),
    fill(0.0,pop_post.n,pop_pre.n),
    -Inf
  )
end

function RateCovarianceEstimator(
    mean_post_estimator::RateMeanEstimator,
    mean_pre_estimator::RateMeanEstimator;
    τ::Union{Nothing,Float64}=nothing,
    dt::Union{Nothing,Float64}=nothing)
  if isnothing(τ)
    @assert mean_post_estimator.τ == mean_pre_estimator.τ "Cannot infer covariance τ: post and pre mean estimators use different τ values"
    τ = mean_post_estimator.τ
  end
  if isnothing(dt)
    @assert mean_post_estimator.dt == mean_pre_estimator.dt "Cannot infer covariance dt: post and pre mean estimators use different dt values"
    dt = mean_post_estimator.dt
  end

  return RateCovarianceEstimator(
    mean_post_estimator.pop,
    mean_pre_estimator.pop,
    mean_post_estimator,
    mean_pre_estimator,
    τ,
    dt,
  )
end


"""
    reset!(est::RateCovarianceEstimator) -> nothing

Reset the second-moment and covariance matrices to zero.

The referenced mean estimators are not reset.
"""
function reset!(est::RateCovarianceEstimator)
  fill!(est.second_moment_now,0.0)
  fill!(est.covariance_now,0.0)
  est.t_last_update = -Inf
  return nothing
end


"""
    local_update!(t_now, dt, est::RateCovarianceEstimator) -> nothing

Update `second_moment_now` and `covariance_now` if at least `est.dt` time has
elapsed since the previous covariance update.

The second moment uses
`Q = b * Q + (1 - b) * (r_post * r_pre')`, with
`b = est.propagation_factor`. The covariance is then computed from the current
mean-estimator states as `C = Q - μ_post * μ_pre'`.
"""
function local_update!(t_now::Float64,dt::Float64,est::RateCovarianceEstimator)
  # update only if enough time has passed
  if t_now - est.t_last_update < est.dt
    return nothing
  end
  est.t_last_update = t_now
  a = est.propagation_factor
  input_factor = 1 - a

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

"""
    CovarianceTransposed(covariance_estimator)

View a covariance estimator with post/pre orientation transposed.

This is useful when two plasticity rules should use the same running covariance
estimate in opposite synaptic directions. Access to covariance entries should go
through `covariance_at`, which returns `C[i,j]` for a plain estimator and
`C[j,i]` for a `CovarianceTransposed` wrapper.
"""
struct CovarianceTransposed <: RateCovarianceAccumulator
  covariance_estimator::RateCovarianceEstimator
end

covariance_pop_post(est::RateCovarianceEstimator) = est.pop_post
covariance_pop_pre(est::RateCovarianceEstimator) = est.pop_pre
covariance_n_post(est::RateCovarianceEstimator) = est.n_post
covariance_n_pre(est::RateCovarianceEstimator) = est.n_pre
covariance_at(est::RateCovarianceEstimator,i::Int64,j::Int64) =
  est.covariance_now[i,j]

covariance_pop_post(est::CovarianceTransposed) =
  est.covariance_estimator.pop_pre
covariance_pop_pre(est::CovarianceTransposed) =
  est.covariance_estimator.pop_post
covariance_n_post(est::CovarianceTransposed) =
  est.covariance_estimator.n_pre
covariance_n_pre(est::CovarianceTransposed) =
  est.covariance_estimator.n_post
covariance_at(est::CovarianceTransposed,i::Int64,j::Int64) =
  est.covariance_estimator.covariance_now[j,i]

###################################
# ===== Plasticity rules ======== #
###################################

"""
    RatePlasticityHomeostaticScaling
    RatePlasticityHomeostaticScaling(pop_post, synapses_post_pre, pop_pre, α, Δt, learning_rate, rate_estimator_post)

Homeostatic scaling plasticity rule for a pair of rate populations.
the rule is w <- w + Δt * w * learning rate * r_post(t) *( α - r_post_mean_estimator(t) )
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.

Note that this rule depends only on the postsynaptic rate, therefore it scales all incoming weights of the same amount.

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
mutable struct RatePlasticityHomeostaticScaling <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α::Float64
  Δt::Float64
  learning_rate::Float64
  rate_estimator_post::RateMeanEstimator
  t_last_update::Float64
  w_min::Float64
  w_max::Float64
end

function RatePlasticityHomeostaticScaling(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    α::Float64,
    Δt::Float64,
    learning_rate::Float64,
    rate_estimator_post::RateMeanEstimator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  @assert rate_estimator_post.pop === pop_post "Rate estimator must track the postsynaptic population"
  return RatePlasticityHomeostaticScaling(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α,
    Δt,
    learning_rate,
    rate_estimator_post,
    -Inf,  # t_last_update
    w_min,
    w_max
  )
end


function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityHomeostaticScaling)
  # update only if enough time has passed
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      if w_old == 0.0
        continue  # only positive weights are plastic
      end
      r_post = rule.pop_post.rates_now[i]
      r_post_mean = rule.rate_estimator_post.mean_now[i]
      w_new = w_old + effective_learning_rate * w_old * r_post * (rule.α - r_post_mean)
      # clip to bounds
      w_new_clipped = clamp(w_new,rule.w_min,rule.w_max)
      rule.synapses_post_pre.weights[i,j] = w_new_clipped
    end
  end

  return nothing
end

"""
    RatePlasticityCovariance
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, Δt, learning_rate, covariance_estimator)
    RatePlasticityCovariance(pop_post, synapses_post_pre, pop_pre, α, Δt, learning_rate, covariance_estimator)

Simple covariance plasticity rule for a pair of rate populations.
the rule is w <- w + Δt * learning_rate * (C_post_pre(t) - α)
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
mutable struct RatePlasticityCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  t_last_update::Float64
  w_min::Float64
  w_max::Float64
end

function RatePlasticityCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  return RatePlasticityCovariance(
    pop_post,
    synapses_post_pre,
    pop_pre,
    0.0,
    Δt,
    learning_rate,
    covariance_estimator;
    w_min=w_min,
    w_max=w_max
  )
end

function RatePlasticityCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    α::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  return RatePlasticityCovariance(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α,
    Δt,
    learning_rate,
    covariance_estimator,
    -Inf,
    w_min,
    w_max
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityCovariance)
  # update only if enough time has passed
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      if w_old == 0.0
        continue  # only positive weights are plastic
      end
      covariance = covariance_at(rule.covariance_estimator,i,j)
      w_new = w_old + effective_learning_rate * (covariance - rule.α)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end

  return nothing
end

"""
    RatePlasticityScaledCovariance
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, Δt, learning_rate, covariance_estimator)
    RatePlasticityScaledCovariance(pop_post, synapses_post_pre, pop_pre, scale_matrix, α, Δt, learning_rate, covariance_estimator)

Covariance-based plasticity rule, with an additional arbitrary pre-post scaling factor, assumed  to depend on distance.
the rule is w <- w + Δt * learning_rate * A_post_pre *(C_post_pre(t) - α)
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.
The A_post_pre scale matrix can be decided arbitrarily, for example acting as a simple mask, but for my purposes it will be 
a distance-based scaling factor. See e.g. utility function `generate_ring_topology`. 

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
mutable struct RatePlasticityScaledCovariance <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  scale_matrix::Matrix{Float64}
  t_last_update::Float64
  w_min::Float64
  w_max::Float64
end


function RatePlasticityScaledCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  return RatePlasticityScaledCovariance(
    pop_post,
    synapses_post_pre,
    pop_pre,
    scale_matrix,
    0.0,
    Δt,
    learning_rate,
    covariance_estimator;
    w_min=w_min,
    w_max=w_max
  )
end

function RatePlasticityScaledCovariance(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    scale_matrix::Matrix{Float64},
    α::Float64,
    Δt::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  @assert size(scale_matrix) == (pop_post.n,pop_pre.n) "Scale matrix dimensions must match (pop_post.n, pop_pre.n)"
  return RatePlasticityScaledCovariance(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α,
    Δt,
    learning_rate,
    covariance_estimator,
    scale_matrix,
    -Inf,
    w_min,
    w_max
  )
end

function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityScaledCovariance)
  # update only if enough time has passed
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      scale = rule.scale_matrix[i,j]
      # if zero weight or zero scale, skip update
      if (w_old == 0.0) || (scale == 0.0)
        continue  # only positive weights are plastic
      end
      covariance = covariance_at(rule.covariance_estimator,i,j)
      w_new = w_old + effective_learning_rate * scale * (covariance - rule.α)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end

  return nothing
end


"""
    RatePlasticityCovarianceQuadraticallyStabilized
    RatePlasticityCovarianceQuadraticallyStabilized(pop_post, synapses_post_pre, pop_pre, Δt, α1, α2, learning_rate, covariance_estimator)

Covariance-based plasticity rule with an additional stabilizing term that
scales quadratically with the weight.
the rule is w <- w + Δt * learning_rate * (α1 * C_post_pre(t) + α2 * w^2)
With an update every Δt seconds. `learning_rate` is interpreted per unit time,
so each update is internally scaled by the elapsed plasticity interval.

α1 and α2 are intended to be of opposite sign, and depend on the effect on the neuron on covariance.

For example, an inhibitory neuron will have α1 < 0 and α2 > 0, so that the covariance term will tend to decrease the weight,
while the quadratic term will tend to increase it, stabilizing the weight at a finite value.

IMPORTANT: by convention the rule acts only on weights > 0 . So you must initialize all weights that you want plastic to a small
positive value, also making sure that w_min > 0 (and very small).
"""
mutable struct RatePlasticityCovarianceQuadraticallyStabilized <: RatePlasticity
  pop_pre::RateNeuralPopulation
  pop_post::RateNeuralPopulation
  synapses_post_pre::RateSynapses
  α1::Float64
  α2::Float64
  Δt::Float64
  learning_rate::Float64
  covariance_estimator::RateCovarianceAccumulator
  t_last_update::Float64
  w_min::Float64
  w_max::Float64
end


function RatePlasticityCovarianceQuadraticallyStabilized(
    pop_post::RateNeuralPopulation,
    synapses_post_pre::RateSynapses,
    pop_pre::RateNeuralPopulation,
    Δt::Float64,
    α1::Float64,
    α2::Float64,
    learning_rate::Float64,
    covariance_estimator::RateCovarianceAccumulator;
    w_min::Float64=1E-8,
    w_max::Float64=Inf
)
  @assert covariance_pop_post(covariance_estimator) === pop_post "Covariance estimator must track the postsynaptic population"
  @assert covariance_pop_pre(covariance_estimator) === pop_pre "Covariance estimator must track the presynaptic population"
  @assert synapses_post_pre.n_post == pop_post.n "Synapse postsynaptic dimension must match pop_post.n"
  @assert synapses_post_pre.n_pre == pop_pre.n "Synapse presynaptic dimension must match pop_pre.n"
  @assert covariance_n_post(covariance_estimator) == pop_post.n "Covariance postsynaptic dimension must match pop_post.n"
  @assert covariance_n_pre(covariance_estimator) == pop_pre.n "Covariance presynaptic dimension must match pop_pre.n"
  return RatePlasticityCovarianceQuadraticallyStabilized(
    pop_pre,
    pop_post,
    synapses_post_pre,
    α1,
    α2,
    Δt,
    learning_rate,
    covariance_estimator,
    -Inf,
    w_min,
    w_max
  )
end


function plasticity!(t_now::Float64,dt::Float64,rule::RatePlasticityCovarianceQuadraticallyStabilized)
  # update only if enough time has passed
  if t_now - rule.t_last_update < rule.Δt
    return nothing
  end
  effective_learning_rate = rule.learning_rate * rule.Δt
  rule.t_last_update = t_now

  @inbounds for j in 1:rule.synapses_post_pre.n_pre
    for i in 1:rule.synapses_post_pre.n_post
      w_old = rule.synapses_post_pre.weights[i,j]
      # if zero weight, skip update
      if w_old == 0.0
        continue  # only positive weights are plastic
      end
      covariance = covariance_at(rule.covariance_estimator,i,j)
      w_new = w_old + effective_learning_rate * (rule.α1 * covariance + rule.α2 * w_old^2)
      rule.synapses_post_pre.weights[i,j] = clamp(w_new,rule.w_min,rule.w_max)
    end
  end

  return nothing
end
