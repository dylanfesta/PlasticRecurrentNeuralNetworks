using PlasticRecurrentNeuralNetworks
using LinearAlgebra
using Random
using Statistics
using Test

@testset "PlasticRecurrentNeuralNetworks.jl" begin
  PNN = PlasticRecurrentNeuralNetworks

  @testset "RateNoisyHomogeneousInput statistics" begin
    Random.seed!(1234)

    dt = 1e-3
    t_end = 5.0
    t_warmup = 1.0
    tau = 0.1
    mu = 20.0
    sigma = 3.0
    n = 200

    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(tau;rate_saturation=100.0),
      n,
      initial_rates=0.0,
    )
    input = PNN.RateNoisyHomogeneousInput(n,mu,sigma)
    rec = PNN.RCRate(population,t_end,dt)
    network = PNN.RecurrentNetwork(
      populations=(population,),
      connections=((population,input),),
      recorders=(rec,),
    )

    t_now = 0.0
    while t_now <= t_end
      t_now = PNN.dynamic_step!(t_now,dt,network)
    end

    content = PNN.get_content(rec)
    rates = vec(content.rates[content.times .> t_warmup,:])

    @test isapprox(mean(rates),mu; atol=0.1)
    @test isapprox(std(rates),sigma; rtol=0.05)
  end

  @testset "Excitatory self connection fixed points" begin
    dt = 1e-3
    t_end = 8.0
    tau = 0.1
    mu = 20.0
    n = 1
    initial_rate = 5.0
    self_connection_weights = (0.0,0.345,0.8)

    for w_self in self_connection_weights
      expected_fixed_point = mu / (1.0 - w_self)
      @test expected_fixed_point > 0.0

      population = PNN.LinearRateNeuralPopulation(
        PNN.ExcitatoryRateNeuron(tau;rate_saturation=100.0),
        n,
        initial_rates=initial_rate,
      )
      input = PNN.RateFixedInput(n,mu)
      synapse = PNN.RateLinearSynapses(fill(w_self,(n,n)))
      rec = PNN.RCRate(population,t_end,0.05)
      network = PNN.RecurrentNetwork(
        populations=(population,),
        connections=((population,input),(population,synapse,population)),
        recorders=(rec,),
      )

      t_now = 0.0
      while t_now <= t_end
        t_now = PNN.dynamic_step!(t_now,dt,network)
      end

      content = PNN.get_content(rec)
      @test content.times[end] >= t_end - rec.dt - dt
      @test isapprox(content.rates[end,1],expected_fixed_point; rtol=1e-3)
      @test isapprox(population.rates_now[1],expected_fixed_point; rtol=1e-3)
    end
  end

  @testset "Two-dimensional E/I fixed point" begin
    dt = 1e-3
    t_end = 30.0
    tau_exc = 90e-3
    tau_inh = 66e-3
    dt_recorder = 50e-3
    mu = 20.0
    n = 1
    initial_rates = [5.0,5.0]
    w_matrix_form = [[0.3,2.0] [-1.0,-2.0]]
    expected_fixed_point = (I - w_matrix_form) \ fill(mu,2)

    @test isapprox(expected_fixed_point,[9.756097560975611,13.170731707317074])

    e_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(tau_exc;rate_saturation=100.0),
      n,
      initial_rates=initial_rates[1],
    )
    i_population = PNN.LinearRateNeuralPopulation(
      PNN.InhibitoryRateNeuron(tau_inh;rate_saturation=100.0),
      n,
      initial_rates=initial_rates[2],
    )

    e_input = PNN.RateFixedInput(n,mu)
    i_input = PNN.RateFixedInput(n,mu)
    rec_e = PNN.RCRate(e_population,t_end,dt_recorder)
    rec_i = PNN.RCRate(i_population,t_end,dt_recorder)

    scalar_weight_matrix(x::Number) = fill(abs(float(x)),n,n)

    network = PNN.RecurrentNetwork(
      populations=(e_population,i_population),
      connections=(
        (e_population,e_input),
        (i_population,i_input),
        (e_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[1,1])),e_population),
        (i_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[2,1])),e_population),
        (e_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[1,2])),i_population),
        (i_population,PNN.RateLinearSynapses(scalar_weight_matrix(w_matrix_form[2,2])),i_population),
      ),
      recorders=(rec_e,rec_i),
    )

    t_now = 0.0
    while t_now <= t_end
      t_now = PNN.dynamic_step!(t_now,dt,network)
    end

    content_e = PNN.get_content(rec_e)
    content_i = PNN.get_content(rec_i)
    final_rates = [content_e.rates[end,1],content_i.rates[end,1]]

    @test isapprox(final_rates,expected_fixed_point; rtol=1e-12)
    @test isapprox(
      [e_population.rates_now[1],i_population.rates_now[1]],
      expected_fixed_point;
      rtol=1e-12,
    )
  end

  @testset "LinearRateNeuralPopulation rate saturation" begin
    neuron_type = PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0)
    inhibitory_type = PNN.InhibitoryRateNeuron(1.0;rate_saturation=100.0)

    @test neuron_type.rate_saturation == 100.0
    @test inhibitory_type.rate_saturation == 100.0

    population = PNN.LinearRateNeuralPopulation(
      neuron_type,
      3,
      initial_rates=[90.0,20.0,5.0],
    )
    population.input_alloc .= [200.0,50.0,-10.0]

    @test PNN.local_update!(0.0,1.0,population) === nothing
    @test population.rates_now == [100.0,50.0,0.0]
  end

  @testset "RateLinearSynapses constructor" begin
    weights = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    synapse = PNN.RateLinearSynapses(weights)

    @test synapse.n_post == 2
    @test synapse.n_pre == 3
    @test synapse.weights === weights
    @test !isdefined(PNN,:RateLinearSynapse)
    @test_throws MethodError PNN.RateLinearSynapses(3,2,weights)
  end

  @testset "RCRate recorder" begin
    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      [1.0,2.0,3.0],
      zeros(3),
      zeros(3),
    )

    rec = PNN.RCRate(population,0.5,0.2)
    @test length(rec.times) == 4
    @test size(rec.rates) == (4,3)
    @test rec.krec == 0

    @test PNN.record!(0.0,rec) === nothing
    @test rec.krec == 1
    @test rec.times[1] == 0.0
    @test rec.rates[1,:] == [1.0,2.0,3.0]

    population.rates_now .= [4.0,5.0,6.0]
    @test PNN.record!(0.1,rec) === nothing
    @test rec.krec == 1
    @test isnan(rec.times[2])

    @test PNN.record!(0.2,rec) === nothing
    @test rec.krec == 2
    @test rec.times[2] == 0.2
    @test rec.rates[2,:] == [4.0,5.0,6.0]

    population.rates_now .= [7.0,8.0,9.0]
    @test PNN.record!(0.39,rec) === nothing
    @test rec.krec == 2

    @test PNN.record!(0.4,rec) === nothing
    @test rec.krec == 3
    @test rec.times[3] == 0.4
    @test rec.rates[3,:] == [7.0,8.0,9.0]

    @test PNN.record!(0.6,rec) === nothing
    @test rec.krec == 3

    content = PNN.get_content(rec)
    @test content.times == [0.0,0.2,0.4]
    @test content.rates == [
      1.0 2.0 3.0
      4.0 5.0 6.0
      7.0 8.0 9.0
    ]

    PNN.reset!(rec)
    @test rec.krec == 0
    @test all(isnan,rec.times)
    @test all(isnan,rec.rates)

    rec_with_offset = PNN.RCRate(population,0.5,0.2;t_start=0.1)
    @test PNN.record!(0.0,rec_with_offset) === nothing
    @test rec_with_offset.krec == 0
    @test PNN.record!(0.1,rec_with_offset) === nothing
    @test rec_with_offset.krec == 1
    @test PNN.record!(0.3,rec_with_offset) === nothing
    @test rec_with_offset.krec == 2
    @test rec_with_offset.times[2] == 0.3
  end

  @testset "RCWeights recorder" begin
    weights = [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    rec = PNN.RCWeights(weights,0.5,0.2)

    @test rec.weight_matrix_now === weights
    @test length(rec.times) == 4
    @test size(rec.weights) == (4,2,3)
    @test rec.krec == 0

    @test PNN.record!(0.0,rec) === nothing
    @test rec.krec == 1
    @test rec.times[1] == 0.0
    @test rec.weights[1,:,:] == weights

    weights .= [
      7.0 8.0 9.0
      10.0 11.0 12.0
    ]
    @test PNN.record!(0.1,rec) === nothing
    @test rec.krec == 1
    @test isnan(rec.times[2])

    @test PNN.record!(0.2,rec) === nothing
    @test rec.krec == 2
    @test rec.times[2] == 0.2
    @test rec.weights[2,:,:] == weights

    weights .= [
      13.0 14.0 15.0
      16.0 17.0 18.0
    ]
    @test PNN.record!(0.4,rec) === nothing
    @test rec.krec == 3
    @test rec.times[3] == 0.4
    @test rec.weights[3,:,:] == weights

    content = PNN.get_content(rec)
    @test content.times == [0.0,0.2,0.4]
    @test size(content.weights) == (3,2,3)
    @test content.weights[1,:,:] == [
      1.0 2.0 3.0
      4.0 5.0 6.0
    ]
    @test content.weights[3,:,:] == [
      13.0 14.0 15.0
      16.0 17.0 18.0
    ]

    synapse = PNN.RateLinearSynapses(weights)
    rec_from_synapse = PNN.RCWeights(synapse,0.4,0.2)
    @test rec_from_synapse.weight_matrix_now === synapse.weights

    legacy_rec = PNN.RCWeights(weights,0.2,0.4,0.1)
    @test legacy_rec.dt == 0.2
    @test legacy_rec.t_end == 0.4
    @test legacy_rec.t_start == 0.1

    @test PNN.reset!(rec) === nothing
    @test rec.krec == 0
    @test all(isnan,rec.times)
    @test all(isnan,rec.weights)
  end

  @testset "RateMeanEstimator" begin
    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[10.0,20.0],
    )
    tau_mean = 0.5
    dt_trace = 0.2
    dt_rate = 0.01
    estimator = PNN.RateMeanEstimator(population,tau_mean,dt_trace)
    decay = exp(-dt_trace / tau_mean)

    @test estimator.propagation_factor == decay
    @test estimator.mean_now == [0.0,0.0]
    @test estimator.t_last_update == -Inf

    @test PNN.local_update!(0.0,dt_rate,estimator) === nothing
    first_expected = (1 - decay) .* [10.0,20.0]
    @test isapprox(estimator.mean_now,first_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    population.rates_now .= [30.0,50.0]
    @test PNN.local_update!(0.1,dt_rate,estimator) === nothing
    @test isapprox(estimator.mean_now,first_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    @test PNN.local_update!(0.2,dt_rate,estimator) === nothing
    second_expected = decay .* first_expected .+ (1 - decay) .* [30.0,50.0]
    @test isapprox(estimator.mean_now,second_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.2

    @test PNN.reset!(estimator) === nothing
    @test estimator.mean_now == [0.0,0.0]
    @test estimator.t_last_update == -Inf
  end

  @testset "RateCovarianceEstimator" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[10.0,20.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[1.0,2.0,3.0],
    )
    tau_covariance = 0.5
    dt_trace = 0.2
    dt_rate = 0.01
    post_mean_estimator = PNN.RateMeanEstimator(post_population,tau_covariance,dt_trace)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,dt_trace)
    estimator = PNN.RateCovarianceEstimator(post_mean_estimator,pre_mean_estimator)
    decay = exp(-dt_trace / tau_covariance)
    input_factor = 1 - decay

    @test estimator.pop_post === post_population
    @test estimator.pop_pre === pre_population
    @test estimator.τ == tau_covariance
    @test estimator.dt == dt_trace
    @test estimator.propagation_factor == decay
    @test estimator.mean_post_estimator === post_mean_estimator
    @test estimator.mean_pre_estimator === pre_mean_estimator
    @test size(estimator.second_moment_now) == (2,3)
    @test size(estimator.covariance_now) == (2,3)
    @test estimator.t_last_update == -Inf

    post_rates = copy(post_population.rates_now)
    pre_rates = copy(pre_population.rates_now)
    @test PNN.local_update!(0.0,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.0,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.0,dt_rate,estimator) === nothing
    mean_post_expected = input_factor .* post_rates
    mean_pre_expected = input_factor .* pre_rates
    second_moment_expected = input_factor .* (post_rates * pre_rates')
    covariance_expected = second_moment_expected .- mean_post_expected * mean_pre_expected'

    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    post_population.rates_now .= [30.0,50.0]
    pre_population.rates_now .= [4.0,5.0,6.0]
    @test PNN.local_update!(0.1,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.1,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.1,dt_rate,estimator) === nothing
    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.0

    post_rates = copy(post_population.rates_now)
    pre_rates = copy(pre_population.rates_now)
    @test PNN.local_update!(0.2,dt_rate,post_mean_estimator) === nothing
    @test PNN.local_update!(0.2,dt_rate,pre_mean_estimator) === nothing
    @test PNN.local_update!(0.2,dt_rate,estimator) === nothing
    mean_post_expected = decay .* mean_post_expected .+ input_factor .* post_rates
    mean_pre_expected = decay .* mean_pre_expected .+ input_factor .* pre_rates
    second_moment_expected = decay .* second_moment_expected .+
      input_factor .* (post_rates * pre_rates')
    covariance_expected = second_moment_expected .- mean_post_expected * mean_pre_expected'

    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test isapprox(estimator.second_moment_now,second_moment_expected; rtol=1e-12)
    @test isapprox(estimator.covariance_now,covariance_expected; rtol=1e-12)
    @test estimator.t_last_update == 0.2

    @test PNN.reset!(estimator) === nothing
    @test isapprox(post_mean_estimator.mean_now,mean_post_expected; rtol=1e-12)
    @test isapprox(pre_mean_estimator.mean_now,mean_pre_expected; rtol=1e-12)
    @test all(iszero,estimator.second_moment_now)
    @test all(iszero,estimator.covariance_now)
    @test estimator.t_last_update == -Inf

    explicit_time_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator;
      τ=1.0,
      dt=0.3,
    )
    @test explicit_time_estimator.τ == 1.0
    @test explicit_time_estimator.dt == 0.3
    @test explicit_time_estimator.propagation_factor == exp(-0.3 / 1.0)

    different_tau_mean_estimator = PNN.RateMeanEstimator(pre_population,0.7,dt_trace)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_tau_mean_estimator,
    )
    @test PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_tau_mean_estimator;
      τ=tau_covariance,
    ).τ == tau_covariance

    different_dt_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,0.3)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_dt_mean_estimator,
    )
    @test PNN.RateCovarianceEstimator(
      post_mean_estimator,
      different_dt_mean_estimator;
      dt=dt_trace,
    ).dt == dt_trace

    mismatched_mean_estimator = PNN.RateMeanEstimator(pre_population,tau_covariance,dt_trace)
    @test_throws AssertionError PNN.RateCovarianceEstimator(
      post_population,
      pre_population,
      mismatched_mean_estimator,
      pre_mean_estimator,
      tau_covariance,
      dt_trace,
    )
  end

  @testset "RatePlasticityHomeostaticScaling" begin
    population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[2.0,4.0],
    )
    synapse = PNN.RateLinearSynapses([
      0.0  1.0
      2.0  0.0
    ])
    mean_estimator = PNN.RateMeanEstimator(
      population,
      1.0,
      0.1;
      initial_mean=[1.0,5.0],
    )
    rule = PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse,
      population,
      3.0,
      0.2,
      0.5,
      mean_estimator;
      w_min=0.1,
      w_max=10.0,
    )

    rule.t_last_update = 0.0
    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test synapse.weights == [
      0.0  1.0
      2.0  0.0
    ]
    @test rule.t_last_update == 0.0

    @test PNN.plasticity!(0.2,0.01,rule) === nothing
    @test isapprox(synapse.weights,[0.0 1.4; 0.4 0.0]; rtol=1e-12)
    @test rule.t_last_update == 0.2

    synapse_small_dt = PNN.RateLinearSynapses([
      0.0  1.0
      2.0  0.0
    ])
    rule_small_dt = PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse_small_dt,
      population,
      3.0,
      0.1,
      0.5,
      mean_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    rule_small_dt.t_last_update = 0.0
    @test PNN.plasticity!(0.1,0.01,rule_small_dt) === nothing
    @test isapprox(synapse_small_dt.weights,[0.0 1.2; 1.2 0.0]; rtol=1e-12)
    @test isapprox(
      synapse.weights[1,2] - 1.0,
      2 * (synapse_small_dt.weights[1,2] - 1.0);
      rtol=1e-12,
    )

    other_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=0.0,
    )
    other_estimator = PNN.RateMeanEstimator(other_population,1.0,0.1)
    @test_throws AssertionError PNN.RatePlasticityHomeostaticScaling(
      population,
      synapse,
      population,
      3.0,
      0.2,
      0.5,
      other_estimator,
    )
  end

  @testset "RatePlasticityCovariance" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(post_population,1.0,0.1)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,1.0,0.1)
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule = PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.t_last_update == -Inf

    plain_rule = PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.1,
      0.2,
      covariance_estimator,
    )
    @test plain_rule.α == 0.0

    rule.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      1.03  0.0   0.1
      2.0   3.01  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityCovariance(
      post_population,
      synapse_small_dt,
      pre_population,
      0.5,
      0.05,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    rule_small_dt.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityCovariance(
      post_population,
      synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityCovariance(
      post_population,
      wrong_size_synapse,
      pre_population,
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )
  end

  @testset "RatePlasticityScaledCovariance" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(post_population,1.0,0.1)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,1.0,0.1)
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    scale_matrix = [
      1.0  0.0  2.0
      0.5  3.0  0.25
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      0.5,
      0.1,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.scale_matrix === scale_matrix
    @test rule.t_last_update == -Inf

    plain_rule = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      0.1,
      0.2,
      covariance_estimator,
    )
    @test plain_rule.α == 0.0

    rule.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      1.03   0.0   0.1
      2.0    3.01  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse_small_dt,
      pre_population,
      scale_matrix,
      0.5,
      0.05,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    rule_small_dt.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      scale_matrix,
      0.5,
      0.1,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      wrong_size_synapse,
      pre_population,
      scale_matrix,
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )

    @test_throws AssertionError PNN.RatePlasticityScaledCovariance(
      post_population,
      synapse,
      pre_population,
      ones(2,2),
      0.5,
      0.1,
      0.2,
      covariance_estimator,
    )
  end

  @testset "RatePlasticityCovarianceQuadraticallyStabilized" begin
    post_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      2,
      initial_rates=[1.0,2.0],
    )
    pre_population = PNN.LinearRateNeuralPopulation(
      PNN.ExcitatoryRateNeuron(1.0;rate_saturation=100.0),
      3,
      initial_rates=[3.0,4.0,5.0],
    )
    post_mean_estimator = PNN.RateMeanEstimator(post_population,1.0,0.1)
    pre_mean_estimator = PNN.RateMeanEstimator(pre_population,1.0,0.1)
    covariance_estimator = PNN.RateCovarianceEstimator(
      post_mean_estimator,
      pre_mean_estimator,
    )
    covariance_estimator.covariance_now .= [
      2.0  -1.0  -100.0
      0.5   1.5    -2.0
    ]
    synapse = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=3.01,
    )

    @test rule.pop_pre === pre_population
    @test rule.pop_post === post_population
    @test rule.synapses_post_pre === synapse
    @test rule.covariance_estimator === covariance_estimator
    @test rule.α1 == -0.5
    @test rule.α2 == 0.25
    @test rule.t_last_update == -Inf

    rule.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule) === nothing
    @test synapse.weights == [
      1.0  0.0  1.0
      2.0  3.0  4.0
    ]
    @test rule.t_last_update == 0.0

    @test PNN.plasticity!(0.1,0.01,rule) === nothing
    @test isapprox(synapse.weights,[
      0.985  0.0   2.005
      2.015  3.01  3.01
    ]; rtol=1e-12)
    @test rule.t_last_update == 0.1

    synapse_small_dt = PNN.RateLinearSynapses([
      1.0  0.0  1.0
      2.0  3.0  4.0
    ])
    rule_small_dt = PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse_small_dt,
      pre_population,
      0.05,
      -0.5,
      0.25,
      0.2,
      covariance_estimator;
      w_min=0.1,
      w_max=10.0,
    )
    rule_small_dt.t_last_update = 0.0
    @test PNN.plasticity!(0.05,0.01,rule_small_dt) === nothing
    @test isapprox(
      synapse.weights[1,1] - 1.0,
      2 * (synapse_small_dt.weights[1,1] - 1.0);
      rtol=1e-12,
    )

    swapped_covariance_estimator = PNN.RateCovarianceEstimator(
      pre_mean_estimator,
      post_mean_estimator,
    )
    @test_throws AssertionError PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      swapped_covariance_estimator,
    )

    wrong_size_synapse = PNN.RateLinearSynapses(ones(2,2))
    @test_throws AssertionError PNN.RatePlasticityCovarianceQuadraticallyStabilized(
      post_population,
      wrong_size_synapse,
      pre_population,
      0.1,
      -0.5,
      0.25,
      0.2,
      covariance_estimator,
    )
  end

  @testset "generate_ring_topology" begin
    locations = PNN.place_neurons_on_ring(4; offset=0.5)
    @test isapprox(locations,0.5 .+ [0.0,π / 2,π,3π / 2]; rtol=1e-12)
    wrapped_locations = PNN.place_neurons_on_ring(4; offset=3π / 2)
    @test all(0.0 .<= wrapped_locations .< 2π)
    @test isapprox(wrapped_locations,[3π / 2,0.0,π / 2,π]; rtol=1e-12, atol=1e-12)
    negative_offset_locations = PNN.place_neurons_on_ring(4; offset=-π / 2)
    @test all(0.0 .<= negative_offset_locations .< 2π)
    @test isapprox(negative_offset_locations,[3π / 2,0.0,π / 2,π]; rtol=1e-12, atol=1e-12)

    scale_factor = 2.5
    kappa = 3.0
    locations_post, locations_pre, topology = PNN.generate_ring_topology(
      4,
      4;
      scale_factor=scale_factor,
      kappa=kappa,
    )

    @test isapprox(locations_post,[0.0,π / 2,π,3π / 2]; rtol=1e-12)
    @test isapprox(locations_pre,[0.0,π / 2,π,3π / 2]; rtol=1e-12)
    @test size(topology) == (4,4)
    @test all(isapprox.(diag(topology),scale_factor; rtol=1e-12))
    @test isapprox(topology[1,2],scale_factor * exp(-kappa); rtol=1e-12)
    @test isapprox(topology[1,4],scale_factor * exp(-kappa); rtol=1e-12)
    @test isapprox(topology[1,3],scale_factor * exp(-2 * kappa); rtol=1e-12)

    _, _, rectangular_topology = PNN.generate_ring_topology(
      2,
      3;
      scale_factor=scale_factor,
      kappa=kappa,
    )
    @test size(rectangular_topology) == (2,3)
    @test isapprox(maximum(rectangular_topology),scale_factor; rtol=1e-12)
  end
end
