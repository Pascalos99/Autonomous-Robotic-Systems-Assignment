import simple_PSO as pso
import numpy as np
import pandas as pd

# two measures:
# 1. convergence time (requires threshold attained for 5 consecutive steps of PSO)
#    1A. to within 0.1 of solution
#    1B. to within 0.01 of solution
# 2. solution accuracy
#    2A. after 50 steps
#    2B. after 100 steps

class Threshold:
    def __init__(self):
        self.reset()
    def increment(self):
        self.counter = self.counter + 1
    def set_count(self, value):
        self.counter = value
    def set_reached(self, value):
        self.reached = value
    def reset(self):
        self.counter = 0
        self.reached = False

def conditioned_PSO(swarm, dt, min_iters, tightest_convergence_threshold, optimal_pos=np.array([0,0]), convergence_check_length=5, max_iters=100000, **PSO_kwargs):
    best_xs, best_ys = [], []

    threshold = Threshold()

    def stop_condition(iter: int, best_particle: pso.Particle, second_best_particle: pso.Particle):
        best_xs.append(best_particle.pos)
        best_ys.append(best_particle.prf)

        if not threshold.reached:
            diff = best_particle.pos - optimal_pos
            if diff.dot(diff) <= tightest_convergence_threshold**2:
                threshold.increment()
                if threshold.counter >= convergence_check_length:
                    threshold.set_reached(True)
            else:
                threshold.set_count(0)
        return threshold.reached and (iter >= min_iters)

    pso.PSO(None, swarm, max_iters, dt, stop_condition=stop_condition, **PSO_kwargs)
    return best_xs, best_ys

def extract_convergence_time(best_xs: list, convergence_threshold: float, optimal_pos=np.array([0,0]), convergence_check_length: int=5):
    threshold = Threshold()
    for i in range(len(best_xs)):
        diff = best_xs[i] - optimal_pos
        if diff.dot(diff) <= convergence_threshold**2:
            threshold.increment()
            if threshold.counter >= convergence_check_length:
                return i - convergence_check_length + 1
        else:
            threshold.set_count(0)
    return -1

def build_duo_test(swarm_paramss: list, PSO_paramss: list, iter_checkpoints: list, convergence_thresholds: list, optimal_pos=np.array([0,0]),convergence_check_length: int=5, max_iters: int=100000):
    # swarm_paramss and PSO_paramss are lists of dictionaries containing all necessary parameters for PSO (except "grapher", "swarm", "num_iters", and "stop_condition" which will be overridden anyways)
    results_checkpoints = {check: [None for exp_id in range(len(swarm_paramss))] for check in iter_checkpoints}
    results_convergence = {convr: [None for exp_id in range(len(swarm_paramss))] for convr in convergence_thresholds}
    min_iters = max(iter_checkpoints)
    tightest_threshold = min(convergence_thresholds)
    def complete_experiment(extra_execute=lambda exp_id: None):
        for exp_id in range(len(swarm_paramss)):
            swarm = pso.init_swarm(**swarm_paramss[exp_id])
            best_xs, best_ys = conditioned_PSO(swarm, min_iters=min_iters, tightest_convergence_threshold=tightest_threshold, optimal_pos=optimal_pos, convergence_check_length=convergence_check_length, max_iters=max_iters, **PSO_paramss[exp_id])
            for check in iter_checkpoints:
                results_checkpoints[check][exp_id] = best_ys[check]
            for thres in convergence_thresholds:
                results_convergence[thres][exp_id] = extract_convergence_time(best_xs, thres, optimal_pos, convergence_check_length)
            extra_execute(exp_id)
        return True
    return complete_experiment, results_checkpoints, results_convergence

def average_result_from_test(repetitions: int, swarm_paramss: list, PSO_paramss: list, iter_checkpoints: list, convergence_thresholds: list, **duo_test_kwargs):
    complete_experiment, results_checkpoints, results_convergence = build_duo_test(swarm_paramss * repetitions, PSO_paramss * repetitions, iter_checkpoints, convergence_thresholds, **duo_test_kwargs)
    avg_results_checkpoints = {check: [-1 for real_id in range(len(swarm_paramss))] for check in iter_checkpoints}
    avg_results_convergence = {convr: [-1 for real_id in range(len(swarm_paramss))] for convr in convergence_thresholds}
    convergence_rates = {convr: [0 for real_id in range(len(swarm_paramss))] for convr in convergence_thresholds}
    did_it_converge = {convr: [False for exp_id in range(len(swarm_paramss) * repetitions)] for convr in convergence_thresholds}
    current_counts = [0 for real_id in swarm_paramss]
    def full_experiment():
        def extra_execute(exp_id):
            real_id = exp_id % len(swarm_paramss)
            current_counts[real_id] += 1
            for check in iter_checkpoints:
                avg_results_checkpoints[check][real_id] = (avg_results_checkpoints[check][real_id] * (current_counts[real_id]-1) + results_checkpoints[check][exp_id]) / current_counts[real_id]
            for thres in convergence_thresholds:
                real_count = convergence_rates[thres][real_id] * (current_counts[real_id] - 1)
                did_it_converge[thres][exp_id] = results_convergence[thres][exp_id] >= 0
                convergence_rates[thres][real_id] = float(convergence_rates[thres][real_id] * (current_counts[real_id] - 1) + int(did_it_converge[thres][exp_id])) / current_counts[real_id]
                if did_it_converge[thres][exp_id]:
                    avg_results_convergence[thres][real_id] = (avg_results_convergence[thres][real_id] * real_count + results_convergence[thres][exp_id]) / (real_count + 1)
        complete_experiment(extra_execute)
    return full_experiment, avg_results_checkpoints, avg_results_convergence, convergence_rates

# DEFINE FUNCTION
def rosenbrock(x):
    A, B = 0, 100
    return (A - x[0])**2 + B * (x[1] - x[0]**2)**2

def rastrigin(x):
    return 2 * 10 + ((x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))
#################

def general_experiment(num_tests, iter_checkpoints, testing_thresholds, functions, x_ranges, num_particless, abcs, abc_deltas, dts, neighbor_protocols, neighbor_params, remember_global_bests, max_iters=1000):

    def name_functions(func):
        if func is rosenbrock: return "rosenbrock"
        if func is rastrigin: return "rastrigin"
        return "custom"

    print("start experiment")
    param_names = ["_".join(["f="+name_functions(_function), "N%d"%_num_particles, "x:%s"%list(_x_range), "abc:%s"%list(_abc), "dabc-"+name_functions(_abc_delta), "dt=%e"%_dt, 'P="%s"'%_neighbor_protocol, 'P-param="%s"'%_neighbor_param, 'R=%s'%_remember_global_best]) for _function in functions for _num_particles in num_particless for _x_range in x_ranges for _abc in abcs for _abc_delta in abc_deltas for _dt in dts for _neighbor_protocol in neighbor_protocols for _neighbor_param in neighbor_params for _remember_global_best in remember_global_bests]
    swarm_paramss = [{"function": _function, "num_particles":_num_particles, "x_range": _x_range, "x_shape": (2,), "abc": _abc, "remember_global_best_of_all_time": _remember_global_best} for _function in functions for _num_particles in num_particless for _x_range in x_ranges for _abc in abcs for _abc_delta in abc_deltas for _dt in dts for _neighbor_protocol in neighbor_protocols for _neighbor_param in neighbor_params for _remember_global_best in remember_global_bests]
    pso_paramss = [{"dt": _dt, "neighbor_setting": _neighbor_protocol, "neighbor_param": _neighbor_param, "abc_delta": _abc_delta} for _function in functions for _num_particles in num_particless for _x_range in x_ranges for _abc in abcs for _abc_delta in abc_deltas for _dt in dts for _neighbor_protocol in neighbor_protocols for _neighbor_param in neighbor_params for _remember_global_best in remember_global_bests]
    experiment, results_checks, results_converge, convergence_rates = average_result_from_test(num_tests, swarm_paramss, pso_paramss, iter_checkpoints, testing_thresholds, max_iters = max_iters)
    experiment()
    dfchecks = pd.DataFrame(results_checks)
    dfconv = pd.DataFrame(results_converge)
    dfrate = pd.DataFrame(convergence_rates)
    print("checkpoint performance:")
    print(dfchecks)
    print("convergence times:")
    print(dfconv)
    print("convergence rates:")
    print(dfrate)
    print("param names:")
    for name in param_names: print(name)
    print("end of experiment")

def experiment1():
    x_range = (-3, 3)
    abc   = (0.9, 2., 2.) # a, b, and c at the start of the simulation
    n_abc = (0.4, 2., 2.) # a, b, and c at the end of the simulation
    num_particless = [5, 10, 15, 20, 25, 50, 100, 150]
    num_iters = 250 # only relevant for abc_delta
    dt = 0.1
    neighbor_protocol = "exclusive_global"
    # choose from: "inclusive_global"      "exclusive_global"
    #              "inclusive_fixed"       "exclusive_fixed"
    #              "inclusive_geographic"  "exclusive_geographic"
    neighbor_param = 1
    # if neighbor_protocol is 'geographic', neighbor_param is the geographic radius within which neighbors are defined
    # if neighbor_protocol is 'fixed', neighbor_param is the number of fixed neighborhoods to create
    def abc_delta(iter):
        if iter < num_iters:
            return tuple([(n_abc[i] - abc[i]) / num_iters for i in range(len(abc))])
        return (0,0,0)
    remember_global_best = False
    func = rosenbrock

    num_tests = 30
    iter_checkpoints = [50, 100, 150, 200, 250]
    testing_thresholds = [1, 0.5, 0.25, 0.1, 0.05, 0.01]

    general_experiment(num_tests, iter_checkpoints, testing_thresholds, [func], [x_range], num_particless, [abc], [abc_delta], [dt], [neighbor_protocol], [neighbor_param], [remember_global_best])

if __name__ == "__main__":
    experiment1()
    
def old_main():
    # PARAMETERS -  you can change these!
    x_range = (-3, 3)
    init_x_range = x_range
    x_shape = (2,)
    abc   = (0.9, 2., 2.) # a, b, and c at the start of the simulation
    n_abc = (0.4, 2., 2.) # a, b, and c at the end of the simulation
    num_particles = 20
    num_iters = 250
    dt = 0.1
    neighbor_protocol = "exclusive_global"
    # choose from: "inclusive_global"      "exclusive_global"
    #              "inclusive_fixed"       "exclusive_fixed"
    #              "inclusive_geographic"  "exclusive_geographic"
    neighbor_param = 1.5
    # if neighbor_protocol is 'geographic', neighbor_param is the geographic radius within which neighbors are defined
    # if neighbor_protocol is 'fixed', neighbor_param is the number of fixed neighborhoods to create

    # EXPERIMENTAL PARAMETERS
    def abc_delta(iter):
        if iter < num_iters:
            return tuple([(n_abc[i] - abc[i]) / num_iters for i in range(len(abc))])
        return (0,0,0)
    particle_kwargs = {"remember_global_best_of_all_time": False} #True} # not sure if this is good?
    
    # SET FUNCTION TO OPTIMIZE
    func = rosenbrock

    # RUN SIMULATION
    print("start")
    # swarm = pso.init_swarm(num_particles, func, init_x_range, x_shape, abc, **particle_kwargs)
    # pso.PSO(None, swarm, num_iters, dt, neighbor_protocol, neighbor_param, abc_delta)
    swarm_paramss = [{"num_particles":num_particles, "function": func, "x_range": init_x_range, "x_shape": x_shape, "abc": abc, "remember_global_best_of_all_time": False}]
    pso_paramss = [{"dt": dt, "neighbor_setting": neighbor_protocol, "neighbor_param": neighbor_param, "abc_delta": abc_delta}]
    experiment, results_checks, results_converge, convergence_rates = average_result_from_test(1000, swarm_paramss, pso_paramss, [50, 100, 150, 200, 250], [1, 0.5, 0.25, 0.1, 0.05, 0.01], max_iters = 1000)
    experiment()
    print(results_checks)
    print(results_converge)
    print(convergence_rates)
    print("done")