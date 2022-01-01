from ParallelTVGL import *
from args import *
from Params import *
from typing import Union, ByteString, Text, Tuple, Dict, Callable, Any, List

class StockExperiment:
    def __init__(self, stock_list, time_set, penalty_function, blocks, sample_per_step):
        self.ADMMPara = ADMMParam
        self.stock_list: List = stock_list
        self.time_set: List = time_set
        self.penalty_function: str = penalty_function
        self.blocks: int = blocks
        self.dimension: int = len(self.stock_list)
        self.read_data_function: Union[Callable, None] = None
        self.func_args: Union[tuple, None] = None
        self.sample_per_step: int = sample_per_step

    def set_read_data_function(self, read_data_function, func_args):
        self.read_data_function = read_data_function
        self.func_args = func_args


    def run(self):
        start_time = time.time()
        solver = ParallelTVGL(penalty_function=self.penalty_function,
                              blocks=self.blocks,
                              sampleperstep=self.sample_per_step,
                              dimension=self.dimension,
                              time_set=self.time_set,
                              stock_list=self.stock_list,
                              read_data_function=self.read_data_function)
        print("Total data samples: %s" % solver.samplePerStep * solver.blocks)
        print("Blocks: %s" % solver.blocks)
        print("Observations in a block: %s" % solver.obs)
        print("Rho: %s" % solver.rho)
        print("Lambda: %s" % solver.lambd)
        print("Beta: %s" % solver.beta)
        print("Penalty function: %s" % solver.penalty_function)
        print("Processes: %s" % solver.processes)

        """ Run algorithm """
        print("\nRunning algorithm...")
        solver.run_algorithm()

        """ Evaluate and print results """
        print("\nNetwork 0:")
        for j in range(solver.dimension):
            print(solver.thetas[10][j, :])
        company_list = get_company_list(self.stock_list)
        company_list_list = list(company_list)
        time_param = [101, self.sample_per_step, 3, self.sample_per_step, 12]
        log_path = get_log_path(self.stock_list, time_param, self.sample_per_step, self.penalty_function, self.time_set)
        if not Path(log_path).exists():
            os.makedirs(log_path + "/")

        save_matrix_plot_exact_number(solver.thetas, self.time_set, company_list_list, log_path +
                                      '/P_theta_exact_number_{}_{}.png'.format(solver.lambd, solver.beta))
        save_matrix_plot(solver.thetas, self.time_set, company_list_list, log_path +
                         '/P_theta_{}_{}.png'.format(solver.lambd, solver.beta))
        save_TD_matrix(solver.deviations, self.time_set, company_list_list, log_path +
                         '/TD_Matrix_{}_{}.png'.format(solver.lambd, solver.beta))
        save_line_plot(solver.deviations_value, self.time_set, solver.lambd, solver.beta, solver.time_span, log_path +
                       '/line_{}_{}.png'.format(solver.lambd, solver.beta),
                       self.sample_per_step, self.penalty_function)
        print("\nTemporal deviations: ")
        print(solver.deviations_value)
        print("Normalized Temporal deviations: ")
        print(solver.norm_deviations)
        try:
            print("Temp deviations ratio: {0:.3g}".format(solver.dev_ratio))
        except ValueError:
            print("Temp deviations ratio: n/a")

        """ Evaluate and create result file """

        """ Running times """
        print("\nAlgorithm run time: %s seconds" % (solver.time_span))
        print("Execution time: %s seconds" % (time.time() - start_time))