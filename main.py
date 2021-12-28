from ParallelTVGL import *
from args import *
from Params import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_time = time.time()
    parameters = get_parser()
    args = parameters.parse_args()
    filename = None
    penalty_function = args.penalty_function
    left_step = args.leftstep
    right_step = args.rightstep
    blocks = left_step + right_step
    samplePerStep = args.samplePerStep
    dimension = args.dimension
    real_data = True
    time_set = timing_set(101, samplePerStep, left_step, samplePerStep, right_step)
    stock_list = [2, 321, 30, 241, 48, 180]
    newADMMPara = ADMMParam
    newADMMPara.LAMBDA = args.alpha
    newADMMPara.BETA = args.beta
    print("\nReading file: %s\n" % filename)
    solver = ParallelTVGL(filename=filename,
                          penalty_function=penalty_function,
                          blocks=blocks,
                          samplePerStep=samplePerStep,
                          dimension=dimension,
                          time_set=time_set,
                          stock_list=stock_list,
                          read_data_function=None,
                          datecolumn=real_data)
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
    company_list = get_company_list(stock_list)
    company_list_list = list(company_list)
    time_param = [101, samplePerStep, 3, samplePerStep, 12]
    log_path = get_log_path(stock_list, time_param, samplePerStep, penalty_function, time_set)
    if not Path(log_path).exists():
        os.makedirs(log_path + "/")

    save_matrix_plot_exact_number(solver.thetas, time_set, company_list_list, log_path +
                                  '/P_theta_exact_number_{}_{}.png'.format(solver.lambd, solver.beta))
    save_matrix_plot(solver.thetas, time_set, company_list_list, log_path +
                     '/P_theta_{}_{}.png'.format(solver.lambd, solver.beta))
    save_line_plot(solver.deviations, time_set, solver.lambd, solver.beta, solver.time_span, log_path +
                   '/line_{}_{}.png'.format(solver.lambd, solver.beta),
                   samplePerStep, penalty_function)
    print("\nTemporal deviations: ")
    print(solver.deviations)
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

