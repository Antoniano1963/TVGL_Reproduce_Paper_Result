
from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import time
from utils import *
import sys


class SerialTVGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes TVGL problem in serial,
    # no parallelization

    def __init__(self, *args, **kwargs):
        super(SerialTVGL, self).__init__(processes=1, *args, **kwargs)

    def theta_update(self):
        for i in range(self.blocks):
            a = (self.z0s[i] + self.z1s[i] + self.z2s[i] -
                 self.u0s[i] - self.u1s[i] - self.u2s[i])/3
            at = a.transpose()
            m = self.eta*(a + at)/2 - self.emp_cov_mat[i]
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/self.eta*np.ones(self.dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            # self.thetas[i] = np.real(
            #     self.eta/2*np.dot(np.dot(q, diagonal), qt))
            self.thetas[i] = self.eta/2*np.dot(np.dot(q, diagonal), qt)

    def z_update(self):
        self.z0_update()
        self.z1_z2_update()

    def z0_update(self):
        self.z0s = [pf.soft_threshold_odd(
            self.thetas[i] + self.u0s[i], self.lambd, self.rho)
                    for i in range(self.blocks)]

    def z1_z2_update(self):
        if self.penalty_function == "perturbed_node":
            dimension = self.dimension
            c = np.zeros((dimension, 3 * dimension))
            c[:, 0:dimension] = np.eye(dimension)
            c[:, dimension:2 * dimension] = -np.eye(dimension)
            c[:, 2 * dimension:3 * dimension] = np.eye(dimension)
            ct = c.transpose()
            cc = np.linalg.inv(np.dot(ct, c) + 2 * np.eye(3 * dimension))
            for i in range(1, self.blocks):
                self.z1s[i-1], self.z2s[i] = pf.perturbed_node(self.thetas[i-1],
                                                               self.thetas[i],
                                                               self.u1s[i-1],
                                                               self.u2s[i],
                                                               self.beta,
                                                               self.rho,
                                                               ct,
                                                               cc)
        else:

            aa = [self.thetas[i] - self.thetas[i-1] + self.u2s[i] - self.u1s[i-1]
                  for i in range(1, self.blocks)]
            ee = [getattr(pf, self.penalty_function)(a, self.beta, self.rho)
                  for a in aa]
            for i in range(1, self.blocks):
                summ = self.thetas[i-1] + self.thetas[i] + self.u1s[i-1] + self.u2s[i]
                self.z1s[i-1] = 0.5*(summ - ee[i-1])
                self.z2s[i] = 0.5*(summ + ee[i-1])

    def u_update(self):
        for i in range(self.blocks):
            self.u0s[i] = self.u0s[i] + self.thetas[i] - self.z0s[i]
        for i in range(1, self.blocks):
            self.u2s[i] = self.u2s[i] + self.thetas[i] - self.z2s[i]
            self.u1s[i-1] = self.u1s[i-1] + self.thetas[i-1] - self.z1s[i-1]


# def timing_set(center, samplesPerStep_left, count_left, samplesPerStep_right, count_right):
#     time_set = []
#     count_left = min(count_left, center / samplesPerStep_left)
#     print('left timesteps: = ', count_left)
#     start = max(center - samplesPerStep_left * (count_left), 0)
#     for i in range(count_left):
#         time_interval = [start, start + samplesPerStep_left - 1]
#         time_set.append(time_interval)
#         start = start + samplesPerStep_left
#     count_right = min(count_right, 245 / samplesPerStep_left)
#     print('right timesteps: = ', count_right)
#     for i in range(count_right):
#         time_interval = [start, start + samplesPerStep_right - 1]
#         time_set.append(time_interval)
#         start = start + samplesPerStep_right
#     return time_set
#
#
#
# def save_matrix_plot(theta_est_list, time_set, company_list_list, path):
#     import matplotlib.pylab as pl
#     row_num = np.ceil(len(time_set) / 5.0)
#     fig = pl.figure(figsize=(20, 4.5 * row_num))
#     for i in range(len(time_set)):
#         theta = theta_est_list[i]
#         theta[theta != 0] = 1
#         time = TIME_MAP[time_set[i][0]] + '~' + TIME_MAP[time_set[i][-1]]
#         fig.add_subplot(row_num, 5, i + 1)
#         pl.imshow(theta, cmap='gray_r', interpolation='nearest')
#         pl.title(time)
#         ax = pl.gca()
#         ticks = []
#         for j in range(len(company_list_list)):
#             ticks.append(j)
#         # ticks = [i for i in range(len(list(company_list)))]
#         ax.set_xticks(ticks)
#         ax.set_yticks(ticks)
#         ax.set_xticklabels(company_list_list)
#         ax.set_yticklabels(company_list_list)
#     pl.savefig(path, dpi=300, bbox_inches='tight')
#
#
# def save_matrix_plot_exact_number(theta_est_list, time_set, company_list_list, path):
#     import matplotlib.pylab as pl
#     row_num = np.ceil(len(time_set) / 5.0)
#     fig = pl.figure(figsize=(20, 4.5 * row_num))
#     for i in range(len(time_set)):
#         # theta = theta_est_list[i]
#         # row, col = theta.shape
#         # for i in range(row):
#         #     for j in range(col):
#         #         theta[i, j] = abs(theta[i, j])
#         # theta[theta != 0] = abs(1)
#         abs_theta = np.abs(theta_est_list[i])
#         row, col = abs_theta.shape
#         # abs_theta[abs_theta == 0] = 1
#         for j in range(row):
#             abs_theta[j, j] = -1
#         max_num = np.max(abs_theta)
#         for j in range(row):
#             abs_theta[j, j] = max_num
#         abs_theta[abs_theta == 0] = -0.1
#         abs_theta = abs_theta * 1000
#         time = TIME_MAP[time_set[i][0]] + '~' + TIME_MAP[time_set[i][-1]]
#         fig.add_subplot(int(row_num), 5, i + 1)
#         pl.imshow(abs_theta, cmap='PuBu', interpolation='nearest')
#         pl.title(time)
#         ax = pl.gca()
#         ticks = []
#         for j in range(len(company_list_list)):
#             ticks.append(j)
#         # ticks = [i for i in range(len(list(company_list)))]
#         ax.set_xticks(ticks)
#         ax.set_yticks(ticks)
#         ax.set_xticklabels(company_list_list)
#         ax.set_yticklabels(company_list_list)
#     pl.savefig(path, dpi=300, bbox_inches='tight')
#
#
# def get_time_map() -> object:
#     m = []
#     with open('time.txt', 'r') as f:
#         for line in f:
#             line = line.replace('\n', '')
#             m.append(line[:4] + '-' + line[4:6] + '-' + line[6:])
#     return m
#
# def get_company_list(stock_list):
#     return map(lambda x : COR[x], stock_list)
#
#
# def getStocks():
#     timesteps = len(time_set)
#     sample_data_set = []
#     empCov_set = []
#     stock_data = np.genfromtxt('Datasets/finance.csv', delimiter=',')
#     size = len(stock_list)
#     for i in range(timesteps):
#         time_interval = time_set[i]
#         sample_data = stock_data[time_interval[0]:time_interval[1] + 1, stock_list].T
#         sample_data_set.append(sample_data)
#         empCov_set.append(genEmpCov(sample_data))
#     return empCov_set
#
# def genEmpCov(samples, useKnownMean=False, m=0):
#     size, samplesPerStep = samples.shape
#     if useKnownMean == False:
#         m = np.mean(samples, axis=1)
#     empCov = 0
#     for i in range(samplesPerStep):
#         sample = samples[:, i]
#         empCov = empCov + np.outer(sample - m, sample - m)
#     empCov = empCov / samplesPerStep
#     return empCov


if __name__ == "__main__" :

    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. Penalty function
    #     1 = "element_wise"
    #     2 = "group_lasso"
    #     3 = "perturbed_node"
    #  3. Number of blocks to be created
    #  4. lambda
    #  5. beta

    start_time = time.time()
    datahandler = DataHandler()
    TIME_MAP = get_time_map()

    """ Parameters for creating solver instance """
    # filename = sys.argv[1]
    # real_data = True
    # if "synthetic_data" in filename:
    #     real_data = False
    # if sys.argv[2] == "1":
    #     penalty_function = "element_wise"
    # elif sys.argv[2] == "2":
    #     penalty_function = "group_lasso"
    # elif sys.argv[2] == "3":
    #     penalty_function = "perturbed_node"
    # blocks = int(sys.argv[3])
    # lambd = float(sys.argv[4])
    # beta = float(sys.argv[5])
    filename = None
    penalty_function = "element_wise"
    blocks = 15
    lambd = 0.17
    beta = 5
    samplePerStep = 7
    dimension = 6
    real_data = True
    time_set = timing_set(101, samplePerStep, 3, samplePerStep, 12)
    stock_list = [2, 321, 30, 241, 48, 180]


    """ Create solver instance """
    print("\nReading file: %s\n" % filename)
    solver = SerialTVGL(filename=filename,
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
        print(solver.thetas[0][j, :])
    company_list = get_company_list(stock_list)
    company_list_list = list(company_list)
    save_matrix_plot_exact_number(solver.thetas, time_set, company_list_list, './theta_exact_number.png')
    save_matrix_plot(solver.thetas, time_set, company_list_list, './theta.png')
    save_line_plot(solver.deviations, time_set, solver.lambd, beta, solver.time_span,'./line.png',
                   samplePerStep, penalty_function)
    print("\nTemporal deviations: ")
    # solver.temporal_deviations()
    print(solver.deviations)
    print("Normalized Temporal deviations: ")
    print(solver.norm_deviations)
    try:
        print("Temp deviations ratio: {0:.3g}".format(solver.dev_ratio))
    except ValueError:
        print("Temp deviations ratio: n/a")

    """ Evaluate and create result file """
    # if not real_data:
    #     solver.correct_edges()
    #     print("\nTotal Edges: %s" % solver.real_edges)
    #     print("Correct Edges: %s" % solver.correct_positives)
    #     print("Total Zeros: %s" % solver.real_edgeless)
    #     false_edges = solver.all_positives - solver.correct_positives
    #     print("False Edges: %s" % false_edges)
    #     print("F1 Score: %s" % solver.f1score)
    #     datahandler.write_results(filename, solver)
    # else:
    #     datahandler.write_network_results(filename, solver)

    """ Running times """
    print("\nAlgorithm run time: %s seconds" % (solver.run_time))
    print("Execution time: %s seconds" % (time.time() - start_time))



