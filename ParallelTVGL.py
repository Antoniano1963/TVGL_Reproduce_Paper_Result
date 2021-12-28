import os
from pathlib import Path

from BaseGraphicalLasso import BaseGraphicalLasso
from DataHandler import DataHandler
import penalty_functions as pf
import numpy as np
import time
from utils import COR
import sys
from utils import *
from multiprocessing import cpu_count
from multiprocessing import Pool
import copy


class ParallelTVGL(BaseGraphicalLasso):

    # Child class of BaseGraphicalLasso class.
    # Computes TVGL problem in serial,
    # no parallelization

    def __init__(self, *args, **kwargs):
        super(ParallelTVGL, self).__init__(processes=cpu_count(), *args, **kwargs)

    def run_algorithm(self, max_iter=10000):
        self.init_algorithm()
        self.iteration = 0
        stopping_criteria = False
        thetas_pre = []
        start_time = time.time()
        with Pool(self.processes) as pool:
            while self.iteration < max_iter and stopping_criteria == False:
                if self.iteration % 500 == 0 or self.iteration == 1:
                    print("\n*** Iteration %s ***" % self.iteration)
                    print("Time passed: {0:.3g}s".format(time.time() - start_time))
                    print("Rho: %s" % self.rho)
                    print("Eta: %s" % self.eta)
                    print("Step: {0:.3f}".format(1/(2*self.eta)))
                if self.iteration % 500 == 0 or self.iteration == 1:
                    s_time = time.time()
                multiple_results = []
                for i in range(self.blocks):
                    multiple_results.append(pool.apply_async(theta_z0_update,
                                                             (self.thetas[i], self.z1s[i],
                                                              self.z2s[i], self.eta, self.emp_cov_mat[i], self.dimension
                                                              , self.u1s[i], self.u2s[i], self.u0s[i], self.lambd,
                                                              self.rho,)))
                for i in range(self.blocks):
                    result = multiple_results[i].get()
                    self.thetas[i] = result[0]
                    self.z0s[i] = result[1]

                if self.iteration % 500 == 0 or self.iteration == 1:
                    print("Theta update: {0:.3g}s".format(time.time() - s_time))
                if self.iteration % 500 == 0 or self.iteration == 1:
                    s_time = time.time()
                multiple_results = []
                for i in range(1, self.blocks):
                    multiple_results.append(pool.apply_async(z1_z2_update,
                                                         (self.dimension, self.penalty_function,
                                                          self.thetas[i], self.thetas[i-1], self.u1s[i], self.u2s[i],
                                                          self.u1s[i-1], self.u2s[i-1], self.beta, self.rho,)))
                for i in range(1, self.blocks):
                    result = multiple_results[i-1].get()
                    self.z1s[i-1] = result[0]
                    self.z2s[i] = result[1]

                if self.iteration % 500 == 0 or self.iteration == 1:
                    print("Z-update: {0:.3g}s".format(time.time() - s_time))
                if self.iteration % 500 == 0 or self.iteration == 1:
                    s_time = time.time()
                self.u0s[0] = self.u0s[0] + self.thetas[0] - self.z0s[0]
                multiple_results = []
                for i in range(1, self.blocks):
                    multiple_results.append(pool.apply_async(u_update,
                                                             (self.thetas[i], self.u0s[i], self.z0s[i], self.u2s[i],
                                                              self.z2s[i], self.u1s[i], self.z1s[i], self.u1s[i-1],
                                                              self.thetas[i-1], self.z1s[i-1],)))
                for i in range(1, self.blocks):
                    result = multiple_results[i-1].get()
                    self.u0s[i] = result[0]
                    self.u1s[i-1] = result[1]
                    self.u2s[i] = result[2]
                if self.iteration % 500 == 0 or self.iteration == 1:
                    print("U-update: {0:.3g}s".format(time.time() - s_time))
                """ Check stopping criteria """
                if self.iteration % 500 == 0 or self.iteration == 1:
                    s_time = time.time()
                if self.iteration > 0:
                    ##这个是原来的收敛判断条件，把它换成了ADMM的标准收敛判断
                    # fro_norm = 0
                    # for i in range(self.blocks):
                    #     dif = self.thetas[i] - thetas_pre[i]
                    #     fro_norm += np.linalg.norm(dif)
                    # if fro_norm < self.e:
                    #     stopping_criteria = True
                    info = self.check_Convergence()
                    stopping_criteria = info[0]
                    if(self.iteration % 500 ==0 ):
                        print("res_pri: {}, e_pri: {}, res_dual: {}, e_dual: {}".format(info[1], info[2], info[3], info[4],))
                        print(self.thetas[1])
                thetas_pre = list(self.thetas)
                self.iteration += 1
                self.pre_z0s = copy.deepcopy(self.z0s)
                self.pre_z1s = copy.deepcopy(self.z1s)
                self.pre_z2s = copy.deepcopy(self.z2s)
                self.pre_u0s = copy.deepcopy(self.u0s)
                self.pre_u1s = copy.deepcopy(self.u1s)
                self.pre_u2s = copy.deepcopy(self.u2s)

        self.time_span = "{0:.3g}".format(time.time() - start_time)
        self.final_tuning(stopping_criteria, max_iter)


"""theta z0 u0 一个线程多计算一点 z1 z2一个进程少计算一点"""

def theta_z0_update(z0, z1, z2, eta, emp_cov_mat, dimension, u1, u2, u0, lambd, rho):
    a = (z0 + z1 + z2 -
         u0 - u1 - u2)/3
    at = a.transpose()
    m = eta*(a + at)/2 - emp_cov_mat
    d, q = np.linalg.eig(m)
    qt = q.transpose()
    sqrt_matrix = np.sqrt(d**2 + 4/eta*np.ones(dimension))
    diagonal = np.diag(d) + np.diag(sqrt_matrix)
    new_theta = eta/2*np.dot(np.dot(q, diagonal), qt)
    new_z0 = pf.soft_threshold_odd(new_theta + u0, lambd, rho)
    return [new_theta, new_z0]


def z0_update(thetas, u0s, lambd, rho, blocks):
    new_z0s = []
    for i in range(blocks):
        new_z0s.append(pf.soft_threshold_odd(thetas[i] + u0s[i], lambd, rho))



def z1_z2_update(dimension, penalty_function, theta, theta_p, u1, u2, u1_p, u2_p, beta, rho):
    if penalty_function == "perturbed_node":
        c = np.zeros((dimension, 3 * dimension))
        c[:, 0:dimension] = np.eye(dimension)
        c[:, dimension:2 * dimension] = -np.eye(dimension)
        c[:, 2 * dimension:3 * dimension] = np.eye(dimension)
        ct = c.transpose()
        cc = np.linalg.inv(np.dot(ct, c) + 2 * np.eye(3 * dimension))
        new_z1, new_z2 = pf.perturbed_node(theta_p,
                                           theta,
                                           u1_p,
                                           u2,
                                           beta,
                                           rho,
                                           ct,
                                           cc)
    else:

        a = theta - theta_p + u2 - u1_p
        e = getattr(pf, penalty_function)(a, beta, rho)
        summ = theta_p + theta + u1_p + u2
        new_z1 = 0.5*(summ - e)
        new_z2 = 0.5*(summ + e)
    return [new_z1, new_z2]

def u_update(theta, u0, z0, u2, z2, u1, z1, u1_p, theta_p, z1_p):
    new_u0 = u0 + theta - z0
    new_u2 = u2 + theta - z2
    new_u1 = u1_p + theta_p - z1_p
    return [new_u0, new_u1, new_u2]





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
    penalty_function = "perturbed_node"
    blocks = 15
    samplePerStep = 6
    dimension = 6
    real_data = True
    time_set = timing_set(101, samplePerStep, 3, samplePerStep, 12)
    stock_list = [2, 321, 30, 241, 48, 180]


    """ Create solver instance """
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
    time_param = [101, samplePerStep,  3, samplePerStep, 12]
    log_path = get_log_path(stock_list, time_param, samplePerStep, penalty_function, time_set)
    if not Path(log_path).exists():
        os.makedirs(log_path + "/")

    save_matrix_plot_exact_number(solver.thetas, time_set, company_list_list,
                                  log_path + '/P_theta_exact_number_{}_{}.png'.format(solver.lambd, solver.beta))
    save_matrix_plot(solver.thetas, time_set, company_list_list,
                     log_path + '/P_theta_{}_{}.png'.format(solver.lambd, solver.beta))
    save_line_plot(solver.deviations, time_set, solver.lambd, solver.beta, solver.time_span, log_path + '/line.png', samplePerStep, penalty_function)
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
    print("\nAlgorithm run time: %s seconds" % (solver.time_span))
    print("Execution time: %s seconds" % (time.time() - start_time))

