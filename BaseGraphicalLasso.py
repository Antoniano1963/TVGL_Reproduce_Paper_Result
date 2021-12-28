import copy
import math

import numpy as np
import time
from DataHandler import DataHandler
from Params import ADMMParam


class BaseGraphicalLasso(object):

    # The parent class for Graphical Lasso
    # problems. Most of the methods and
    # attributes are defined and initialized here.

    np.set_printoptions(precision=3)

    """ Initialize attributes, read data """
    def __init__(self, filename, blocks, processes, samplePerStep, dimension, time_set, stock_list,
                 read_data_function, penalty_function="group_lasso",datecolumn=True, newADMMParam=ADMMParam):
        self.datecolumn = datecolumn
        self.processes = processes
        self.blocks = blocks
        self.samplePerStep = samplePerStep
        self.penalty_function = penalty_function
        self.dimension = dimension
        self.stock_list = stock_list
        self.emp_cov_mat = [0] * self.blocks
        self.real_thetas = [0] * self.blocks
        self.time_set = time_set
        if self.datecolumn:
            self.blockdates = [0] * self.blocks
        self.obs = self.dimension
        self.read_data_function = read_data_function
        self.getStocks()
        self.rho = self.get_rho()
        self.max_step = 0.1
        self.lambd = newADMMParam.LAMBDA
        self.beta = newADMMParam.BETA
        self.thetas = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.z0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.z1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.z2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.u2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_thetas = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_z0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_z1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_z2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_u0s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_u1s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.pre_u2s = [np.zeros((self.dimension, self.dimension))] * self.blocks
        self.eta = float(self.obs)/float(3*self.rho)
        self.e = newADMMParam.E
        self.roundup = newADMMParam.ROUNDUP # 结果四舍五入保留的位数
        self.e_rel = newADMMParam.E_REL # ADMM收敛条件的两个参数
        self.e_abs = newADMMParam.E_ABS
        self.rho = newADMMParam.RHO
        self.time_span = None


    """调用传入的读入数据函数以获取经验协方差"""

    def read_data(self, *args, **kwargs):
        empCov_set = self.read_data_function(*args, *kwargs)
        for i in range(len(empCov_set)):
            self.emp_cov_mat[i] = empCov_set[i]

    """读取股票数据"""
    def getStocks(self):
        timesteps = len(self.time_set)
        sample_data_set = []
        empCov_set = []
        stock_data = np.genfromtxt('Datasets/finance.csv', delimiter=',')
        size = len(self.stock_list)
        for i in range(timesteps):
            time_interval = self.time_set[i]
            sample_data = stock_data[time_interval[0]:time_interval[1] + 1, self.stock_list].T
            sample_data_set.append(sample_data)
            empCov_set.append(self.genEmpCov(sample_data))
        for i in range(len(empCov_set)):
            self.emp_cov_mat[i] = empCov_set[i]
        return size, timesteps, sample_data_set, empCov_set

    def genEmpCov(self, samples, useKnownMean=False, m=0):
        size, samplesPerStep = samples.shape
        if useKnownMean == False:
            m = np.mean(samples, axis=1)
        empCov = 0
        for i in range(samplesPerStep):
            sample = samples[:, i]
            empCov = empCov + np.outer(sample - m, sample - m)
            # empCov = empCov + np.outer(sample, sample)
        empCov = empCov / samplesPerStep
        return empCov

    """ Computes real inverse covariance matrices with DataHandler,
        if provided in the second line of the data file """
    def generate_real_thetas(self, line, splitter):
        dh = DataHandler()
        infos = line.split(splitter)
        for network_info in infos:
            filename = network_info.split(":")[0].strip("#").strip()
            datacount = network_info.split(":")[1].strip()
            sub_blocks = int(datacount)/self.obs
            for i in range(sub_blocks):
                dh.read_network(filename, inversion=False)
        self.real_thetas = dh.inverse_sigmas
        dh = None

    #这个不是到是干啥的，但是似乎很烂
    """ Assigns rho based on number of observations in a block """
    def get_rho(self):
        return float(self.obs + 0.1) / float(3)

    """ The core of the ADMM algorithm. To be called separately.
        Contains calls to the three update methods, which are to be
        defined in the child classes. """
    def run_algorithm(self, max_iter=10000):
        self.init_algorithm()
        self.iteration = 0
        stopping_criteria = False
        thetas_pre = []
        start_time = time.time()
        while self.iteration < max_iter and stopping_criteria == False:
            if self.iteration % 500 == 0 or self.iteration == 1:
                print("\n*** Iteration %s ***" % self.iteration)
                print("Time passed: {0:.3g}s".format(time.time() - start_time))
                print("Rho: %s" % self.rho)
                print("Eta: %s" % self.eta)
                print("Step: {0:.3f}".format(1/(2*self.eta)))
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.theta_update()
            if self.iteration % 500 == 0 or self.iteration == 1:
                print("Theta update: {0:.3g}s".format(time.time() - s_time))
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.z_update()
            if self.iteration % 500 == 0 or self.iteration == 1:
                print("Z-update: {0:.3g}s".format(time.time() - s_time))
            if self.iteration % 500 == 0 or self.iteration == 1:
                s_time = time.time()
            self.u_update()
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
                    print(self.thetas[0])
            thetas_pre = list(self.thetas)
            self.iteration += 1
            self.pre_z0s = copy.deepcopy(self.z0s)
            self.pre_z1s = copy.deepcopy(self.z1s)
            self.pre_z2s = copy.deepcopy(self.z2s)
            self.pre_u0s = copy.deepcopy(self.u0s)
            self.pre_u1s = copy.deepcopy(self.u1s)
            self.pre_u2s = copy.deepcopy(self.u2s)

        self.run_time = "{0:.3g}".format(time.time() - start_time)
        self.time_span = time.time() - start_time
        self.final_tuning(stopping_criteria, max_iter)

    def theta_update(self):
        pass

    def z_update(self):
        pass

    def u_update(self):
        pass

    def terminate_processes(self):
        pass

    def init_algorithm(self):
        pass

    """将二维对称矩阵转化成对应的一维向量"""
    def array_to_1D(self, arr):
        shape = arr.shape
        change_list = []
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                change_list.append(arr[i][j])
        return np.array(change_list)

    """使用的ADMM标准停止准则，模仿样例代码，将数据转成一维进行操作，计算维数也跟一维有关"""
    def check_Convergence(self):
        norm = np.linalg.norm
        Ax = self.thetas + self.thetas[:-1] + self.thetas[1:]
        Ax_1D_lists = []
        for arr in Ax:
            Ax_1D_lists.append(self.array_to_1D(arr))
        Ax_1D = np.copy(Ax_1D_lists[0])
        for i in range(1,len(Ax_1D_lists)):
            Ax_1D = np.hstack((Ax_1D, Ax_1D_lists[i]))

        z = self.z0s + self.z1s[:-1] + self.z2s[1:]
        z_1D_lists = []
        for arr in z:
            z_1D_lists.append(self.array_to_1D(arr))
        z_1D = np.copy(z_1D_lists[0])
        for i in range(1, len(z_1D_lists)):
            z_1D = np.hstack((z_1D, z_1D_lists[i]))

        pre_z = self.pre_z0s + self.pre_z1s[:-1] + self.pre_z2s[1:]
        pre_z_1D_lists = []
        for arr in pre_z:
            pre_z_1D_lists.append(self.array_to_1D(arr))
        pre_z_1D = np.copy(pre_z_1D_lists[0])
        for i in range(1, len(pre_z_1D_lists)):
            pre_z_1D = np.hstack((pre_z_1D, pre_z_1D_lists[i]))

        u = self.u0s + self.u1s + self.u2s
        u_1D_lists = []
        for arr in u:
            u_1D_lists.append(self.array_to_1D(arr))
        u_1D = np.copy(u_1D_lists[0])
        for i in range(1, len(u_1D_lists)):
            u_1D = np.hstack((u_1D, u_1D_lists[i]))

        r = Ax_1D - z_1D
        s = self.rho * (z_1D - pre_z_1D)

        single_p = (self.dimension + 1) * self.dimension / 2
        p = single_p * self.blocks
        n = (3 * self.blocks - 2) * single_p

        e_pri = math.sqrt(p) * self.e_abs + self.e_rel * max(norm(Ax_1D), norm(z_1D)) + 0.0001
        e_dual = math.sqrt(n) * self.e_abs + self.e_rel * norm(self.rho * u_1D) + 0.0001

        res_pri = norm(r, ord=2)
        res_dual = norm(s, ord=2)

        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)


    """ Performs final tuning for the converged thetas,
        closes possible multiprocesses. """
    def final_tuning(self, stopping_criteria, max_iter):
        self.temporal_deviations()
        self.thetas = [np.abs(np.round(theta, self.roundup)) for theta in self.thetas]
        self.terminate_processes()
        if stopping_criteria:
            print("\nIterations to complete: %s" % self.iteration)
        else:
            print("\nMax iterations (%s) reached" % max_iter)


    """ Converts values in the thetas into boolean values,
        informing only the existence of an edge without weight. """
    def only_true_false_edges(self):
        for k in range(self.blocks):
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if self.thetas[k][i, j] != 0:
                        self.thetas[k][i, j] = 1
                        self.thetas[k][j, i] = 1
                    else:
                        self.thetas[k][i, j] = 0
                        self.thetas[k][j, i] = 0


    """ Computes the Temporal Deviations between neighboring
        thetas, both absolute and normalized values. 
        TD Value"""
    def temporal_deviations(self):
        self.deviations = np.zeros(self.blocks - 1)
        for i in range(0, self.blocks - 1):
            dif = self.thetas[i+1] - self.thetas[i]
            # np.fill_diagonal(dif, 0)
            self.deviations[i] = np.linalg.norm(dif, 'fro')
        try:
            self.norm_deviations = self.deviations/max(self.deviations)
            self.dev_ratio = float(max(self.deviations))/float(
                np.mean(self.deviations))
        except ZeroDivisionError:
            self.norm_deviations = self.deviations
            self.dev_ratio = 0

    """ Computes the measures of correct edges in thetas,
        if true inverse covariance matrices are provided. """
    def correct_edges(self):
        self.real_edges = 0
        self.real_edgeless = 0
        self.correct_positives = 0
        self.all_positives = 0
        for real_network, network in zip(self.real_thetas, self.thetas):
            for i in range(self.dimension - 1):
                for j in range(i + 1, self.dimension):
                    if real_network[i, j] != 0:
                        self.real_edges += 1
                        if network[i, j] != 0:
                            self.correct_positives += 1
                            self.all_positives += 1
                    elif real_network[i, j] == 0:
                        self.real_edgeless += 1
                        if network[i, j] != 0:
                            self.all_positives += 1
        self.precision = float(self.correct_positives)/float(
            self.all_positives)
        self.recall = float(self.correct_positives)/float(
            self.real_edges)
        self.f1score = 2*(self.precision*self.recall)/float(
            self.precision + self.recall)
