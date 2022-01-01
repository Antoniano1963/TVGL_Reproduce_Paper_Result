import time

from StockExperiment import StockExperiment
from args import get_parser
from utils import *
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
    time_set = timing_set(101, samplePerStep, left_step, samplePerStep, right_step)
    stock_list = [2, 321, 30, 241, 48, 180]
    newADMMPara = ADMMParam
    newADMMPara.LAMBDA = args.alpha
    newADMMPara.BETA = args.beta
    print("\nReading file: %s\n" % filename)
    stock_experiment = StockExperiment(stock_list=stock_list, time_set=time_set, penalty_function=penalty_function,
                                       blocks=blocks, sample_per_step=samplePerStep)
    stock_experiment.run()



