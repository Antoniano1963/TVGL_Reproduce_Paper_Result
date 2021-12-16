import numpy as np

COR = ['A', 'AA', 'AAPL', 'ABC', 'ABT', 'ACE', 'ACS', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES',
       'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM', 'AKS', 'ALL', 'ALTR', 'AMAT', 'AMD', 'AMGN', 'AMP', 'AMT',
       'AMZN', 'AN', 'ANF', 'AOC', 'AON', 'APA', 'APC', 'APD', 'APH', 'APOL', 'ARG', 'ATI', 'AVB', 'AVP', 'AVY',
       'AXP', 'AYE', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'BBT', 'BBY', 'BCR', 'BDK', 'BDX', 'BEN', 'BF.B', 'BHI',
       'BIG', 'BIIB', 'BJS', 'BK', 'BLL', 'BMC', 'BMS', 'BMY', 'BNI', 'BRCM', 'BRK.B', 'BSX', 'BTU', 'BXP', 'C',
       'CA', 'CAG', 'CAH', 'CAM', 'CAT', 'CB', 'CBE', 'CBG', 'CBS', 'CCE', 'CCL', 'CEG', 'CELG', 'CEPH', 'CERN',
       'CF', 'CFN', 'CHK', 'CHRW', 'CI', 'CIEN', 'CINF', 'CL', 'CLF', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMI', 'CMS',
       'CNP', 'CNX', 'COF', 'COG', 'COH', 'COL', 'COP', 'COST', 'CPB', 'CPWR', 'CRM', 'CSC', 'CSCO', 'CSX', 'CTAS',
       'CTL', 'CTSH', 'CTXS', 'CVG', 'CVH', 'CVS', 'CVX', 'D', 'DD', 'DE', 'DELL', 'DF', 'DFS', 'DGX', 'DHI', 'DHR',
       'DIS', 'DISCA', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DPS', 'DRI', 'DTE', 'DTV', 'DUK', 'DV', 'DVA', 'DVN',
       'DYN', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EK', 'EL', 'EMC', 'EMN', 'EMR', 'EOG', 'EP', 'EQR', 'EQT', 'ERTS',
       'ESRX', 'ESV', 'ETFC', 'ETFCD', 'ETN', 'ETR', 'EXC', 'EXPD', 'EXPE', 'F', 'FAST', 'FCX', 'FDO', 'FDX', 'FE',
       'FHN', 'FII', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FO', 'FPL', 'FRX', 'FSLR', 'FTI', 'FTR',
       'GAS', 'GCI', 'GD', 'GE', 'GENZ', 'GILD', 'GIS', 'GLW', 'GME', 'GNW', 'GOOG', 'GPC', 'GPS', 'GR', 'GS', 'GT',
       'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HCBK', 'HCN', 'HCP', 'HD', 'HES', 'HIG', 'HNZ', 'HOG', 'HON', 'HOT',
       'HP', 'HPQ', 'HRB', 'HRL', 'HRS', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 'INTC', 'INTU',
       'IP', 'IPG', 'IRM', 'ISRG', 'ITT', 'ITW', 'IVZ', 'JAVA', 'JBL', 'JCI', 'JCP', 'JDSU', 'JEC', 'JNJ', 'JNPR',
       'JNS', 'JPM', 'JWN', 'K', 'KBH', 'KEY', 'KFT', 'KG', 'KIM', 'KLAC', 'KMB', 'KMX', 'KO', 'KR', 'KSS', 'L',
       'LEG', 'LEN', 'LH', 'LIFE', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC', 'LO', 'LOW', 'LSI', 'LTD', 'LUK',
       'LUV', 'LXK', 'M', 'MA', 'MAR', 'MAS', 'MAT', 'MBI', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDP', 'MDT', 'MEE', 'MET',
       'MFE', 'MHP', 'MHS', 'MI', 'MIL', 'MJN', 'MKC', 'MMC', 'MMM', 'MO', 'MOLX', 'MON', 'MOT', 'MRK', 'MRO', 'MS',
       'MSFT', 'MTB', 'MTW', 'MU', 'MUR', 'MWV', 'MWW', 'MYL', 'NBL', 'NBR', 'NDAQ', 'NEE', 'NEM', 'NI', 'NKE',
       'NOC', 'NOV', 'NOVL', 'NRG', 'NSC', 'NSM', 'NTAP', 'NTRS', 'NU', 'NUE', 'NVDA', 'NVLS', 'NWL', 'NWSA', 'NYT',
       'NYX', 'ODP', 'OI', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBG', 'PBI', 'PCAR', 'PCG', 'PCL',
       'PCLN', 'PCP', 'PCS', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGN', 'PGR', 'PH', 'PHM', 'PKI', 'PLD',
       'PLL', 'PM', 'PNC', 'PNW', 'POM', 'PPG', 'PPL', 'PRU', 'PSA', 'PTV', 'PWR', 'PX', 'PXD', 'Q', 'QCOM', 'QEP',
       'QLGC', 'R', 'RAI', 'RDC', 'RF', 'RHI', 'RHT', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RRD', 'RSG', 'RSH', 'RTN',
       'RX', 'S', 'SAI', 'SBUX', 'SCG', 'SCHW', 'SE', 'SEE', 'SGP', 'SHLD', 'SHW', 'SIAL', 'SII', 'SJM', 'SLB',
       'SLE', 'SLM', 'SNA', 'SNDK', 'SNI', 'SO', 'SPG', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STR', 'STT', 'STZ',
       'SUN', 'SVU', 'SWK', 'SWN', 'SWY', 'SYK', 'SYMC', 'SYY', 'T', 'TAP', 'TDC', 'TE', 'TEG', 'TER', 'TGT', 'THC',
       'TIE', 'TIF', 'TJX', 'TLAB', 'TMK', 'TMO', 'TROW', 'TRV', 'TSN', 'TSO', 'TSS', 'TWC', 'TWX', 'TXN', 'TXT',
       'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'USB', 'UTX', 'V', 'VAR', 'VFC', 'VIA', 'VIA.B', 'VLO', 'VMC', 'VNO',
       'VRSN', 'VTR', 'VZ', 'WAG', 'WAT', 'WDC', 'WEC', 'WFC', 'WFMI', 'WFR', 'WHR', 'WIN', 'WLP', 'WM', 'WMB',
       'WMT', 'WPI', 'WPO', 'WU', 'WY', 'WYE', 'WYN', 'WYNN', 'X', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XTO',
       'YHOO', 'YUM', 'ZION', 'ZMH']


def timing_set(center, samplesPerStep_left, count_left, samplesPerStep_right, count_right):
    time_set = []
    count_left = min(count_left, center / samplesPerStep_left)
    print('left timesteps: = ', count_left)
    start = max(center - samplesPerStep_left * (count_left), 0)
    for i in range(count_left):
        time_interval = [start, start + samplesPerStep_left - 1]
        time_set.append(time_interval)
        start = start + samplesPerStep_left
    count_right = min(count_right, 245 / samplesPerStep_left)
    print('right timesteps: = ', count_right)
    for i in range(count_right):
        time_interval = [start, start + samplesPerStep_right - 1]
        time_set.append(time_interval)
        start = start + samplesPerStep_right
    return time_set



def save_matrix_plot(theta_est_list, time_set, company_list_list, path):
    TIME_MAP = get_time_map()
    import matplotlib.pylab as pl
    row_num = np.ceil(len(time_set) / 5.0)
    fig = pl.figure(figsize=(20, 4.5 * row_num))
    for i in range(len(time_set)):
        theta = theta_est_list[i]
        theta[theta != 0] = 1
        time = TIME_MAP[time_set[i][0]] + '~' + TIME_MAP[time_set[i][-1]]
        fig.add_subplot(row_num, 5, i + 1)
        pl.imshow(theta, cmap='gray_r', interpolation='nearest')
        pl.title(time)
        ax = pl.gca()
        ticks = []
        for j in range(len(company_list_list)):
            ticks.append(j)
        # ticks = [i for i in range(len(list(company_list)))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(company_list_list)
        ax.set_yticklabels(company_list_list)
    pl.savefig(path, dpi=300, bbox_inches='tight')


def save_matrix_plot_exact_number(theta_est_list, time_set, company_list_list, path):
    TIME_MAP = get_time_map()
    import matplotlib.pylab as pl
    row_num = np.ceil(len(time_set) / 5.0)
    fig = pl.figure(figsize=(20, 4.5 * row_num))
    for i in range(len(time_set)):
        # theta = theta_est_list[i]
        # row, col = theta.shape
        # for i in range(row):
        #     for j in range(col):
        #         theta[i, j] = abs(theta[i, j])
        # theta[theta != 0] = abs(1)
        abs_theta = np.abs(theta_est_list[i])
        row, col = abs_theta.shape
        # abs_theta[abs_theta == 0] = 1
        for j in range(row):
            abs_theta[j, j] = -1
        max_num = np.max(abs_theta)
        for j in range(row):
            abs_theta[j, j] = max_num
        abs_theta[abs_theta == 0] = -0.1
        abs_theta = abs_theta * 1000
        time = TIME_MAP[time_set[i][0]] + '~' + TIME_MAP[time_set[i][-1]]
        fig.add_subplot(int(row_num), 5, i + 1)
        pl.imshow(abs_theta, cmap='PuBu', interpolation='nearest')
        pl.title(time)
        ax = pl.gca()
        ticks = []
        for j in range(len(company_list_list)):
            ticks.append(j)
        # ticks = [i for i in range(len(list(company_list)))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(company_list_list)
        ax.set_yticklabels(company_list_list)
    pl.savefig(path, dpi=300, bbox_inches='tight')


def get_time_map() -> object:
    m = []
    with open('time.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            m.append(line[:4] + '-' + line[4:6] + '-' + line[6:])
    return m

def get_company_list(stock_list):
    return map(lambda x : COR[x], stock_list)



def genEmpCov(samples, useKnownMean=False, m=0):
    size, samplesPerStep = samples.shape
    if useKnownMean == False:
        m = np.mean(samples, axis=1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:, i]
        empCov = empCov + np.outer(sample - m, sample - m)
    empCov = empCov / samplesPerStep
    return empCov