# System-Level Simulator functions


import utils as ut
import numpy as np
import numpy.linalg 

import scipy.io
import scipy.interpolate


import application_traffic as at


def id_tti(tti, n_slots_per_frame, UL_DL_split):
    """
    Based on the TDD split, figure if a tti is meant for UL or DL,
    and return a string accordingly ('UL' or 'DL').
    'F' (Flexible) options are not enabled currently.
    """
    
    return 'DL'
    
    
    if (tti % n_slots_per_frame + 1) > round(n_slots_per_frame * UL_DL_split):
        return 'UL'
    else:
        return 'DL'



""" Functions for the Scheduling process part 1"""


def compute_avg_bitrate(previous_avg, curr_bitrate, tc=100):
    """ Returns the average throughput experienced in the last tc time
    intervals
    """
    alphaPF = 1 / tc
    return previous_avg * (1 - alphaPF) + curr_bitrate * alphaPF
    

def pf_scheduler(avg_thrput, curr_expected_bitrate):
    """
    Returns the well known proportional fairness ratio, the ratio between the
    passible to being achieved instantaneous bitrate and the average 
    experienced bitrate.
    """
    
    # Only to cope with the initialisation possibility and avoid crashes
    if avg_thrput == 0:
        return 1e20
    
    return curr_expected_bitrate / avg_thrput


def MLWDF_scheduler(avg_thrput, curr_expected_bitrate, 
                    curr_delay, delay_threshold, delta=0.1):
    """
    Parameters
    ----------
    avg_thrput : weighted over many ttis
    curr_expected_bitrate : bitrate estimated as achievable for the curr tti
    lat : current delay of the packet at the head of the queue
    delta : upper limit of packet loss rate 
            (0: NO PACKET CAN BE LOST!, 1: who cares)
            Note: it was made to differentiate between several QoS.
            So, if all users have the same priority, there's no weight from
            it, and can be considered a constant.
    Returns
    -------
    Returns the priority for a given user computed with the Maximum-Largest
    Weighted Delay First Scheduler.
    """
    
    # Is it natural log or log10?
    a = -np.log(delta) / delay_threshold
    
    return a * curr_delay * pf_scheduler(avg_thrput, curr_expected_bitrate)


def exp_pf_scheduler(avg_thrput, curr_expected_bitrate,
                     curr_delay, c, delay_threshold, all_delays,
                     kappa=100, epsilon=0.1):
    """
    Parameters
    ----------
    avg_thrput : average experienced throughput
    curr_expected_bitrate : estimated achievable throughput
    curr_delay : current delay of the head of the queue
    c : constant, for prioritising traffic flows.
    delay_threshold : maximum delay 
    all_delays : delay of each queue's head
    n_rt : Number of real-time traffic flows (equivalent to the number of 
    buffers to be served)
    Returns
    -------
    Double: A priority measure.
    """
    
    print('This NEEDS testing!!!!')
    
    a = c / delay_threshold
    
    n_rt = len(all_delays)
    
    aW_avg = sum(a * all_delays) / n_rt
    
    return np.exp((a * curr_delay - aW_avg) / (1 + np.sqrt(aW_avg)) * 
                  pf_scheduler(avg_thrput, curr_expected_bitrate))



def scheduler(scheduler_choice, avg_throughput_ue, estimated_bitrate,
              buffer_head_of_queue_delay, delay_threshold, 
              scheduler_param_delta, scheduler_param_c, all_delays):
    
    if scheduler_choice == 'PF':
        priority = pf_scheduler(avg_throughput_ue, 
                                estimated_bitrate)
    
    elif scheduler_choice == 'M-LWDF':
        priority = MLWDF_scheduler(avg_throughput_ue, 
                                   estimated_bitrate, 
                                   buffer_head_of_queue_delay, 
                                   delay_threshold, 
                                   scheduler_param_delta)
   
    elif scheduler_choice == 'EXP/PF':
        priority = exp_pf_scheduler(avg_throughput_ue, 
                                    estimated_bitrate,
                                    buffer_head_of_queue_delay, 
                                    scheduler_param_c,
                                    delay_threshold, 
                                    all_delays)
    else:
        raise Exception("The only available schedulers are 'PF', 'M-LWDF'"
                        " and 'EXP/PF'.")

    return priority





"""
MCS table implemented in the functions below:
    Table 5.2.2.1-3 from 38.214
    For maximum BLER of 0.1 and up to 256 QAM.
    
    CQI index | modulation | code rate x 1024 | efficiency | Bit Rate [kbps]
    1              QPSK             78 	      0.1523 	     25,59375
    2              QPSK            193 	      0.3770 	     63,328125
    3              QPSK            449 	      0.8770         147,328125
    4             16QAM            378           1.4766 	    248,0625
    5             16QAM            490    	      1.9141 	    321,5625
    6	         16QAM 	           616 	          2.4063       404,25
    7	         64QAM             466           2.7305 	    458,71875
    8	         64QAM             567           3.3223 	    558,140625
    9	         64QAM             666           3.9023 	    655,59375
    10	         64QAM             772           4.5234 	    759,9375
    11	         64QAM             873           5.1152 	    859,359375
    12	        256QAM             711           5.5547 	    933,1875
    13	        256QAM             797           6.2266	   1046,0625
    14	        256QAM             885           6.9141	   1161,5625
    15	        256QAM             948           7.4063 	   1244,25
    
"""


def calc_CQI(sinr):
    """
    Uses BLER(SINR) curves fitted for several tables of MCSs.
    Here is implemented Table 5.2.2.1-3 from 38.214. 
    For maximum BLER of 0.1 and up to 256 QAM.
    
    
    CQI index | modulation | code rate x 1024 efficiency | Bit Rate [kbps]
    1	QPSK 	78 	0.1523 	25,59375
    2	QPSK 	193 	0.3770 	63,328125
    3	QPSK 	449 	0.8770 	147,328125
    4	16QAM 	378 	1.4766 	248,0625
    5	16QAM 	490 	1.9141 	321,5625
    6	16QAM 	616 	2.4063 	404,25
    7	64QAM 	466 	2.7305 	458,71875
    8	64QAM 	567 	3.3223 	558,140625
    9	64QAM 	666 	3.9023 	655,59375
    10	64QAM 	772 	4.5234 	759,9375
    11	64QAM 	873 	5.1152 	859,359375
    12	256QAM 	711 	5.5547 	933,1875
    13	256QAM 	797 	6.2266	1046,0625
    14	256QAM 	885 	6.9141	1161,5625
    15	256QAM 	948 	7.4063 	1244,25
    
    
    Parameters
    ----------
    sinr : estimated achievable SINR.
    
    Return
    ----------
    mcs_idx : the index of the MCS to be used.
    """

    # 
    for cqi_idx in range(15, 0, -1):
        bler = get_BLER_from_fitted_MCS_curves(cqi_idx, sinr)
        if bler < 0.1:
            return (cqi_idx, bler)

    # if not even the lowest modulation could cope with it, return 
    # the cqi for 'no_signal/out of range', at 100% BLER.
    return (0, 1)


def get_BLER_from_fitted_MCS_curves(cqi, sinr):
    """
    Parameters
    ----------
    cqi : index for the MCS to be used
    sinr : wanna guess this one?
    Returns
    -------
    A tuple with 
        - CQI
        - The Block Error Rate for a given SINR (expected or realised)
    """
    
    x = sinr
    ut.parse_input(cqi, [i for i in range(1, 15 + 1)])
    
    
    if cqi >= 15 and sinr < 23.7:
        bler =  1
    elif cqi >= 14 and sinr < 22:
        bler = 1
    elif cqi >= 13 and sinr < 20:
        bler = 1
    elif cqi >= 12 and sinr < 18.3:
        bler = 1
    elif cqi >= 11 and sinr < 16.6:
        bler = 1
    elif cqi >= 10 and sinr < 14.8:
        bler = 1
    elif cqi >= 9 and sinr < 12.5:
        bler = 1
    elif cqi >= 8 and sinr < 10.7:
        bler = 1
    elif cqi >= 7 and sinr < 9:
        bler = 1
    elif cqi >= 6 and sinr < 6.7:
        bler = 1
    elif cqi >= 5 and sinr < 4.8:
        bler = 1
    elif cqi >= 4 and sinr < 3.4:
        bler = 1
    elif cqi >= 3 and sinr < -1.1:
        bler = 1
    elif cqi >= 2 and sinr < -6:
        bler = 1
    elif cqi >= 1 and sinr < -9.6:
        bler = 1
    else:
        # Fitted curves
        bler = {
            1: (0.8942 * np.exp(-((x + 10.05) / 1.28)**2) +
                0.5795 * np.exp(-((x + 8.602) / 0.9784)**2)),
            2: 1 - (9.182 / (np.exp(-4.293 * x - 16.31) + 9.171)),
            3: 1 - (0.7106 / (np.exp(-6.388 * x) + 0.7106)),
            4: 1 - (1 / (np.exp(-6.138 * x + 28.19) + 0.9996)),
            5: 1 - (1 / (np.exp(-7.502 * x + 44.68) + 0.9985)),
            6: 1 - (1 / (np.exp(-8.279 * x + 64.07) + 0.9996)),
            7: 1 - (1 / (np.exp(-7.981 * x + 79.61) + 0.9998)),
            8: 1 - (1 / (np.exp(-8.217 * x + 96.46) + 0.9995)),
            9: 1 - (1 / (np.exp(-9.292 * x + 124.6) + 0.9989)),
            10: (0.6046 * np.exp(-((x - 15.1) / 0.3454)**2) + 
                 0.9940 * np.exp(-((x - 13.47) / 0.969)**2) + 
                 0.6544 * np.exp(-((x - 14.56) / 0.5685)**2)),
            11: (0.6768 * np.exp(-((x - 16.3) / 0.6109)**2) + 
                 0.9575 * np.exp(-((x - 15.24) / 0.895)**2) + 
                 0.5245 * np.exp(-((x - 17.08) / 0.2716)**2) + 
                 0.4280 * np.exp(-((x - 16.73) / 0.3344)**2)),
            12: 1 - (1 / (np.exp(-10.22 * x + 196.7) + 0.9999)),
            13: 1 - (1 / (np.exp(-9.939 * x + 208.5) + 0.9988)),
            14: 1 - (1 / (np.exp(-10.23 * x + 234.7) + 0.9995)),
            15: 1 - (1 / (np.exp(-9.504 * x + 235.7) + 0.9997))}[cqi]
    
    # Limit non-sensical values
    if bler < 0:
        bler = 0
    
    if bler > 1:
        bler = 1
        
    return bler


def bits_per_PRB(cqi):
    """
    Returns the an (inflated) estimate of number of bits that can be sent 
    given a MCS and #PRBs.
    
    This is 100% copy from the bits per PRB column.
    From the implemented table, efficiency x 168 resource elements, each 
    containing a symbol. 
    
    V2 may implement something like this:
    https://www.sharetechnote.com/html/5G/5G_MaxThroughputEstimation.html
    """
    
    if cqi < 1:
        return 0
    if cqi > 15:
        cqi = 15
    
    bits_per_prb = {1: 25.5864,
                    2: 63.3360,
                    3: 147.3360,
                    4: 248.0688,
                    5: 321.5688,
                    6: 404.2584,
                    7: 458.7240,
                    8: 558.1464,
                    9: 655.5864,
                    10: 759.9312,
                    11: 859.3536,
                    12: 933.1896,
                    13: 1046.0688,
                    14: 1161.5688,
                    15: 1244.2584}[cqi]

    return bits_per_prb


def estimate_bits_to_be_sent(cqi, n_prbs, freq_compression_ratio=1):
    """
    The number of bits sent per PRB is constant for all numerologies, for
    a given MCS (we use 'cqi' here). This happens because as a PRB gets larger
    in frequency due to increase in numerology, it gets proportionally 
    shorter in time, and the bits we can send with more Hz but in a smaller
    interval stay the same.
    
    This mean that from the number of PRBs we can tell right away how many 
    bits can be sent.
    
    (THE IMPORTANT PART NOW)
    
    Furthermore, this function takes into account the possibility of less
    frequency granularity. Basically, we can have PRBs that are larger 
    in frequency than expected, without scalling their duration. For such 
    inflated PRBs, this function scales the bits to be sent accordingly. 
    
    The frequency compression ratio corresponds to the number of PRBs
    that each sample in frequency represents.   
        
    """
    
    bits_per_prb = bits_per_PRB(cqi) * freq_compression_ratio
    
    return int(np.floor(bits_per_prb * n_prbs))


def bits_per_symb_from_cqi(cqi):
    """
    Derived by visual inspection from the table above.
    """
    if cqi < 1:
        bits_per_symbol = 0
    elif 1 <= cqi <= 3:
        bits_per_symbol = 2
    elif 4 <= cqi <= 6:
        bits_per_symbol = 4
    elif 7 <= cqi <= 11:
        bits_per_symbol = 6
    elif 12 <= cqi <= 15:
        bits_per_symbol = 8
    else:
        raise Exception('No other CQIs are supported in a transmission. '
                        'Only from 1 to 15')

    return bits_per_symbol



"""
Transport Blocks' Logic
"""


class Transport_Block():
    def __init__(self, size, start_idx):
        # Number of bits of the TB
        self.size = size
        
        # This Transport Block has assigned some bits in a certain part of
        # the buffer. This will be used to remove these bits from the buffer 
        # afterwards if the block is transported successfully
        self.start_idx = start_idx
       
        
    def print_tb(self):
        print(f'TB start: {self.start_idx}; Size: {self.size} bits.')
        


def get_TB_size(bits_to_be_sent, tbs_divisor, n_layers=0, v='v1'):
    """
    Objective: returns the transport block size.
    
    Currently V1:
    Returns the a TBS which equals all bits estimated to get across given a
    certain MCS and #PBRs.
    Therefore, only one transport block is used. (more eggs in the same basket)
    
    V2 Implements: 
    https://5g-tools.com/5g-nr-tbs-transport-block-size-calculator/
    Described more carefully here:
    https://www.sharetechnote.com/html/5G/5G_MCS_TBS_CodeRate.html#PDSCH_TBS
    
    
    See also:
        https://www.resurchify.com/5G-tutorial/5G-NR-Throughput-Calculator.php
    
    Note before implementing v2: for the uplink is too complicated to do 
    the same. So, either do the same as for the DL, or stick with v1.
    """
    if v == 'v1':
        # here we estimate the bits a priori
        pass
    else:
        # here we apply a complicated process to estimate the bits to be sent
        # (decide about this!!!!!!!!!)
        
        # Number of resource elements
        # n_re_dmrs = 0
        
        # Number of useful resource elements
        # n_re = 
        
        
        
        bits_to_be_sent = 0  # V2 still to implement.

    
    return np.ceil(bits_to_be_sent / tbs_divisor)


"""
Convenient Formulas
"""


def calc_SINR(tx_pow, ch_pow_gain, interference, noise_power):
    """
    Compute the SINR.
    """
    sig_pow = tx_pow * ch_pow_gain
    
    sinr_linear = sig_pow / (noise_power + interference)
    
    return 10 * np.log10(sinr_linear)



def get_curr_time_div(tti, time_div_ttis):
    
    return int(np.floor(tti / time_div_ttis))


"""
SINR precise formula
"""

def calc_rx_power_lin(tx_pow, tx_precoder, rx_precoder, ch_coeffs):
    """
    Receives channel coefficients, and precoders for the scheduled users
    """
    
    ch_power_gain_before_combining = np.dot(ch_coeffs, tx_precoder)
    ch_power_gain_after_combining = np.dot(ch_power_gain_before_combining, 
                                           rx_precoder)
    
    return tx_pow * (abs(ch_power_gain_after_combining) ** 2)


""" 
Functions to index the table of Information bits
Assumes the table has been written with an interval of 0.5 between SINRs.
"""


def load_info_bits_table(table_path):
    """
    Loads to memory the complete table required for MIESM
    """
    return np.genfromtxt(table_path, delimiter=',')


def get_information_bits(sinr, k, table, low_extreme=-10, step=0.5):
    """ 
    Rounds the sinr and sees which index is that SINR belongs to.
    """
    
    sinr_rounded = ut.round_to_value(sinr, step)
    
    idx_of_sinr = int((sinr_rounded - low_extreme) / step)
    
    k_idx = int(k / 2)
    
    return table[idx_of_sinr, k_idx]
                   
                   
def get_sinr_from_info_bits_closest(info_bits, k, table):
    """
    Returns the SINR that is closer to a certain number of information bits
    of the table. Search along the column of that modulation is required.
    """
    k_idx = int(k / 2)
    
    idx_of_closest = (np.abs(table[:, k_idx] - info_bits)).argmin()
    
    return table[idx_of_closest, 0]
       

def get_sinr_from_info_bits_interpolated(info_bits, k, table):
    """
    Returns the SINR that is closer to a certain number of information bits
    of the table. Search along the column of that modulation is required.
    """
    k_idx = int(k / 2)
    
    for i in range(table.shape[0]):
        if table[i, k_idx] - info_bits > 0:
            break
    
    idx_of_previous = i - 1
    idx_of_next = i
    
    info_bits_diff = (table[idx_of_next, k_idx] - 
                      table[idx_of_previous, k_idx])
    if info_bits_diff == 0:
        return get_sinr_from_info_bits_closest(info_bits, k, table)
    
    # Manual interpolation
    sinr_diff = (table[idx_of_next, 0] - table[idx_of_previous, 0])
    slope =  sinr_diff / info_bits_diff
                 
    diff = info_bits - table[idx_of_previous, k_idx]
    
    sinr = table[idx_of_previous, 0] + slope * diff
    
    return sinr


def avg_SINRs_MIESM(sinrs, table, k):
    """
    Compute the SINR that each PRB would have to need to have in order to 
    transmit the average number of information bits each PRB can transmit.
    
    k is the number of bits enconded in each symbol. E.g. In QPSK and 4-QAM, 
    there are 4 symbols, thus each enconding the value of 2 bits.
    With an avergae number of information bits <=k, we search the table
    
    Note: if the table is changed from [-10, 32] dB with steps of 0.5, the
    indexing/search functions should have their arguments changed here.
    """
    # The low extreme of the table
    low_ext = -10
    # The high extreme of the table
    high_ext = 32
    # Step between SINRs
    step = 0.1
    
    n_subcarriers = len(sinrs)
    
    total_i_bits = 0
    for i in range(n_subcarriers):
        if sinrs[i] > high_ext:
            total_i_bits += k
        elif sinrs[i] < low_ext:
            total_i_bits += 0
        else:
            total_i_bits += get_information_bits(sinrs[i], k, table, 
                                                 low_ext, step)
    
    # per symbol
    avg_information_bits = total_i_bits / n_subcarriers
    
    eff_sinr = get_sinr_from_info_bits_interpolated(avg_information_bits, 
                                                    k, table)
    
    return eff_sinr

##########################################################################################

""" 
Functions around channel coefficients loading and handling/managing.
"""

def load_coeffs_part(fname):
    
    fname_real = fname + '_r.bin'
    fname_imag = fname + '_i.bin'
    
    # Read coeffs in binary
    with open(fname_real) as fp:
        real_part = np.fromfile(fp, dtype=np.single)
    
    with open(fname_imag) as fp:
        imag_part = np.fromfile(fp, dtype=np.single)
    
    # Merge them into a complex numbers and attribute the values to coeffs
    coeffs_part = np.complex64(real_part + 1j * imag_part)
    
    return coeffs_part
    

def get_coeff_part_idx(time_div_idx, freq_idx, bs_idx, ue_idx,  
                       n_freq, n_bs, n_ue):
    
    """
    Given the time division, it computes which parts (coefficients files) 
    should be loaded next in order to give continuity to the simulation.
    """

    if freq_idx >= n_freq or bs_idx >= n_bs or ue_idx >= n_ue:
        print('Wront index required!')
        print(f"There are {n_freq} freqs, {n_bs} BSs and {n_ue} UEs."
              f"The (zero-indexed!!!) index required was "
              f"({freq_idx, bs_idx, ue_idx})")
        raise Exception()
        
    part_idx = (time_div_idx * (n_ue * n_bs * n_freq) + 
                freq_idx * (n_ue * n_bs) + 
                bs_idx * (n_ue) +
                ue_idx * 1) + 1
    
    return int(part_idx)



""" Coefficient Loading functions"""

def interp1d(xx, yy, kind='linear', ax=-1):
    
    if kind == 'linear':
        return scipy.interpolate.interp1d(xx, yy, 'linear', axis=ax)
    elif kind == 'log':
        logx = np.log10(xx)
        logy = np.log10(yy)
        lin_interp = scipy.interpolate.interp1d(logx, logy, 'linear', 
                                                axis=ax)
        log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
        return log_interp
    else:
        raise Exception("Only 'linear' and 'log' interpolation types "
                        "are available.")
        
def get_time_interpolation_ttis(target_interpolation_ttis,
                                time_compression_ratio):

    # Compressed TTIs
    ttis_c = np.arange(0, target_interpolation_ttis+1, time_compression_ratio)
    
    # Non-compressed TTIs
    ttis = np.arange(0, target_interpolation_ttis)
    
    return ttis, ttis_c


def time_interpolation(ttis, ttis_c, coeffs_c, mode='fast'):
    """
    Make an interpolation in the complex domain. For each coefficient, more
    many TTIs are generated. Namely, time_compression_ratio of them.
    We interpolate amplitude and phase linearly, through linear interpolations
    of the real and imaginary parts.
    """
    # Compressed real and imaginary parts
    real_c = np.real(coeffs_c)
    imag_c = np.imag(coeffs_c)
    
    # Interpolated functions
    real_i = interp1d(ttis_c, real_c, kind='linear')
    imag_i = interp1d(ttis_c, imag_c, kind='linear')
    
    
    if mode == 'fast':
        interpolated_coeffs = np.complex64(real_i(ttis) + 1j * imag_i(ttis))
    else:
        mag_c = np.abs(coeffs_c)
        mag_i = interp1d(ttis_c, mag_c, kind='linear')
        
        interpolated_coeffs = np.complex64( \
            mag_i(ttis) * np.exp(1j * np.angle(real_i(ttis) + 1j * imag_i(ttis))))
        
    
    # Interpolate and reassemble
    return interpolated_coeffs
        

def load_coeffs(tti, time_div_idx, n_time_divs, ttis_per_time_div,
                time_compression_ratio, f_idx, n_freq, n_bs_gen, n_ue_gen, 
                specific_bss, specific_ues, coeff_file_prefix, 
                coeff_file_suffix, n_ue_coeffs, n_bs_coeffs, ae_ue, ae_bs, 
                prbs, ttis_per_batch):
    """
    Loads coefficients from the respective time division:
        In order to have generation and simulation decoupled, we may
        generate with the number of time divisions we desire, and have a proper
        amount of coefficients loaded, adjusted to their size. E.g. in 26 GHz
        there are 16x more antennas than in 3.5 GHz, so it is less likely that
        all users can be simulated simulatenously.
        
    Samples are generated and we want to end up with TTIs. There can be a TTI
    per sample, when time_compression_ratio = 1, but that is the minimum. 
    """
    
    if time_div_idx >= n_time_divs:
        raise Exception('More time divisions than the ones the simulation '
                        'has, are not supported yet.')
        
    coeffs = {}

    if time_compression_ratio != 1:
        ttis, ttis_c = get_time_interpolation_ttis(ttis_per_batch,
                                                   time_compression_ratio)
        
    # The +1 is there for interpolation. It is actually the first 
    # sample of the next time division.
    if time_compression_ratio == 1:
        extra_sample = 0
    else:
        extra_sample = 1
        
    samples_per_time_div = int(ttis_per_time_div / 
                               time_compression_ratio) + extra_sample
    
    samples_per_batch = int(ttis_per_batch / 
                            time_compression_ratio) + extra_sample
    
    if samples_per_batch < samples_per_time_div:
        offset_in_ttis = int(tti - (time_div_idx) * ttis_per_time_div)
        
        # offset in samples
        offset = int(offset_in_ttis / time_compression_ratio)
        
        print('Loading: ')
                    
    for bs in specific_bss:
        for ue in specific_ues:
            # Given the indices, figure what part should be loaded.
            part_idx = get_coeff_part_idx(time_div_idx, 
                                          f_idx, bs, ue, 
                                          n_freq, n_bs_gen, n_ue_gen)
            
            name_to_load = (coeff_file_prefix + 
                            str(part_idx) + 
                            coeff_file_suffix)
            
            coeff_shape = (n_ue_coeffs[ue], n_bs_coeffs[bs], 
                           prbs, samples_per_time_div)
            
            print(f'\rLoading for UE {ue}, BS {bs}...', end='')
            
            coeffs_aux = \
                load_coeffs_part(name_to_load).reshape(coeff_shape, order='F')
            
            # Trim to fit the batch size
            if samples_per_batch < samples_per_time_div:
                coeffs_aux = coeffs_aux[:,:,:, 
                                        offset:offset + samples_per_batch]
            
            bs_idx = specific_bss.index(bs)
            ue_idx = specific_ues.index(ue)
            if time_compression_ratio == 1:
                # When there were as many samples generated as TTIs needed
                # no interpolation is needed
                coeffs[(bs_idx, ue_idx)] = coeffs_aux
            else:
                # interpolate!
                if time_compression_ratio > 10:
                    interp_mode = 'accurate'
                else:
                    interp_mode = 'fast'
                
                coeffs[(bs_idx, ue_idx)] = \
                    time_interpolation(ttis, ttis_c, coeffs_aux, interp_mode)
            
    # In addition, return the last tti to which these coefficients apply
    last_coeff_tti = tti + ttis_per_batch - 1
    
    return coeffs, last_coeff_tti


def update_channel_vars(tti, TTIs_per_batch, n_ue, coeffs, channel, 
                        channel_per_prb, save_prb_vars):
    
    """
    Aggregate channel responses across antenna elements per ue. Assumes BS 0.
    
    """
    ttis = np.arange(tti, tti + TTIs_per_batch)
    
    for ue in range(n_ue):
        
        # coeffs is a dictionary with the channel betwee BS-UE
        c = coeffs[(0, ue)] # c is [n_rx, n_tx, n_prb, n_tti]
        
        
        # a) take the power between any two elements
        # b) get the average per prb (is not )
        # c) 
        
        for t_idx in range(len(ttis)):
            # channel is [n_ue,tti]
            channel[ttis[t_idx]][ue] = \
                10 * np.log10(np.sum(np.mean(np.abs(c[:,:,:,t_idx]) ** 2, 2)))
                # channel_per_prb is [n_ue, tti, n_prb]
            # The second check is to prevent this to run before it is properly
            # implemented and tested. The save_prb_vars should be enough.
            # PS: actually, separate in channel and sig_pow vars to be specific
            if save_prb_vars and channel_per_prb != []:
                channel_per_prb[ttis[t_idx]][ue] = 10 * \
                    np.log10(np.sum(np.sum(np.abs(c[:,:,:,t_idx]) ** 2, 0), 0))



def copy_last_coeffs(coeffs, last_x):
    
    """
    Given a dictionary of coefficients, copy the last x coefficients to a new
    dictionary with the same keys, but much smaller.
    """
    
    # For initialisation purposes
    if coeffs == '':
        return ''
    
    new_coeffs = {}
    
    for key, coeff_vals in coeffs.items():
        new_coeffs[key] = coeff_vals[:,:,:,-last_x:]
    
    
    return new_coeffs



"""
Form channel matrix functions
"""

def channel_matrix(coeffs, ue, bs, prb, tti_relative, pol=-1):
    
    if pol == -1:
        c = coeffs[(bs, ue)][:,:,prb,tti_relative]
    elif pol in [0, 1]:
        c = coeffs[(bs, ue)][:,pol::2,prb,tti_relative]
    else:
        raise Exception('Only polarisations 0 and 1 are supported.')
        
    return c

"""
Precoder related functions.
"""


def load_precoders(precoders_paths, vectorize_GoB):
    """
    Creates a dictionary of base stations and angles, based on the precoder
    files. 
    
    The precoder is for an array of a certain size that spans a certain 
    angular domain with a given resolution. This information will be read from
    the precoder file.
    """
    precoders_dict = {}
    
    for bs in range(len(precoders_paths)):
        
        precoder_file = scipy.io.loadmat(precoders_paths[bs])
        
        precoders_dict[(bs, 'matrix')] = precoder_file['precoders_matrix']
        precoders_dict[(bs, 'directions')] = \
            precoder_file['precoders_directions']
        n_azi_beams = precoder_file['n_azi_beams'][0][0] # 11
        n_ele_beams = precoder_file['n_ele_beams'][0][0] # 11
        n_directions = precoders_dict[(bs, 'directions')].shape[1]
        
        # Store angle information along with the precoders
        # Size = [# of precoders with very similar azimuths, 
        #         # of precoders with very similar elevations] 
        # for a square GoBs, it is the square root of the total # of precoders
        precoders_dict[(bs, 'size')] = [n_azi_beams, n_ele_beams]
        precoders_dict[(bs, 'n_directions')] = n_directions
        
        
        
        # TODO: try vectorizing the GoB
        # if vectorize_GoB:
        #     # If the GoB is vectorized, create the full precoder matrix 
        #     # AE_BS x N_GoB, where N_GoB is the number of beams in the grid
        #     n_beams = np.prod(precoders_dict[(bs)])
        #     precoders_dict[(bs, 'full-matrix')] = np.zeros()
        #     for azi_idx in range(n_azi_vals):
        #         for el_idx in range(n_el_vals):
        #             beam_idx = el_idx + azi_idx * n_el_vals
        #             precoders_dict[(bs, 'full-matrix')][:, beam_idx] = \
        #                 precoders_dict[(bs, azi_idx, el_idx)]
        
    return precoders_dict
    

def print_precoders_dict(precoders_dict, bs_idx, print_directions=False, 
                        print_precoders=False):
    try:
        size = precoders_dict[(bs_idx, 'size')]
        n_directions = precoders_dict[(bs_idx, 'n_directions')]
        print(f'Codebook for BS {bs_idx} has size {size} '
              f'-> {n_directions} directions')
        if print_directions or print_precoders:
            for dir_idx in range(n_directions):               
                if print_directions:
                    ang = precoders_dict[(bs_idx, 'directions')][:,dir_idx]
                    print(f'Ang: [{ang[0]:2},{ang[1]:2}];')
                if print_precoders:    
                    p = precoders_dict[(bs_idx, 'matrix')][:,dir_idx]
                    print(p)
    except KeyError:
        print('KEY ERROR!!')


class Beam_pair():
    """
    This class holds a beam pair between one BS polarisation and all 
    UE antennas. For organisational purposes, we divide the UE's antennas in 
    two, having a set of antenas (and a beamformer) per polarisation.
    
    pol 0 means -45? antennas
    pol 1 means +45? antennas
    """
    
    
    def __init__(self):
        # Grid of Beams specific: direction at which the BS beam is pointing
        self.ang = [0, 0]
        self.beam_idx = [0, 0]

        # The correct polarisation combination must be saved, because this
        # determines the coefficients to be used when using that beam pair.
        # pol = -1 -> both polarisations jointly
        # pol = 0 -> pol 0 on bs, both polarisations on the ue
        #                         (pol_comb = 0 + pol_comb = 2)
        # pol = 1 -> pol 1 on bs, both polarisations on the ue
        #                         (pol_comb = 1 + pol_comb = 3)
        # For other combinations, change channel_matrix() to create the 
        # appropriate channel matrix. For now, all transmissions use all 
        # antennas.
        self.pol = -1  # Note: this is not used anymore, but
                       #       the channel_matrix() function still supports it.
        
        # Notice that polarisation and polarisation combination are different
        # things. When talking about combinations, we are referring to a 
        # combination of polarisations one at the UE, and one at the BS.
        # If we just mention ONE polarisation, it is ALWAYS on the BS side,
        # because, remember, the UE uses all antennas always
        
        
        # The normalised precoders/combiners:
        
        # Precoder used at BS side (for TX and RX)
        self.bs_weights = []
        
        # UE RX/TX precoders 
        # Derived with MRC/MRT from channel and BS precoder
        # one per polarisation
        self.ue_weights = []
        
        # The linear power channel gain 
        self.ch_power_gain = 0

        # When the precoder list was last updated (absolute tti)
        # (In this TTI, the most up-to-date CSI was used)
        self.last_updated = -1
    
    def print_pair(self):
        print(f'Ang: [{self.ang[0]:2},{self.ang[1]:2}]; '
              f'Gain: {self.ch_power_gain:.2e}')


def print_curr_beam_pairs(curr_beam_pairs, n_bs, n_ue, n_layers):
    for bs in range(n_bs):
        for ue in range(n_ue):
            # layers are updated at the same time, so we can use the time of 
            # update of just one of them.
            
            print(f'BS {bs}, UE {ue}, All layers [Last updated in '
                      f'TTI {curr_beam_pairs[(bs, ue, 0)].last_updated}]:')
            for l in range(n_layers):
                curr_beam_pairs[(bs, ue, l)].print_list()
            

def interleave(arrays, axis=0, out=None):
    """
    From user 'clwainwright' in https://stackoverflow.com/questions/5347065
    
    Interleaves any number of arrays along any axis.
    The arrays must have the exact same shape.
    """
    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis+1] + [len(arrays)] + shape[axis+1:])
    shape[axis] = -1
    return np.stack(arrays, axis=axis+1, out=out).reshape(shape)


def find_best_beam_pairs(codebook_subset, azi_len, el_len, q_idxs, 
                         codebook_subset_directions, ch_resp, bs, n_csi_beams,
                         save_power_per_CSI_beam, vectorize):
    """
    Given a precoder dictionary, and a channel response, and a bs index, 
    returns index pair for the best precoder for that channel (highest absolute
    value of internal product). 
    """
    
    if n_csi_beams > 1:
        raise Exception('NEEDS IMPLEMENTATION!')
        # The change is really simple:
            # create a beam list before the loop
            # every time the channel gain passes the minimum channel
            # gain in the list add the beam to the list, or replace the
            # last element of the list in case it already has n_best beams in
            # it. 
    
    
    # Rule of Thumb: always normalise precoders before using them.
    # Sometimes the precoder is a beam steering vector and is makes
    # no difference since the weights all have the same absolute 
    # value leading to |w|=1, but some other times it is not the 
    # case. Moreover, since normalization is a projection, the already 
    # normalized vectors won't suffer any change.
    
    
    # Compute best beam in a vectorized manner
    matrix_instead_of_loop = False
    
    curr_max_ch_gain = 0
    
    power_per_beam_list = []
    best_beam_relative_idxs = []    
    beam_pairs = []
    
    # Check whether beamforming is ON or OFF (if there's only 1 element, no BF)
    if ch_resp.shape == (1,1):
        best_idx = 0
        curr_max_ch_gain = np.abs(ch_resp)[0][0]
        best_ue_weights = [1]
        best_bs_weights = [1]
    
    
    elif matrix_instead_of_loop:
        # create macro channel matrix (AE_UE x N_GOB) x AE_BS
        H = np.vstack(ch_resp)
        
        # get full codebook
        W = codebook_subset
        
        # assume * means the dot product (Although in Python it doesn't!!!)
        # W_UE = (H*W)^H/|H*W|
        # W_UE * H * W = |H*W| = channel gain
        
        ch_gains_matrix = abs(np.dot(H, W))
        # IF DOESN'T WORK: try to compute the hermitian and multiply both.
        
        list_of_ch_gains = np.diag(ch_gains_matrix)
        
        # Compute best beam pair detailed information
        best_beam_idx = np.argmax(list_of_ch_gains)
        # best_azi_idx = np.floor(best_beam_idx / 11).astype(int)
        # best_el_idx = best_beam_idx % 11
        curr_max_ch_gain = abs(list_of_ch_gains[best_beam_idx])
        
        w_bs = W[:, best_beam_idx]
        w_ue = np.dot(ch_resp, w_bs).conj().T
        w_ue = w_ue / np.linalg.norm(w_ue)
            
        best_ue_weights = w_ue
        best_bs_weights = w_bs
                
        if save_power_per_CSI_beam:
            power_per_beam_list = list_of_ch_gains
    else:
        for dir_idx in range(codebook_subset.shape[1]):
            
            w = codebook_subset[:,dir_idx]
            
            # Compute internal product between ch coeffs and precoder, 
            # that is what the UE will see from a transmission with w
            at_ue_ant = np.dot(ch_resp, w)
            
            # The UE will use the Maximum Ratio Beamformer, 
            # both for receiving and for transmitting
            mr_precoder = at_ue_ant.conj().T
            mr_precoder = mr_precoder / np.linalg.norm(mr_precoder)
            
            # Resulting in a amplitude channel gain of:
            ch_gain = np.dot(at_ue_ant, mr_precoder)
            
            # The channel gain should be a scalar by now...
            # Save the precoder that performs the best
            
            if abs(ch_gain) > curr_max_ch_gain:
                best_idx = dir_idx
                curr_max_ch_gain = abs(ch_gain)
                best_ue_weights = mr_precoder
                best_bs_weights = w
            
            if save_power_per_CSI_beam:
                power_per_beam_list.append(abs(ch_gain))
    
    
    # Create and load the best Beam Pair found
    beam_pair = Beam_pair()
    
    best_beam_relative_idxs.append(best_idx)
    beam_pair.beam_idx = q_idxs[best_idx]
    beam_pair.ang = codebook_subset_directions[:,best_idx]
    
    beam_pair.bs_weights = best_bs_weights
    beam_pair.ue_weights = best_ue_weights
    
    # Save the channel gain (linear/electric field)**2 = power gain
    beam_pair.ch_power_gain = curr_max_ch_gain ** 2
    
    beam_pairs.append(beam_pair)
    
    # Return best n_best beams 
    return (beam_pairs, power_per_beam_list, best_beam_relative_idxs)


def update_precoders(bs, ue, curr_beam_pairs, precoders_dict, curr_coeffs, 
                     last_coeffs, tti_csi, n_layers, n_csi_beams, rot_factor, 
                     power_per_beam, save_power_per_CSI_beam, vectorize):
    
    """
    For a BS-UE pair, compute the best beam per polarisation.
    This can be done independently of the beam at the ue side. 
    After finding the best pair, update
    """
    
    # tti_csi is the tti relative to the coefficients from where the csi
    # should be use. If it is negative, it means the coefficients to use are
    # the last_coeffs
    
    coeffs = curr_coeffs
    if tti_csi < 0:
        coeffs = last_coeffs
    
    # mean across frequency
    mean_coeffs = []
    
    # Note: Currently we using the same codebook_subset for both layers.
    subset_GoB = not (rot_factor is None)
    if subset_GoB:
        q_idxs = orthogonal_precoder_indices1(N1=4, N2=4, O1=4, O2=4, 
                                              RI=n_layers, q=rot_factor)
    else:
        q_idxs = np.arange(precoders_dict[(bs, 'n_directions')])
    
    codebook_subset = precoders_dict[(bs, 'matrix')][:, q_idxs]
    # The channel response is a square matrix of AE_UE x AE_BS
    [azi_len, el_len] = precoders_dict[(bs, 'size')]
    
    codebook_subset_directions = precoders_dict[(bs, 'directions')]
    
    for l in range(n_layers):
        # Compute the means across frequency
        mean_coeffs.append(np.mean(coeffs[(bs, ue)][:,:,:,tti_csi], 2))
        
        # Save list of best beam pairs on that polarisation combination
        (best_beam_pairs, power_per_beam[l], best_beam_relative_idxs) = \
            find_best_beam_pairs(codebook_subset, azi_len, el_len, q_idxs,
                                 codebook_subset_directions, mean_coeffs[l], 
                                 bs, n_csi_beams, 
                                 save_power_per_CSI_beam, vectorize)
        
        # Best Beam Pairs is a list with the best n_csi_beams pairs for a layer
        
        # Remove beams picked from codebook (so the next layer doesn't pick them)
        if n_layers > 1:
            # Trim indices, codebook and directions accordingly
            q_idxs = np.delete(q_idxs, best_beam_relative_idxs)
            codebook_subset = np.delete(codebook_subset, 
                                        best_beam_relative_idxs, axis=1)
            codebook_subset_directions = np.delete(codebook_subset_directions, 
                                                   best_beam_relative_idxs, 
                                                   axis=1)
        
        # Take the beam list and compress it to a single precoder
        # e.g. by scaling each beam according with the feedback
        # beam list is n_csi_beams=l long, and we need to merge it into 1 
        # beam pair (don't forget to update the receiver precoder!)
        if n_csi_beams == 1:
            created_beam_pair = best_beam_pairs[0]
        else:
            pass # to implement when l>1
        curr_beam_pairs[(bs,ue,l)] = created_beam_pair
                
    return

def orthogonal_precoder_indices1(N1, N2, O1, O2, RI, q, q1=-1, q2=-1):
    """
    Parameters
    ----------
    N1 : Logical ports along horizontal.
    N2 : Logical ports along vertical.
    O1 : Oversampling factor along horizontal logical ports.
    O2 : Oversampling factor along vertical logical ports.
    RI : Rank Indicator (1 = single rank/layer codebook, 2 = dual-rank)
    q: rotation factor (0 to 15):
        3 7 11 15
        2 6 10 14
        1 5  9 13
        0 4  8 12
    q1: horizontal rotation factor
    q2: vertical rotation factor
    Returns
    -------
    q_idxs : column indices of orthogonal beams in the set given by q.
    """
    
    if q1 != -1 and q2 != -1:
        q = q2 + q1 * O1
    
    
    # Step 1: Map to the column index of the first beam in the orthogonal set
    gob_col_size = N2 * O2
    
    if q <= 3:
        pass 
    elif 4 <= q <= 7:
        q = q + (gob_col_size - O2) * 1
    elif 8 <= q <= 11:
        q = q - 8 + (gob_col_size - O2) * 2
    elif 12 <= q <= 15:
        q = q - 12 + (gob_col_size - O2) * 3
    else: 
        raise Exception('That value of q is not supported. Integers 0 to 15.')
    
    # Step 2: Sum 'offsets' to get the remaining beams in the set
    q_col_idxs = q + np.arange(0,N2) * N2*O2*O1
    q_idxs_list = [q_idx + np.arange(0,N1) * N1 for q_idx in q_col_idxs]
    q_idxs = np.array(q_idxs_list).reshape((-1))
    
    if RI == 2:
        q_idxs = np.hstack((q_idxs, q_idxs + N1*N2*O1*O2))

    return q_idxs


def orthogonal_precoder_indices2(N1, N2, O1, O2, q, q1=-1, q2=-1):    
    """
    THIS FUNCTION IS NOT WORKING. It's the 3GPP way of getting the rotation
    factor q.
    """
    N1 = 4
    N2 = 4
    O1 = 4
    O2 = 4
    N = N1 * N2
    n1 = np.arange(0, N1)
    n2 = np.arange(0, N1)
    q1 = np.arange(0, O1)
    q2 = np.arange(0, O2)
    q1 = 0
    q2 = 0
    k1 = O1 * n1 + q1
    k2 = O2 * n2 + q2
    
    # continue here...
    k = k2 + k1 * N1*O1
    print(N,k)
    q = q2 + q1 * O1  # 0-15
    q1 = q * 4
    q2 = q * 4
    q_idxs = []
    
    return q_idxs

"""
OLLA functions
"""

def apply_olla(cqi, olla):
    """
    Part of the Outer Loop Link Adaptation mechanism to enhance cqi estimation:
        Changes the CQI estimation.
    """
    
    d = olla
    
    if np.random.uniform() < np.ceil(d) - d:
        x = np.floor(d)
    else:
        x = np.ceil(d)
    
    updated_cqi = int(cqi + x)
    
    if updated_cqi < 1:
        updated_cqi = 1
    
    if updated_cqi > 15:
        updated_cqi = 15
    
    return updated_cqi


def update_olla(olla, success_or_fail, bler_target, step_size):
    """
    Part of the Outer Loop Link Adaptation mechanism to enhance cqi estimation:
        Updates the OLLA parameter in accordance to the correctness of the 
        past estimation.
    """
    
    d = olla
    
    if success_or_fail == 'success':
        d = d + bler_target * step_size
    else:
        d = d - (1 - bler_target) * step_size
    
    new_olla = d
    return new_olla



"""
Polaristion combination functions and single vs multi-layer transmission
bitrates estimation
"""

def su_mimo_setting_bitrate_single_layer(bs, ue, n_prb,
                                         tx_pow_dl_per_layer, curr_beam_pairs, 
                                         est_dl_interference, noise_power,
                                         tti, TTI_duration, 
                                         freq_compression_ratio,
                                         use_olla, olla, debug=0):
    
    """
    Estimates the bitrate that results from transmitting with one 
    polarisation and receiving with both.
    
    Check which polarisation would get the best SINR and uses that polarisation
    with double the power.
    
    This is used to decide if it is best to use one layer or two layers 
    to a UE.
    """
    
    if debug:
        print('---------- Option 1 ------------')
    

    beam_pair = curr_beam_pairs[(bs, ue, 0)]
    signal_power = tx_pow_dl_per_layer * beam_pair.ch_power_gain
    
    sinr_db = \
        10 * np.log10(signal_power /   # single layer interference
                      (noise_power + est_dl_interference[tti][ue][0]))
    
    if debug:
        print(f"RSS: {10 * np.log10(signal_power):.2f} dBW")
        print(f"Estimated interference: "
              f"{10 * np.log10(est_dl_interference[tti][ue][0]):.2f} dbW")
        print(f"Noise: {10 * np.log10(noise_power):.2f} dBW")
        print(f"SINR: {sinr_db:.2f} dB")
    
    (cqi, bler) = calc_CQI(sinr_db)
    if debug:
        print(f"CQI: {cqi:.2f}")
        print(f"BLER: {bler:.2f}")
    
    if use_olla:
        cqi = apply_olla(cqi, olla)
        if debug:
            print(f"CQI after OLLA: {cqi:.2f}")
    
    bits_across = estimate_bits_to_be_sent(cqi, n_prb, freq_compression_ratio)
    
    if debug:
        print(f"Option 1 bits across estimation: {bits_across:.2f}")
    
    # From the bits (estimated to be) sent, calc estimated throughput
    bitrate = (bits_across / ut.get_seconds(TTI_duration))
    
    if debug:
        print(f"bit rate: {bitrate:.2f} bps")
    
    return bitrate


def su_mimo_setting_bitrate_dual_layer(bs, ue, n_layers, n_prb,
                                       tx_pow_dl_per_layer, curr_beam_pairs, 
                                       est_dl_interference, noise_power,
                                       tti, TTI_duration, 
                                       freq_compression_ratio,
                                       use_olla, olla, debug=0):
    """
    Estimates the bitrate of transmitting with two polarisation and receiving
    with two.
    This is used to decide if it is best to use one layer or two layers 
    to a UE.
    
    NOTE: THIS FUNCTION, ONLY RETURNS THE SUM OF THE BITRATES OF THE TWO
          LAYERS WITH THE ESTIMATED INTEFERENCE PER LAYER
    """
    
    # TODO: ADD multi-layer interference or some way of choosing when 
    #       multi-layer transmission is worth having.
    #       Don't forget to change in su_mimo_choice
    
    if debug:
        print('---------- Option 2 ------------')
    
    opt2_sinr_db = [0] * n_layers
    opt2_cqi = [0] * n_layers
    opt2_bler = [0] * n_layers
    opt2_bits_across = [0] * n_layers
    opt2_bitrates = [0] * n_layers
    
        
    # Compute signal power of both layers
    for l in range(n_layers):
        beam_pair = curr_beam_pairs[(bs, ue, l)]
        signal_power = tx_pow_dl_per_layer * beam_pair.ch_power_gain
        
        if debug:
            print(f"Opt2-L{l} RSS: {10*np.log10(signal_power*1e3):.2f} dBm")
            print(f"Opt2-L{l} Inter: "
                  f"{10*np.log10(est_dl_interference[tti][ue][l]*1e3):.2f} db")
            print(f"Opt2-L{l} Noise: {10*np.log10(noise_power*1e3):.2f} dBm")

            
        opt2_sinr_db[l] = 10 * np.log10(signal_power / 
                                        (est_dl_interference[tti][ue][l]
                                         + noise_power))
    
        if debug:
            print(f"Opt2-L{l} SINR: {opt2_sinr_db[l]:.2f} dB")
        
        (opt2_cqi[l], opt2_bler[l]) = calc_CQI(opt2_sinr_db[l])
        if debug:
            print(f"Opt2-L{l} CQI: {opt2_cqi[l]:.2f}")
            print(f"Opt2-L{l} BLER: {opt2_bler[l]:.2f}")
        
        if use_olla:
            opt2_cqi[l] = apply_olla(opt2_cqi[l], olla)
            if debug:
                print(f"Opt2-P{l} MCS idx after OLLA: {opt2_cqi[l]:.2f}")
    
        opt2_bits_across[l] = \
            estimate_bits_to_be_sent(opt2_cqi[l], n_prb, 
                                     freq_compression_ratio)
    
        if debug:
            print(f"Opt2-P{l} bits across est.: {opt2_bits_across[l]:.2f}")
    
        # From the bits estimated to be sent in the TTI, 
        # compute the expected bitrate
        opt2_bitrates[l] = (opt2_bits_across[l] / 
                            ut.get_seconds(TTI_duration))
    
        if debug:
            print(f"Opt2-L{l} partial bit rate: {opt2_bitrates[l]:.2f}")

    opt2_final_bitrate = sum(opt2_bitrates)
    if debug:
        print(f"Opt2-L{l} bit rate: {opt2_final_bitrate:.2f}")
    
    return opt2_final_bitrate
    

"""
Scheduling data structures and functions
"""

class Schedule_entry():
    def __init__(self, bs, ue, beam_pair):
        # Serving BS
        self.bs = bs
        # Served UE
        self.ue = ue
        # Beam pair used
        self.beam_pair = beam_pair
        
        # When transmitting two layers, this indexes the layer of each entry.
        self.layer_idx = 0
            
        # The remaining variables will be filled only when a more accurate
        # estimation is made: after the MU-MIMO choice of UEs
        self.est_sinr = 0
        #  Estimated bits to go across in the current tti
        self.bits_to_send = 0
        # Estimated achievable bitrate
        self.est_bitrate = 0
        # TX power
        self.tx_power = 0
        # Estimated MCS index (or cqi, who cares)
        self.cqi = 0
        # Number of PRBs allocated to this entry
        self.n_prbs = 0
        # Transport Block Maximum Size
        self.tb_max_size = 0
    
    
    def print_entry(self):
        
        print(f'Entry: BS {self.bs}, UE {self.ue}, '
              f'Beam Angle [{self.beam_pair.ang[0]:3},'
              f'{self.beam_pair.ang[1]:3}], '
              f'Tx Power {self.tx_power} W, CQI {self.cqi}, '
              f'PRBS {self.n_prbs}, Est_bitrate {self.est_bitrate} bits/s')


def are_beam_pairs_compatible(bp1, bp2, beam_dist_lim):
    """
    Computes the distance between beams, and based on that returns the 
    compatiblity between beams.
    
    If the beam is at least the so much appart, it is compatible. Less than
    that, and it is not.
    """
    # TODO: update for the new GoB (currently only works for beam_distance=1)
    
    # Don't compute distances if the limit distance if off
    if beam_dist_lim <= 0:
        return True
    
    beam_distance = bp1.beam_idx - bp2.beam_idx
    
    if bp1.beam_idx - bp2.beam_idx <= 1:
        pass

    #beam_distance = np.linalg.norm(abs(np.array(bp1.beam_idx) - 
                                       #np.array(bp2.beam_idx)))
    
    return beam_distance >= beam_dist_lim


def is_compatible_with_schedule(new_entry, schedule, beam_dist_lim):
    """
    Iterates over the schedule entries and returns if the beam is compatible
    or not.
    """
    
    is_compatible = True
    for schedule_entry in schedule:
        # Skip verification with layers of the say UE.
        if schedule_entry.ue == new_entry.ue:
            continue
        
        if are_beam_pairs_compatible(new_entry.beam_pair,
                                     schedule_entry.beam_pair, 
                                     beam_dist_lim):
            continue
        else:
            is_compatible = False
            break
    
    return is_compatible
    
    
def compute_interference(entry, schedule):
    """
    Computes the intra-cell and inter-cell interference caused by the 
    other entries on the schedule on the given entry in the schedule.
    """
    
    
def print_schedule(schedule):
    """
    Prints all information in the schedule.
    """
    
    print('Index, (BS, UE, Beam_pair angle), tx pow, cqi, n_prb, est_bitrate')
    
    for entry_idx in range(len(schedule)):
        print(f'{entry_idx} ', end='')
        schedule[entry_idx].print_entry()
        

def get_delayed_tti(tti, tti_rel, tti_delay):
    delayed_tti = tti_rel - tti_delay
    
    # prevention for the first couple of ttis, where the delay can't be applied
    if tti < tti_delay:
        delayed_tti = 0

    return delayed_tti




##############################################################################
############ Wrappers Zone - Functions to improve readibility ################
##############################################################################

# The only purpose of having functions with a lot of tasks is to make the 
# main file more easily readable, and as small as possible.


def copy_interference_estimates(n_ue, n_layers, interference, tti):
    """
    Copies the interference estimates from the last tti for all UEs
    """
    for ue in range(n_ue):
        for l in range(n_layers):
            interference[tti][ue][l] = interference[tti-1][ue][l]
            

def copy_avg_bitrate(n_ue, avg_bitrate, tti):
    """
    Propagate the average throughput, for fair
    """
    for ue in range(n_ue):
        avg_bitrate[tti][ue] = avg_bitrate[tti-1][ue]


def copy_power_per_beam(power_per_beam, n_phy, n_layers, tti):
    for ue in range(n_phy):
        for l in range(n_layers):
            power_per_beam[tti][ue][l][:] = \
                power_per_beam[tti - 1][ue][l][:]


def tti_info_copy_and_update(tti, TTI_duration, first_coeff_tti, n_phy, 
                             n_layers, est_dl_interference, avg_bitrate, 
                             olla, use_olla, power_per_beam,
                             save_power_per_CSI_beam):

    
    # Timestamp used for packets (note that tti is zero-indexed)
    tti_timestamp = (tti + 1) * TTI_duration
    
    # tti relative to the current time division (for coeff management purposes)
    tti_relative = tti - first_coeff_tti
    
    # This just needs to be here because the UL TTI jumps over the rest of the 
    # loop. This way, we guarantee that the next DL slot has the right info
    
    # Copy interference estimated from the previous tti
    copy_interference_estimates(n_phy, n_layers, est_dl_interference, tti)
    
    # Copy bit rate averages from the previous tti
    copy_avg_bitrate(n_phy, avg_bitrate, tti)

    if save_power_per_CSI_beam:
        if tti > 0:
            copy_power_per_beam(power_per_beam, n_phy, n_layers, tti)

    if use_olla:
        # copy ollas from last TTI if the ollas in this TTI are null
        for ue in range(n_phy):
            if olla[tti][ue] == 0:
                olla[tti][ue] = olla[tti-1][ue]
                

    return tti_timestamp, tti_relative


def update_queues(ue_idxs, buffers, tti_timestamp, active_UEs, tti):
    for ue in ue_idxs:
        
        # 1- Queue Update 
        #   a) Add new packets (update entry cursor)
        #   b) Update head of queue delay
        #   c) Discard packets that won't make it in the latency budged
        buffers[ue].update_queue_time(tti_timestamp)
    
        if not buffers[ue].is_empty:
            # These UEs have something to send this TTI.
            active_UEs[tti].append(ue)



############### CSI UPDATE WRAPPERS ##############
def interference_measurements_update(ues, n_layers, tti, last_csi_tti, 
                                     csi_tti_delay, est_interference,
                                     real_interference):
    for ue in ues:
        for p in range(n_layers):
            # 2- Update Downlink Interference Estimation from previous 
            # CSI Measurements
            
            t = last_csi_tti - csi_tti_delay
            est_interference[tti][ue][p] = real_interference[t][ue][p]
            
            # however, if the measurement is null, use a past measurement
            if est_interference[tti][ue][p] == 0:
                # last_non_zero = np.max(np.nonzero(est_interference[:][ue][p]))
                est_interference[tti][ue][p] = est_interference[tti-1][ue][p]


def update_all_precoders(tti, tti_with_csi, active_UEs, n_bs, 
                         curr_beam_pairs, last_csi_tti, 
                         precoders_dict, coeffs, last_coeffs, 
                         n_layers, n_csi_beams, rot_factor, power_per_beam,
                         save_power_per_CSI_beam, vectorize):
    
    for ue in active_UEs[tti]:
        for bs in range(n_bs):
            # 1- Precoder Estimation
            # a) Check if the precoder should be updated (CSI periodicity)
            if curr_beam_pairs[(bs, ue, 0)].last_updated < last_csi_tti:
                pow_per_beam = [[] for i in range(n_layers)]
                # Update the precoders of all layers with info from the 
                # latest_csi_tti (passed relative: tti_with_csi)
                update_precoders(bs,
                                 ue,
                                 curr_beam_pairs,
                                 precoders_dict,
                                 coeffs,
                                 last_coeffs,
                                 tti_with_csi,
                                 n_layers,
                                 n_csi_beams,
                                 rot_factor, 
                                 pow_per_beam,
                                 save_power_per_CSI_beam,
                                 vectorize)
                
                if save_power_per_CSI_beam:
                    power_per_beam[tti][ue] = pow_per_beam
                
                for l in range(n_layers):
                    curr_beam_pairs[(bs, ue, l)].last_updated = tti


############### SCHEDULING UPDATE WRAPPERS ##############

def su_mimo_choice(tti, tti_for_scheduling, bs_max_pow, 
                   schedulable_UEs_dl, serving_BS_dl,
                   n_layers, n_prb, curr_beam_pairs,
                   est_dl_interference, wideband_noise_power_dl, 
                   TTI_duration, freq_compression_ratio, 
                   use_olla, olla, debug_su_mimo_choice, 
                   su_mimo_bitrates, est_su_mimo_bitrate, su_mimo_setting,
                   dl_radio_efficiency, bandwidth_mult):
    
    """
    Serves to make a first estimation of how many independent layers to
    schedule for each UE. 
    Returns the bitrates of the two options, the estimated bitrate of the 
    best option and which option was the best, respectively, in 
    'su_mimo_bitrates', 'est_su_mimo_bitrate' and 'su_mimo_setting'.
    
    """
    
    for ue in schedulable_UEs_dl:
        bs = serving_BS_dl[ue]
        # estimate the power of each layer
        tx_pow_dl_per_layer = bs_max_pow / len(schedulable_UEs_dl)
        
        # Option 1: single-layer
        option1_bitrate = \
            su_mimo_setting_bitrate_single_layer(bs, 
                                                 ue,
                                                 n_prb,
                                                 tx_pow_dl_per_layer, 
                                                 curr_beam_pairs, 
                                                 est_dl_interference,
                                                 wideband_noise_power_dl,
                                                 tti_for_scheduling,
                                                 TTI_duration,
                                                 freq_compression_ratio,
                                                 use_olla, 
                                                 olla[tti][ue],
                                                 debug=debug_su_mimo_choice)
        
        if n_layers > 1:
            # estimate the power of each layer
            tx_pow_dl_per_layer = (bs_max_pow / 
                                   (len(schedulable_UEs_dl) * n_layers))
            
            # Option 2: dual-layer
            option2_bitrate = \
                su_mimo_setting_bitrate_dual_layer(bs, 
                                                   ue, 
                                                   n_layers,
                                                   n_prb,
                                                   tx_pow_dl_per_layer, 
                                                   curr_beam_pairs, 
                                                   est_dl_interference,
                                                   wideband_noise_power_dl,
                                                   tti_for_scheduling,
                                                   TTI_duration,
                                                   freq_compression_ratio,
                                                   use_olla, 
                                                   olla[tti][ue],
                                                   debug=debug_su_mimo_choice)
        else:
            option2_bitrate = 0
        
        su_mimo_bitrates[tti][ue][:] = bandwidth_mult * dl_radio_efficiency * \
            np.array([option1_bitrate, option2_bitrate])
        est_su_mimo_bitrate[tti][ue] = max(su_mimo_bitrates[tti][ue][:])
        
        # Number of layers to be transmitted for each ue
        su_mimo_setting[tti][ue] = 1 + \
            np.where(su_mimo_bitrates[tti][ue] == 
                     est_su_mimo_bitrate[tti][ue])[0][0]
        
        if debug_su_mimo_choice:
            print(f"Su_mimo_bitrates: {su_mimo_bitrates[tti][ue]}")
            print(f"est_su_mimo_bitrates: {est_su_mimo_bitrate[tti][ue]}")
            print(f"su_mimo_setting: {su_mimo_setting[tti][ue]}")


def compute_priorities(tti, ue_priority, all_delays, buffers, 
                       schedulable_UEs_dl, scheduler_name, avg_bitrate, 
                       est_su_mimo_bitrate, delay_threshold, 
                       scheduler_param_delta, scheduler_param_c):
        
    # Current head of queue delays of all UEs, necessary for scheduling 
    # with a latency-aware scheduler
    
    all_delays[tti] = [buffers[ue].head_of_queue_lat.microseconds * 1e6 
                       for ue in schedulable_UEs_dl]
    
    # schedulable_UEs_dl should only have the UEs that have something to send
    # only for those UEs, we compute priorities and attempt to schedule them.
    
    for ue in schedulable_UEs_dl:
        ue_priority[tti][ue] = \
            scheduler(scheduler_name, 
                      avg_bitrate[tti-1][ue],
                      est_su_mimo_bitrate[tti][ue],
                      ut.get_seconds(buffers[ue].head_of_queue_lat),
                      delay_threshold,
                      scheduler_param_delta,
                      scheduler_param_c,
                      all_delays)
        # print(f"UE {ue} has priority {ue_priority[tti][ue]}")
        
    curr_priorities = sorted([(ue, ue_priority[tti][ue])
                              for ue in schedulable_UEs_dl],
                             key=lambda x: x[1], reverse=True)
    
    return curr_priorities


def mu_mimo_choice(tti, curr_priorities, curr_schedule, serving_BS_dl, 
                   su_mimo_setting, curr_beam_pairs, 
                   min_beam_distance, scheduled_UEs, scheduling_method,
                   scheduled_layers, debug):
    
    """
    The first user has as many layers as it can handle.
    The next users have the layers that are compatible with the layers
    already added to the schedule.
    """
        
    curr_schedule['DL'] = []

    if debug:
        curr_scheduled_ues = []
    
    last_ue = -1

    for (ue, priority) in curr_priorities:
        if debug:
            print(f"UE {ue} has priority {priority}")
            
        bs = serving_BS_dl[ue]
        if scheduling_method == 'SU':
            # check whether the next entry is still for the same UE. If not,
            # do not schedule!
            if ue != last_ue and last_ue != -1:
                break
        elif scheduling_method == 'MU':
            pass # No action needed..

        last_ue = ue

        # The layers in curr_beam_pairs are sorted based on channel_quality
        for l in range(su_mimo_setting[tti][ue]):
            beam_pair = curr_beam_pairs[(bs,ue,l)]
                
            new_schedule_entry = Schedule_entry(bs, ue, beam_pair)
            
            new_schedule_entry.layer_idx = l

            if is_compatible_with_schedule(new_schedule_entry, 
                                           curr_schedule['DL'], 
                                           min_beam_distance):
                
                curr_schedule['DL'].append(new_schedule_entry)
                
                if scheduled_UEs[tti][ue] == 0:
                    scheduled_UEs[tti][ue] = 1
                    if debug:
                        curr_scheduled_ues.append(ue)
                
                scheduled_layers[tti][ue] += 1
                # print(f'UE {ue} layer {l} is compatible with schedule')
            else:
                # print(f'UE {ue} layer with polarisation {p} is NOT '
                #       f'compatible with schedule')
                # if tti > 5:
                #     print(f'here at ue {ue}')
                continue
    
    if debug:
        print(f"MU-MIMO scheduled UEs: {curr_scheduled_ues}")
        print(f"Scheduled layers: {scheduled_layers[tti]}")


def power_control(tti, bs_max_pow, scheduled_UEs, scheduled_layers, 
                  curr_schedule):
    """
    Now that we know which UEs will be scheduled, we may attribute the
    correct power to each one.
    
    Each UE gets the same amount of power. If only one layer is 
    transmitted, it is transmitted with double the power.
    
    ASSUMPTION:
    We don't consider power limitations such as limiting the power per 
    antenna element: if all layers come out
    of the same set of antennas (because the polarisation combo lead to 
    that) then so be it, there will not be some power weighting that is 
    layer dependent.
    """
    
    if len(curr_schedule['DL']) == 0:
        return
    
    tx_pow_per_ue = bs_max_pow / sum(scheduled_UEs[tti])
    
    for schedule_entry in curr_schedule['DL']:
        # If the UE has 2 layers, distribute the power between them
        schedule_entry.tx_power = (tx_pow_per_ue / 
                                   scheduled_layers[tti][schedule_entry.ue])


def final_mcs_update(tti, curr_schedule, est_interference,
                     wideband_noise_power, n_prb, TTI_dur_in_secs,
                     freq_compression_ratio, estimated_SINR, 
                     use_olla, olla, tbs_divisor, efficiency, bw_multiplier,
                     scheduled_UEs, scheduled_layers):
    # With all choices made, there may have been changes to the SINRs
    # Update the estimations such that the best MCS is used
    # There may also have been updates to the interference, pro-actively
    # from the radiation pattern or with other techniques...
    for entry in curr_schedule['DL']:
        
        sinr = calc_SINR(entry.tx_power, 
                         entry.beam_pair.ch_power_gain, 
                         est_interference[tti][entry.ue][entry.layer_idx],
                         wideband_noise_power)
        
        (cqi, bler) = calc_CQI(sinr)
        
        # efficiency has to do with the overhead
        # bw_multiplier has to do with using a different bandwidth than
        # the generation, and there's some scalling there.
        bits = (estimate_bits_to_be_sent(cqi, n_prb, freq_compression_ratio) * 
                efficiency * bw_multiplier)
        
        bitrate = bits / TTI_dur_in_secs
        
        entry.est_sinr = sinr
        estimated_SINR[tti][entry.ue][entry.layer_idx] = sinr
        
        entry.bits_to_send = bits
        entry.est_bitrate = bitrate
        entry.cqi = cqi
        
        # Use OLLA parameter to further tune the mcs choice
        if use_olla:
            entry.cqi = apply_olla(entry.cqi, olla[tti][entry.ue])
        
        
        # Wideband scheduling for now
        entry.n_prbs = n_prb 

        entry.tb_max_size = get_TB_size(entry.bits_to_send, tbs_divisor)


    for entry in curr_schedule['DL']:
        if entry.cqi == 0:
            scheduled_UEs[tti][entry.ue] = 0
            scheduled_layers[tti][entry.ue] -= 1 
            
    curr_schedule['DL'] = [entry for entry in curr_schedule['DL'] 
                           if entry.cqi != 0]
    
    
    
######################### SIMULATION WRAPPERS ######################

def tti_simulation(curr_schedule, slot_type, n_prb, debug, coeffs, 
                   tti_relative, intercell_interference_power_per_prb, 
                   noise_power_per_prb, tti, real_dl_interference, 
                   info_bits_table, buffers, n_transport_blocks, realised_bits, 
                   olla, use_olla, bler_target, olla_stepsize, 
                   blocks_with_errors, realised_SINR, TTI_dur_in_secs, 
                   realised_bitrate_total, beams_used, sig_pow_per_prb, 
                   mcs_used, save_per_prb_variables, experienced_signal_power):
    
    for entry in curr_schedule[slot_type]:
        if debug:
            print(f'_____Entry {curr_schedule[slot_type].index(entry)}: '
                  f'UE {entry.ue}, Layer {entry.layer_idx}')
        
        # Save which precoder was used in said tti
        beams_used[tti][entry.ue][entry.layer_idx][:] = entry.beam_pair.ang[:]
        
        # Save the MCS used
        mcs_used[tti][entry.ue][entry.layer_idx] = entry.cqi
        
        # 1- Compute SINR per PRB
        # But now we have to supply the appropriate ch_coeffs to determine
        # the sinr per PRB
        sig_pow_of_prb = [0] * n_prb
        sinr_per_prb = [0] * n_prb
        interference_pow_per_prb = [0] * n_prb
        
        for prb in range(n_prb):
                
            # Compute the interference from the other schedule entries
            for other_entry in curr_schedule[slot_type]:
                if other_entry == entry:
                    continue
                
                ch_matrix = channel_matrix(coeffs, entry.ue, other_entry.bs,
                                           prb, tti_relative)
                
                # Sum the interference (worst case scenario probabilistically)
                interference_pow_per_prb[prb] += \
                    calc_rx_power_lin(other_entry.tx_power / n_prb, 
                                      other_entry.beam_pair.bs_weights,
                                      entry.beam_pair.ue_weights, ch_matrix)

            # Add the external inter-cell interference 
            interference_pow_per_prb[prb] += \
                intercell_interference_power_per_prb
            
            # Compute Signal Power
            ch_matrix = channel_matrix(coeffs, entry.ue, entry.bs,
                                       prb, tti_relative)
            
            # Here the use of wideband precoders is very clear and evident
            sig_pow_of_prb[prb] = \
                calc_rx_power_lin(entry.tx_power / n_prb, 
                                  entry.beam_pair.bs_weights,
                                  entry.beam_pair.ue_weights,
                                  ch_matrix)
            
            if save_per_prb_variables:
                sig_pow_per_prb[tti][entry.ue][entry.layer_idx][prb] = \
                    sig_pow_of_prb[prb]
                
            
            # Compute the final sinr of the prb
            sinr_per_prb[prb] = 10 * np.log10(sig_pow_of_prb[prb] / 
                                              (interference_pow_per_prb[prb] +
                                               noise_power_per_prb))
        
        
        # We can store all power for a UE like this because there's only
        # one layer:
        experienced_signal_power[tti][entry.ue][entry.layer_idx] = \
            np.mean(sig_pow_of_prb[prb])
        
        # Update realised interference
        real_dl_interference[tti][entry.ue][entry.layer_idx] = \
            sum(interference_pow_per_prb)
        
        # print('First 5 SINRs: ' + 
        #       str([f'{sinr:.2e}, ' for sinr in sinr_per_prb[1:5]]))
        
        # 2- Aggregate and 'average' the sinrs over all the assigned prbs
        if n_prb > 1:
            experienced_sinr = \
                avg_SINRs_MIESM(sinr_per_prb, info_bits_table,
                                k=bits_per_symb_from_cqi(entry.cqi))
        else:
            # for the case of optimised simulations with a single PRB 
            experienced_sinr = sinr_per_prb[0]
        
        if debug:
            print(f'Estimated SINR: {entry.est_sinr:6.1f} dB')
            print(f'Experienced SINR: {experienced_sinr:4} dB')
        
        # 3- Compute the TB size per entry (from n_prbs scheduled and MCS)
        
        # Calculate how many TBS are needed and allocate them
        #    Note: the transport blocks to be used need to be based on the 
        #          bits that are currently waiting in the buffer. Each 
        #          Transport Block is buffer dependent: it has pointers to 
        #          the beginning and ending of the packets inside that tb.
        
            
        # list of (size, start_packet_idx) pairs, one for each TB
        tb_sizes_and_idxs = at.gen_transport_blocks(buffers[entry.ue], 
                                                    entry.bits_to_send,
                                                    entry.tb_max_size, tti)
        
        # The Application traffic takes into account the estimated 
        # of bits, tb_max_size and the buffer state to generate pairs of 
        # (size, start_packet_idx). Then the SLS will actually create them.
        transport_blocks = [Transport_Block(tb_info[0], tb_info[1])
                            for tb_info in tb_sizes_and_idxs]
        
        # 4- Compute the BLER (from SINR and MCS)
        bler = get_BLER_from_fitted_MCS_curves(entry.cqi, 
                                               experienced_sinr)
        success_tb = 0
        fail_tb = 0
        if not buffers[entry.ue].is_empty:
            # 5- Flip a BLER biased coin to assess the errors on TBs
            for tb in transport_blocks:
                # print(f'Flipping with BLER = {bler}. Result: ', end='')
                if ut.success_coin_flip(prob_of_error=bler):
                    # print('Success!')
                    success_tb += 1
                    # If the transport block is well received, remove bits from
                    # buffer
                    realised_bits[tti][entry.ue][entry.layer_idx] += tb.size
                    
                    buffers[entry.ue].remove_bits(tb.size, tb.start_idx)
                    suc_or_fail = 'success'
                    
                else:
                    # print('Fail!')
                    fail_tb += 1
                    # Transport block has errors, don't remove it from the 
                    # buffer instead, update some stats only                
                    blocks_with_errors[tti][entry.ue][entry.layer_idx] += 1
                    suc_or_fail = 'fail'
                    
                if use_olla and len(olla) > tti + 1:
                    olla[tti+1][entry.ue] = update_olla(olla[tti][entry.ue], 
                                                        suc_or_fail, 
                                                        bler_target, 
                                                        olla_stepsize)
        
        # save number of blocks attempted transmitted (successfully or not)
        n_transport_blocks[tti][entry.ue][entry.layer_idx] = \
            len(transport_blocks)
            
        if debug:
            print(f'Flipping with BLER = {bler}. '
              f'Result: {success_tb} Success + {fail_tb} Fail = '
              f'{len(transport_blocks)}')
        
        # 6- Update the realised bitrates with the bitrate of given entry
        realised_SINR[tti][entry.ue][entry.layer_idx] = experienced_sinr
        r = (realised_bits[tti][entry.ue][entry.layer_idx] / 
             TTI_dur_in_secs / 1e6)
        
        realised_bitrate_total[tti][entry.ue] += r
            
        
        if debug:
            print(f'Realised bitrate: {r} Mbits/s') 
         
        
        
def update_avg_bitrates(tti, n_ue, realised_bitrate, avg_bitrate):
    for ue in range(n_ue):
        avg_bitrate[tti][ue] = compute_avg_bitrate(avg_bitrate[tti-1][ue], 
                                                   realised_bitrate[tti][ue], 
                                                   tc=100)
        