import numpy
import torch
import termcolor
import sys
from functools import wraps
import triton
import triton.language as tl

from exitFunction_models import RQPFUNCTIONS_MODEL
import rqpfunctions

ALLOCATIONRATIO = 0.90
TRADINGFEE      = 0.0005

BPST_KVALUE        = 2/(100+1)
BPST_PRINTINTERVAL = 100e6

_TORCHDTYPE = torch.float32

def removeConsoleLines(nLinesToRemove: int) -> None:
    for _ in range (nLinesToRemove): 
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()

def timeStringFormatter(time_seconds: int) -> str:
    if   (time_seconds < 60):    return "00:{:02d}".format(time_seconds)                                                                                                                                  #Less than a minute
    elif (time_seconds < 3600):  return "{:02d}:{:02d}".format(int(time_seconds/60), time_seconds%60)                                                                                                     #Less than an hour
    elif (time_seconds < 86400): return "{:02d}:{:02d}:{:02d}".format(int(time_seconds/3600), int((time_seconds-int(time_seconds/3600)*3600)/60), time_seconds%60)                                        #Less than a day
    else: return "{:d}:{:02d}:{:02d}:{:02d}".format(int(time_seconds/86400), int((time_seconds-int(time_seconds/86400)*86400)/3600), int((time_seconds-int(time_seconds/3600)*3600)/60), time_seconds%60) #More than a day

def BPST_Timer(func):
    ce_beg = torch.cuda.Event(enable_timing=True)
    ce_end = torch.cuda.Event(enable_timing=True)
    @wraps(func)
    def wrapper(*args, **kwargs):
        #Before
        ce_beg.record()
        #Function call
        result = func(*args, **kwargs)
        #After
        ce_end.record()
        ce_end.synchronize()
        t_elapsed_ms = ce_beg.elapsed_time(ce_end)
        #Return
        return result, t_elapsed_ms
    return wrapper

#Batch Processing Triton Kernel Function ================================================================================================================================================================================================================
@triton.jit
def processBatch_triton_kernel(#Constants
                                leverage:        tl.constexpr,
                                allocationRatio: tl.constexpr,
                                tradingFee:      tl.constexpr,
                                #Base Data
                                data_price_open,
                                data_price_high,
                                data_price_low,
                                data_price_close,
                                data_volume,
                                data_volume_tb,
                                data_pip_lst,
                                data_pip_lsp,
                                data_pip_nna,
                                data_pip_csf,
                                data_pip_ivp,
                                data_pip_ivp_stride: tl.constexpr,
                                params_trade_fslImmed,
                                params_trade_fslClose,
                                params_trade_pslReentry: tl.constexpr,
                                params_model,
                                params_model_stride: tl.constexpr,
                                #Result Buffers
                                balance_finals,
                                balance_bestFit_intercepts,
                                balance_bestFit_growthRates,
                                balance_bestFit_volatilities,
                                balance_wallet_history,
                                balance_margin_history,
                                balance_ftIndexes,
                                #Sizes
                                size_paramsBatch: tl.constexpr,
                                size_dataLen:     tl.constexpr,
                                size_loop:        tl.constexpr,
                                #Model & Mode
                                MODELNAME:  tl.constexpr,
                                SEEKERMODE: tl.constexpr
                                ):
    #Process ID Check
    pid = tl.program_id(0)
    if size_paramsBatch <= pid: return

    #Model Parameters and State Trackers <!!! EDIT HERE FOR MODEL ADDITION !!!>
    mp_base_ptr = params_model + (pid * params_model_stride)
    if (MODELNAME == 'ROTATIONALGAUSSIAN1'):
        #Parameters
        mp_delta   = tl.load(mp_base_ptr + 0)
        mp_theta_S = tl.load(mp_base_ptr + 1)
        mp_alpha_S = tl.load(mp_base_ptr + 2)
        mp_beta0_S = tl.load(mp_base_ptr + 3)
        mp_beta1_S = tl.load(mp_base_ptr + 4)
        mp_gamma_S = tl.load(mp_base_ptr + 5)
        mp_theta_L = tl.load(mp_base_ptr + 6)
        mp_alpha_L = tl.load(mp_base_ptr + 7)
        mp_beta0_L = tl.load(mp_base_ptr + 8)
        mp_beta1_L = tl.load(mp_base_ptr + 9)
        mp_gamma_L = tl.load(mp_base_ptr + 10)
        #State Trackers
        rqp_st_pip_csf_prev      = -1.0
        rqp_st_cycleContinuation = -1.0
        rqp_st_cycleBeginPrice   = 0.0
    elif (MODELNAME == 'CLASSICALSIGNALDEFAULT'):
        #Parameters
        mp_delta      = tl.load(mp_base_ptr + 0)
        mp_strength_S = tl.load(mp_base_ptr + 1)
        mp_strength_L = tl.load(mp_base_ptr + 2)
        #State Trackers
        rqp_st_pip_csf_prev = -1.0

    # RQP Values
    rqp_val  = 0.0
    rqp_prev = 0.0
    #Trade Parameters Load
    tp_fsl_immed = tl.load(params_trade_fslImmed + pid)
    tp_fsl_close = tl.load(params_trade_fslClose + pid)
    #Trade Simulation State
    balance_wallet    = 1.0
    balance_allocated = balance_wallet * allocationRatio
    balance_margin    = 1.0
    balance_ftIndex   = -1
    quantity          = 0.0
    entryPrice        = 0.0
    forceExited       = 0.0
    #Balance Trend
    bt_sum         = 0.0
    bt_sum_squared = 0.0
    bt_sum_xy      = 0.0

    #Loop
    for i in range(0, size_loop):
        #[1]: Base Data Load --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        d_price_open  = tl.load(data_price_open  + i) 
        d_price_high  = tl.load(data_price_high  + i) 
        d_price_low   = tl.load(data_price_low   + i) 
        d_price_close = tl.load(data_price_close + i) 
        d_volume      = tl.load(data_volume      + i)
        d_volume_tb   = tl.load(data_volume_tb   + i)
        d_pip_lst     = tl.load(data_pip_lst     + i) 
        d_pip_lsp     = tl.load(data_pip_lsp     + i)
        d_pip_nna     = tl.load(data_pip_nna     + i)
        d_pip_csf     = tl.load(data_pip_csf     + i) 
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





        #[2]: RQP Values  <!!! EDIT HERE FOR MODEL ADDITION !!!> --------------------------------------------------------------------------------------------------------------------------------------
        if (MODELNAME == 'ROTATIONALGAUSSIAN1'):
            (
                rqp_val,
                rqp_st_pip_csf_prev,
                rqp_st_cycleContinuation,
                rqp_st_cycleBeginPrice
            ) = rqpfunctions.rqpf_ROTATIONALGAUSSIAN1.getRQPValue(#Model Parameters <UNIQUE>
                                                                    mp_delta   = mp_delta,
                                                                    mp_theta_S = mp_theta_S,
                                                                    mp_alpha_S = mp_alpha_S,
                                                                    mp_beta0_S = mp_beta0_S,
                                                                    mp_beta1_S = mp_beta1_S,
                                                                    mp_gamma_S = mp_gamma_S,
                                                                    mp_theta_L = mp_theta_L,
                                                                    mp_alpha_L = mp_alpha_L,
                                                                    mp_beta0_L = mp_beta0_L,
                                                                    mp_beta1_L = mp_beta1_L,
                                                                    mp_gamma_L = mp_gamma_L,
                                                                    #Model State Trackers <UNIQUE>
                                                                    st_pip_csf_prev      = rqp_st_pip_csf_prev,
                                                                    st_cycleContinuation = rqp_st_cycleContinuation,
                                                                    st_cycleBeginPrice   = rqp_st_cycleBeginPrice,
                                                                    #Base Data <COMMON>
                                                                    data_price_close = d_price_close,
                                                                    data_pip_csf     = d_pip_csf,
                                                                    rqpVal_prev      = rqp_prev)
        elif (MODELNAME == 'CLASSICALSIGNALDEFAULT'):
            (
                rqp_val,
                rqp_st_pip_csf_prev,
            ) = rqpfunctions.rqpf_CLASSICALSIGNALDEFAULT.getRQPValue(#Model Parameters <UNIQUE>
                                                                     mp_delta      = mp_delta,
                                                                     mp_strength_S = mp_strength_S,
                                                                     mp_strength_L = mp_strength_L,
                                                                     #Model State Trackers <UNIQUE>
                                                                     st_pip_csf_prev = rqp_st_pip_csf_prev,
                                                                     #Base Data <COMMON>
                                                                     data_pip_csf = d_pip_csf,
                                                                     rqpVal_prev  = rqp_prev)

        rqp_val = tl.floor(rqp_val*1e4+0.5)*1e-4

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





        #[3]: Trade Processing ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # region
        #Position Side & Has #qty_entry
        position_side = tl.where(0 < quantity,  1.0, 0.0)
        position_side = tl.where(quantity < 0, -1.0, position_side)
        position_has = (quantity != 0)

        #Exit Conditions
        price_act_FSLImmed = entryPrice * (1.0 - position_side*tp_fsl_immed)
        price_act_FSLClose = entryPrice * (1.0 - position_side*tp_fsl_close)
        price_liquidation  = entryPrice * (1.0 - position_side/leverage)

        price_worst = tl.where(0 < quantity, d_price_low,  d_price_close)
        price_worst = tl.where(quantity < 0, d_price_high, price_worst)
        
        hit_liquidation = position_has & ((position_side*price_worst)   <= (position_side*price_liquidation))
        hit_fslImmed    = position_has & ((position_side*price_worst)   <= (position_side*price_act_FSLImmed))
        hit_fslClose    = position_has & ((position_side*d_price_close) <= (position_side*price_act_FSLClose))

        #Execution Price
        price_execution = tl.where(hit_fslImmed,    price_act_FSLImmed, d_price_close)
        price_execution = tl.where(hit_liquidation, price_liquidation,  price_execution)
        
        #Quantity Reduce
        balance_committed = tl.abs(quantity)  * entryPrice
        balance_toCommit  = balance_allocated * tl.abs(rqp_val)

        status_forceExit        = hit_liquidation | hit_fslImmed | hit_fslClose
        status_positionReversal = (rqp_prev * rqp_val) < 0
        status_partialExit      = (balance_toCommit - balance_committed) < 0

        quantity_new = tl.where(position_has & status_partialExit,          (balance_toCommit / entryPrice) * position_side, quantity)
        quantity_new = tl.where(status_forceExit | status_positionReversal, 0.0,                                             quantity_new)

        quantity_delta = quantity_new - quantity
        profit         = quantity_delta * (entryPrice-price_execution)
        fee            = tl.abs(quantity_delta) * price_execution * tradingFee

        #Wallet Balance Post-Exit Update
        balance_wallet = balance_wallet + (profit - fee) * leverage
        balance_wallet = tl.maximum(balance_wallet, 0.0)
        
        #Allocated Balance Update
        balance_allocated = tl.where(quantity_new == 0.0, 
                                     balance_wallet*allocationRatio, 
                                     balance_allocated) 
        
        #Force Exit State Update
        if (params_trade_pslReentry == False): 
            forceExited = tl.where(status_forceExit,        1.0, forceExited)
            forceExited = tl.where(status_positionReversal, 0.0, forceExited)

        #Quantity Increase
        balance_committed = tl.abs(quantity_new) * entryPrice
        balance_toCommit  = balance_allocated * tl.abs(rqp_val)
        balance_toCommit_entry = tl.maximum(balance_toCommit-balance_committed, 0.0)

        quantity_entry = tl.where(forceExited == 0.0,
                                  (balance_toCommit_entry / price_execution)*tl.where(rqp_val < 0, -1.0, 1.0),
                                  0.0)
        quantity_final = quantity_new + quantity_entry
        
        #Entry Price Update
        entryPrice_new = tl.where(quantity_final == 0.0, 
                                  0.0, 
                                  (tl.abs(quantity_new)*entryPrice + tl.abs(quantity_entry)*price_execution) / tl.maximum(tl.abs(quantity_final), 1e-6))
        
        #Wallet Balance Post-Entry Update
        fee = tl.abs(quantity_entry) * price_execution * tradingFee
        balance_wallet = balance_wallet - fee * leverage
        balance_wallet = tl.maximum(balance_wallet, 0.0)

        #Margin Balance
        balance_margin = balance_wallet + quantity_final * (d_price_close - entryPrice_new) * leverage
        balance_margin = tl.maximum(balance_margin, 0.0)

        #Update State
        balance_ftIndex = tl.where((balance_ftIndex == -1) & (quantity_final != quantity), i, balance_ftIndex)
        quantity   = quantity_final
        entryPrice = entryPrice_new
        # endregion
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #[4]: Balance Trend Trackers ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        first_trade_occurred = (0 <= balance_ftIndex)
        bt_val_x = tl.where(first_trade_occurred, (i-balance_ftIndex).to(tl.float32), 0.0)
        bt_val_y = tl.where(first_trade_occurred, tl.log(balance_wallet),             0.0)
        bt_sum         += bt_val_y
        bt_sum_squared += bt_val_y*bt_val_y
        bt_sum_xy      += bt_val_x*bt_val_y
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #[5]: History Record --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if not SEEKERMODE:
            off_write = pid * size_loop + i
            tl.store(balance_wallet_history  + off_write, balance_wallet)
            tl.store(balance_margin_history  + off_write, balance_margin)
        rqp_prev = rqp_val
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #Balance Trend Evaluation
    bt_n      = (size_loop-balance_ftIndex).to(tl.float32)
    bt_valid  = (0 <= balance_ftIndex) & (1.0 < bt_n)
    bt_n_safe = tl.where(bt_valid, bt_n, 1.0)

    bt_sum_x  = bt_n*(bt_n-1.0)*0.5
    bt_mean_y = bt_sum   / bt_n_safe
    bt_mean_x = bt_sum_x / bt_n_safe

    denominator_growth      = (bt_n * bt_n * (bt_n * bt_n - 1.0)) / 12.0
    denominator_growth_safe = tl.where(bt_valid, denominator_growth, 1.0)
    numerator_growth        = (bt_n * bt_sum_xy) - (bt_sum_x * bt_sum)
    raw_growthRate          = numerator_growth / denominator_growth_safe

    raw_intercepts = bt_mean_y - (raw_growthRate * bt_mean_x)

    bt_var_x = (bt_n * bt_n - 1.0) / 12.0
    bt_var_y = (bt_sum_squared / bt_n_safe) - (bt_mean_y * bt_mean_y)
    
    raw_variance_resid = tl.maximum(bt_var_y - (raw_growthRate * raw_growthRate * bt_var_x), 0.0)
    raw_volatility     = tl.sqrt(raw_variance_resid)

    bt_growthRate = tl.where(bt_valid, raw_growthRate, 0.0)
    bt_intercepts = tl.where(bt_valid, raw_intercepts, 0.0)
    bt_volatility = tl.where(bt_valid, raw_volatility, 0.0)

    #Final Results Store
    tl.store(balance_bestFit_intercepts   + pid, bt_intercepts)
    tl.store(balance_bestFit_growthRates  + pid, bt_growthRate)
    tl.store(balance_bestFit_volatilities + pid, bt_volatility)
    tl.store(balance_finals               + pid, balance_wallet)
    tl.store(balance_ftIndexes            + pid, balance_ftIndex)
# =======================================================================================================================================================================================================================================================










#Exit Function Model ====================================================================================================================================================================================================================================
class exitFunction():
    def __init__(self, modelName, isSeeker, leverage, pslReentry, parameterBatchSize = 32):
        self.MODELNAME = modelName
        self.model     = RQPFUNCTIONS_MODEL[self.MODELNAME]
        self.isSeeker           = isSeeker
        self.leverage           = leverage
        self.pslReentry         = pslReentry
        self.parameterBatchSize = parameterBatchSize

        #Data Set
        self.__data_price_open  = None
        self.__data_price_high  = None
        self.__data_price_low   = None
        self.__data_price_close = None
        self.__data_volume      = None
        self.__data_volume_tb   = None
        self.__data_pip_lst     = None
        self.__data_pip_lsp     = None
        self.__data_pip_nna     = None
        self.__data_pip_csf     = None
        self.__data_pip_ivp     = None
        self.__data_nValidSamples = None

        #Seeker
        self.__seeker = None

    def preprocessData(self, data: numpy.ndarray) -> None:
        # data[:,0]  - Timestamp
        # data[:,1]  - Open Price
        # data[:,2]  - High Price
        # data[:,3]  - Low  Price
        # data[:,4]  - Close Price
        # data[:,5]  - Base Asset Volume
        # data[:,6]  - Base Asset Volume - Taker Buy
        # data[:,7]     - PIP Last Swing Type
        # data[:,8]     - PIP Last Swing Price
        # data[:,9]     - NNA Signal
        # data[:,10]    - PIP Classical Signal Filtered
        # data[:,11:21] - IVP Boundaries

        #[1]: Tensor Conversion
        _data = torch.from_numpy(data[:,1:]).to(device='cuda', dtype=_TORCHDTYPE)
        #---[2-1]: Normalization Base Values
        _closePrice_initial = _data[0,3]
        _nonzero_indices = (_data[:,4] != 0).nonzero()
        if 0 < _nonzero_indices.size(0):
            _first_nonzero_idx       = _nonzero_indices[0,0]
            _baseAssetVolume_initial = _data[_first_nonzero_idx,4]
            if (_first_nonzero_idx != 0): print(termcolor.colored(f"      - [WARNING] None zero-index volume used during the volume normalization. Used Index: {_first_nonzero_idx}", 'light_red'))
        else:
            _baseAssetVolume_initial = 1.0
            print(termcolor.colored("      - [WARNING] No non-zero volume found during the volume normalization. Setting the initial value to 1.0", 'light_red'))
        #---[2-2]: Base Values
        _data[:,0:4] = _data[:,0:4]/_closePrice_initial
        _data[:,4:6] = _data[:,4:6]/_baseAssetVolume_initial
        _data[:,7]   = _data[:,7]  /_closePrice_initial
        #---[2-3]: First Full PIP Index
        _data_pip_hasNan_lastSwing = torch.isnan(_data[:,6:8]).any(dim  =1)
        _data_pip_hasNan_NNASignal = torch.isnan(_data[:,8:9]).any(dim  =1)
        _data_pip_hasNan_CSSignal  = torch.isnan(_data[:,9:10]).any(dim =1)
        _data_pip_hasNan_IVP       = torch.isnan(_data[:,10:21]).any(dim=1)
        if _data_pip_hasNan_lastSwing.all(): print(termcolor.colored("      - [NOTICE] No PIP data found for 'LAST SWING'. User attention advised.", 'light_yellow'))
        if _data_pip_hasNan_NNASignal.all(): print(termcolor.colored("      - [NOTICE] No PIP data found for 'NNA Signal'. User attention advised.", 'light_yellow'))
        if _data_pip_hasNan_CSSignal.all():  print(termcolor.colored("      - [NOTICE] No PIP data found for 'CS Signal'. User attention advised.",  'light_yellow'))
        if _data_pip_hasNan_IVP.all():       print(termcolor.colored("      - [NOTICE] No PIP data found for 'IVP'. User attention advised.",        'light_yellow'))

        #[4]: Zero-Padding
        _data_len        = _data.size(dim=0)
        _data_len_nToPad = (32-(_data_len%32))%32
        self.__data_nValidSamples = _data_len
        if (0 < _data_len_nToPad): _data = torch.cat([_data, torch.zeros((_data_len_nToPad, _data.size(dim = 1)), device='cuda', dtype=_TORCHDTYPE)], dim=0).contiguous()

        #[5]: Finally
        self.__data_price_open  = _data[:,0].contiguous()
        self.__data_price_high  = _data[:,1].contiguous()
        self.__data_price_low   = _data[:,2].contiguous()
        self.__data_price_close = _data[:,3].contiguous()
        self.__data_volume      = _data[:,4].contiguous()
        self.__data_volume_tb   = _data[:,5].contiguous()
        self.__data_pip_lst     = _data[:,6].contiguous()
        self.__data_pip_lsp     = _data[:,7].contiguous()
        self.__data_pip_nna     = _data[:,8].contiguous()
        self.__data_pip_csf     = _data[:,9].contiguous()
        self.__data_pip_ivp     = _data[:,10:20].contiguous()

    def initializeSeeker(self, 
                         paramConfig: list, 
                         nSeekerPoints:        int, 
                         nRepetition:          int,
                         learningRate:         float, 
                         deltaRatio:           float,
                         beta_velocity:        float,
                         beta_momentum:        float,
                         repopulationRatio:    float, 
                         repopulationInterval: int,
                         scoring:              tuple[str, float],
                         scoringSamples:       int,
                         terminationThreshold: float) -> None:
        """
        self.model = [{'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed <NECESSARY>
                      {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close <NECESSARY>
                      {'PRECISION': 4, 'LIMIT': (-1.0000,  1.0000)},    #Delta
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Theta - SHORT
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Alpha - SHORT
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Beta0 - SHORT
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Beta0 - SHORT
                      {'PRECISION': 0, 'LIMIT': (1,        10)},       #Gamma - SHORT
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Theta - LONG
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Alpha - LONG
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Beta0 - LONG
                      {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Beta0 - LONG
                      {'PRECISION': 0, 'LIMIT': (1,        10)},       #Gamma - LONG
                     ]
        """
        #[1]: Specifications
        nParameters = len(self.model)

        #[2]: Seeker Parameters Check [TO BE IMPLEMENTED]
        #---[2-1]:  nSeekerPoints
        if type(nSeekerPoints) is not int: nSeekerPoints = 10
        if not (1 <= nSeekerPoints):       nSeekerPoints = 10
        #---[2-2]:  nRepetition
        if type(nRepetition) is not int: nRepetition = 10
        if not (1 <= nRepetition):       nRepetition = 10
        #---[2-3]:  learningRate
        if type(learningRate) not in (float, int): learningRate = 0.001
        if not (0.0 < learningRate <= 1.0):        learningRate = 0.001
        #---[2-4]:  deltaRatio
        if type(deltaRatio) not in (float, int): deltaRatio = 0.1
        if not (0.0 < deltaRatio < 1.0):         deltaRatio = 0.1
        #---[2-5]:  beta_velocity
        if type(beta_velocity) not in (float, int): beta_velocity = 0.9
        if not (0.0 <= beta_velocity < 1.0):        beta_velocity = 0.9
        #---[2-6]:  beta_momentum
        if type(beta_momentum) not in (float, int): beta_momentum = 0.99
        if not (0.0 <= beta_momentum < 1.0):         beta_momentum = 0.99
        #---[2-7]:  repopulationRatio
        if type(repopulationRatio) not in (float, int): repopulationRatio = 0.1
        if not (0.0 <= repopulationRatio <= 1.0):       repopulationRatio = 0.1
        #---[2-8]:  repopulationInterval
        if type(repopulationInterval) is not int: repopulationInterval = 10
        if not (1 <= repopulationInterval):       repopulationInterval = 10
        #---[2-9]:  scoring
        if type(scoring) is not tuple or not len(scoring) == 2: scoring = ('SHARPERATIO', (1e-4, 1.0))
        _scoring_type, _scoring_params = scoring
        if _scoring_type not in ('FINALBALANCE', 'GROWTHRATE', 'VOLATILITY', 'SHARPERATIO'): scoring = ('SHARPERATIO', (1e-4, 1.0))
        if   _scoring_type == 'FINALBALANCE': scoring = ('FINALBALANCE', None)
        elif _scoring_type == 'GROWTHRATE':   scoring = ('GROWTHRATE',   None)
        elif _scoring_type == 'VOLATILITY':
            if type(_scoring_params) is not tuple or not len(_scoring_params) == 1: _scoring_params = (1e-4,)
            _sp_volatility_min = _scoring_params[0]
            if type(_sp_volatility_min) not in (float, int): _sp_volatility_min = 1e-4
            if not (0.0 < _sp_volatility_min):               _sp_volatility_min = 1e-4
            scoring = ('VOLATILITY', (_sp_volatility_min,))
        elif _scoring_type == 'SHARPERATIO':
            if type(_scoring_params) is not tuple or not len(_scoring_params) == 3: _scoring_params = (1e-4, 1.0, 1.0)
            _sp_volatility_min    = _scoring_params[0]
            _sp_volatility_weight = _scoring_params[1]
            _sp_gainingBonus      = _scoring_params[2]
            if type(_sp_volatility_min) not in (float, int):    _sp_volatility_min    = 1e-4
            if not (0.0 < _sp_volatility_min):                  _sp_volatility_min    = 1e-4
            if type(_sp_volatility_weight) not in (float, int): _sp_volatility_weight = 1.0
            if not (0.0 < _sp_volatility_weight):               _sp_volatility_weight = 1.0
            if type(_sp_gainingBonus) not in (float, int):      _sp_gainingBonus      = 1.0
            if not (0.0 <= _sp_gainingBonus):                   _sp_gainingBonus      = 1.0
            scoring = ('SHARPERATIO', (_sp_volatility_min, _sp_volatility_weight, _sp_gainingBonus))

        #---[2-10]: scoringSamples
        if type(scoringSamples) is not int: scoringSamples = 20
        if not (1 <= scoringSamples):       scoringSamples = 20
        #---[2-11]: terminationThreshold
        if type(terminationThreshold) not in (float, int): terminationThreshold = 0.0001
        if not (0.0 <= terminationThreshold <= 1.0):       terminationThreshold = 0.0001

        #[3]: Rounding Tensor
        params_rounding_factors = 10.0 ** torch.tensor([p['PRECISION'] for p in self.model], device='cuda', dtype=_TORCHDTYPE).unsqueeze(0)

        #[4]: Parameter Configuration Fixed Value Mask Generation
        params_fixed_mask   = torch.zeros(nParameters, dtype=torch.bool,    device='cuda')
        params_fixed_values = torch.zeros(nParameters, dtype=torch.float32, device='cuda')
        for pIndex, val in enumerate(paramConfig):
            if val is None: continue
            params_fixed_mask[pIndex]   = True
            params_fixed_values[pIndex] = val
        params_fixed_values = (torch.round(params_fixed_values * params_rounding_factors) / params_rounding_factors).squeeze(0)

        #[5]: Parameter Range Tensors
        params_min = torch.tensor([[pDesc['LIMIT'][0] for pDesc in self.model]], device='cuda', dtype = _TORCHDTYPE)
        params_max = torch.tensor([[pDesc['LIMIT'][1] for pDesc in self.model]], device='cuda', dtype = _TORCHDTYPE)
        params_min = torch.round(params_min * params_rounding_factors) / params_rounding_factors
        params_max = torch.round(params_max * params_rounding_factors) / params_rounding_factors

        #[6]: Base Tensors
        params_base = torch.rand(size = (nSeekerPoints, nParameters), device = 'cuda', dtype = _TORCHDTYPE)
        velocity    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)
        momentum    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)

        params_base = params_base * (params_max - params_min) + params_min                         #Range Mapping
        params_base = torch.round(params_base * params_rounding_factors) / params_rounding_factors #Rounding
        params_base[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]                 #Fixed Parameters Overwrite

        #[7]: Seeker Update
        self.__seeker = {#Seeker Parameters
                         'paramConfig':          paramConfig.copy(),
                         'nSeekerPoints':        nSeekerPoints,
                         'nRepetition':          nRepetition,
                         'learningRate':         learningRate,
                         'deltaRatio':           deltaRatio,
                         'beta_velocity':        beta_velocity,
                         'beta_momentum':        beta_momentum,
                         'repopulationRatio':    repopulationRatio,
                         'repopulationInterval': repopulationInterval,
                         'scoring':              scoring,
                         'scoringSamples':       scoringSamples,
                         'terminationThreshold': terminationThreshold,
                         #Process Variables
                         '_params_rounding_factors': params_rounding_factors,
                         '_params_fixed_mask':       params_fixed_mask,
                         '_params_fixed_values':     params_fixed_values,
                         '_params_min':              params_min,
                         '_params_max':              params_max,
                         '_params_base':             params_base,
                         '_velocity':                velocity,
                         '_momentum':                momentum,
                         '_currentRepetition':       0,
                         '_currentStep':             1,
                         '_bestResults':             [[] for _ in range (nRepetition)],
                         '_bestScore_delta_ema':     None}
        
        #[8]: Applied Seeker Configuration
        seeker_applied = dict()
        seeker_applied['paramConfig']          = paramConfig.copy()
        seeker_applied['nSeekerPoints']        = self.__seeker['nSeekerPoints']
        seeker_applied['nRepetition']          = self.__seeker['nRepetition']
        seeker_applied['learningRate']         = self.__seeker['learningRate']
        seeker_applied['deltaRatio']           = self.__seeker['deltaRatio']
        seeker_applied['beta_velocity']        = self.__seeker['beta_velocity']
        seeker_applied['beta_momentum']        = self.__seeker['beta_momentum']
        seeker_applied['repopulationRatio']    = self.__seeker['repopulationRatio']
        seeker_applied['repopulationInterval'] = self.__seeker['repopulationInterval']
        seeker_applied['scoring']              = self.__seeker['scoring']
        seeker_applied['scoringSamples']       = self.__seeker['scoringSamples']
        seeker_applied['terminationThreshold'] = self.__seeker['terminationThreshold']

        return seeker_applied
    
    def __getTestParams(self):
        #[1]: Tensors & Scalars
        seeker = self.__seeker
        params_rounding_factors = seeker['_params_rounding_factors']
        params_fixed_mask       = seeker['_params_fixed_mask']
        params_fixed_values     = seeker['_params_fixed_values']
        params_min              = seeker['_params_min']
        params_max              = seeker['_params_max']
        params_base             = seeker['_params_base']

        deltaRatio = seeker['deltaRatio']
        nSeekers, nParams = params_base.shape

        #[1]: Delta Compuation
        delta = params_base * deltaRatio                                               #Raw Value
        delta = torch.round(delta * params_rounding_factors) / params_rounding_factors #Rounded Value
        delta = torch.diag_embed(delta)                                                #Delta Diagonalized

        #[2]: Parameters Base Dimension Expansion
        params_base_expanded = params_base.unsqueeze(1).expand(-1, nParams, -1)

        #[3]: Delta Plus & Minus Parameters
        #---Raw Values
        params_plus  = params_base_expanded + delta
        params_minus = params_base_expanded - delta
        #---Rounding
        params_plus  = torch.round(params_plus  * params_rounding_factors) / params_rounding_factors
        params_minus = torch.round(params_minus * params_rounding_factors) / params_rounding_factors
        #---Limit
        params_plus  = torch.max(torch.min(params_plus,  params_max), params_min)
        params_minus = torch.max(torch.min(params_minus, params_max), params_min)
        #---Fixed Parameters Overwrite
        params_plus[:,:,  params_fixed_mask] = params_fixed_values[params_fixed_mask]
        params_minus[:,:, params_fixed_mask] = params_fixed_values[params_fixed_mask]

        #[4]: Test Parameters Stacking & Flattening
        params_test = torch.stack([params_plus, params_minus], dim=1).reshape(-1, nParams)

        #[5]: Return
        return params_test, params_plus, params_minus
    
    def __scoreResults(self, balance_finals, balance_bestFit_growthRates, balance_bestFit_volatilities):
        scoringType, scoringParam = self.__seeker['scoring']
        #[1]: TYPE - 'FINALBALANCE'
        if (scoringType == 'FINALBALANCE'): 
            scores = balance_finals

        #[2]: TYPE - 'GROWTHRATE'
        elif (scoringType == 'GROWTHRATE'):   
            scores = balance_bestFit_growthRates

        #[3]: TYPE - 'VOLATILITY'
        elif (scoringType == 'VOLATILITY'):
            volatility_min = scoringParam[0]

            scores = 1 / balance_bestFit_volatilities.clamp_min(min = volatility_min)

        #[4]: TYPE - 'SHARPERATIO'
        elif (scoringType == 'SHARPERATIO'):
            volatility_min    = scoringParam[0]
            volatility_weight = scoringParam[1]
            gainingBonus      = scoringParam[2]

            scores = torch.where(1 < balance_finals,
                                 balance_bestFit_growthRates / balance_bestFit_volatilities.clamp_min(min = volatility_min)**volatility_weight + gainingBonus,
                                 balance_bestFit_growthRates)

        return scores

    def runSeeker(self) -> tuple[bool, tuple]:
        #[1]: Parameters
        seeker = self.__seeker
        nSeekerPoints = seeker['nSeekerPoints']
        nParameters   = len(seeker['paramConfig'])
        learningRate  = seeker['learningRate']
        beta_velocity = seeker['beta_velocity']
        beta_momentum = seeker['beta_momentum']
        params_rounding_factors = seeker['_params_rounding_factors']
        params_fixed_mask       = seeker['_params_fixed_mask']
        params_fixed_values     = seeker['_params_fixed_values']
        params_min              = seeker['_params_min']
        params_max              = seeker['_params_max']
        nRepetition_current     = seeker['_currentRepetition']
        currentStep             = seeker['_currentStep']
        eps = 1e-8

        #[1]: Get Test Parameters & Split into batches
        params_test, params_plus, params_minus = self.__getTestParams()
        params_test_batches = torch.split(params_test, self.parameterBatchSize)

        #[2]: Process Batches
        bestResults = []
        scores      = []
        t_elapsed_gpu_simulation_total_ms = 0
        for params_test_batch in params_test_batches:
            #[2-1]: Batch Processing
            balances, t_elapsed_gpu_ms = self.__performOnParams_Timed(params = params_test_batch)
            (balance_finals, 
             balance_bestFit_growthRates, 
             balance_bestFit_volatilities) = balances
            t_elapsed_gpu_simulation_total_ms += t_elapsed_gpu_ms
            
            #[2-2]: Scoring
            scores_batch = self.__scoreResults(balance_finals               = balance_finals,
                                               balance_bestFit_growthRates  = balance_bestFit_growthRates,
                                               balance_bestFit_volatilities = balance_bestFit_volatilities)
            scores.append(scores_batch)
            
            #[2-3]: Best Result Record
            _, max_idx = torch.max(scores_batch, dim = 0)
            max_idx = max_idx.item()
            bestParams = params_test_batch[max_idx].detach().cpu().numpy().tolist()
            bestParams = tuple(round(bestParams[pIndex], pDesc['PRECISION']) for pIndex, pDesc in enumerate(self.model))
            bestResult = (bestParams,                                              #Parameters
                          round(float(balance_finals[max_idx]),               12), #Final Wallet Balance
                          round(float(balance_bestFit_growthRates[max_idx]),  12), #Growth Rate
                          round(float(balance_bestFit_volatilities[max_idx]), 12), #Volatility
                          round(float(scores_batch[max_idx]),                 12)) #Score
            bestResults.append(bestResult)

        #[3]: Best Reulst Record
        bestResult          = max(bestResults, key=lambda x: x[4])
        bestResults         = seeker['_bestResults']
        bestResults_thisRep = bestResults[nRepetition_current]
        if bestResults_thisRep:
            if bestResults_thisRep[-1][4] < bestResult[4]: bestResult_eff = bestResult
            else:                                          bestResult_eff = bestResults_thisRep[-1]
        else:
            bestResult_eff = bestResult
        bestResults_thisRep.append(bestResult_eff)

        #[4]: Compute Gradients
        scores      = torch.cat(scores)
        scores_view = scores.view(nSeekerPoints, 2, nParameters)
        dx = torch.diagonal(params_plus-params_minus, dim1=-2, dim2=-1)
        dx = torch.where(dx == 0, torch.tensor(1e-9, device='cuda'), dx)
        dy = scores_view[:,0,:]-scores_view[:,1,:]
        gradients = dy / (dx + 1e-12)

        #[5]: Update Velocity, Momentum, and Parameters Base
        params_base = seeker['_params_base']
        velocity    = seeker['_velocity']
        momentum    = seeker['_momentum']

        velocity = beta_velocity * velocity + (1 - beta_velocity) * (gradients**2)
        momentum = beta_momentum * momentum + (1 - beta_momentum) * gradients
        velocity_hat = velocity / (1 - beta_velocity ** currentStep)
        momentum_hat = momentum / (1 - beta_momentum ** currentStep)
        
        params_base_step_size = learningRate * momentum_hat / (torch.sqrt(velocity_hat) + eps)
        
        params_base = params_base + params_base_step_size
        params_base = torch.round(params_base * params_rounding_factors) / params_rounding_factors
        params_base = torch.max(torch.min(params_base, params_max), params_min)
        params_base[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]

        seeker['_velocity']    = velocity
        seeker['_momentum']    = momentum
        seeker['_params_base'] = params_base

        #[6]: Compute Best Score Delta EMA
        scoringSamples           = seeker['scoringSamples']
        bestScore                = bestResult[4]
        bestScore_delta_ema      = None
        bestScore_delta_ema_prev = seeker['_bestScore_delta_ema']
        if (scoringSamples+1 <= currentStep):
            #[6-1]: Calculate SMA (the first value)
            if bestScore_delta_ema_prev is None:
                bestScore_deltas_sum = sum((bestResults_thisRep[rIndex][4]/max(bestResults_thisRep[rIndex-1][4], 1e-12))-1 for rIndex in range (1, len(bestResults_thisRep)))
                bestScore_deltas_sma = bestScore_deltas_sum/scoringSamples
                bestScore_delta_ema = bestScore_deltas_sma
            #[6-2]: Calculate EMA
            else:
                bestScore_delta = (bestScore/max(bestResults_thisRep[-2][4], 1e-12))-1
                ema_k = 2/(scoringSamples+1)
                bestScore_delta_ema = (bestScore_delta*ema_k) + (bestScore_delta_ema_prev*(1-ema_k))
            #[6-4]: Update EMA
            seeker['_bestScore_delta_ema'] = bestScore_delta_ema

        #[7]: Check Termination
        terminationThreshold = seeker['terminationThreshold']
        nRepetition_total    = seeker['nRepetition']
        if bestScore_delta_ema is not None and bestScore_delta_ema < terminationThreshold:
            nRepetition_next             = nRepetition_current + 1
            seeker['_currentRepetition'] = nRepetition_next
            if nRepetition_next == nRepetition_total:
                return (True, nRepetition_current, currentStep, bestResults_thisRep[-1])
            else:
                #Reset base parameters, velocity, and momentum
                params_base = torch.rand(size = (nSeekerPoints, nParameters), device = 'cuda', dtype = _TORCHDTYPE)
                params_base = params_base * (params_max - params_min) + params_min                         #Range Mapping
                params_base = torch.round(params_base * params_rounding_factors) / params_rounding_factors #Rounding
                params_base[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]                 #Fixed Parameters Overwrite
                seeker['_params_base'] = params_base
                seeker['_velocity']    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)
                seeker['_momentum']    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)
                #Reset State variables
                seeker['_currentStep']         = 1
                seeker['_bestScore_delta_ema'] = None
                #Return Results
                return (False, nRepetition_current, currentStep, bestResults_thisRep[-1])

        #[8]: Step Count Update
        seeker['_currentStep'] = currentStep+1

        #[9]: Repopulate (If needed)
        repop_interval = seeker['repopulationInterval']
        repop_ratio    = seeker['repopulationRatio']
        if (currentStep % repop_interval == 0):
            #[5-1]: Number of seekers to repopulate (Randomize)
            n_toRepopulate = int(nSeekerPoints * repop_ratio)
            if 0 < n_toRepopulate:
                #[5-2]: Best scores per seeker
                best_scores_perSeeker, _ = torch.max(scores_view.view(nSeekerPoints, -1), dim=1)
                #[5-3]: Repopulation target indices
                bsps_indices_repopTarget = torch.argsort(best_scores_perSeeker)[:n_toRepopulate]
                #[5-4]: Generate random bases parameter
                params_base_new_rt = torch.rand((n_toRepopulate, nParameters), device='cuda', dtype=_TORCHDTYPE)
                params_base_new_rt = params_base_new_rt * (params_max - params_min) + params_min
                params_base_new_rt = torch.round(params_base_new_rt * params_rounding_factors) / params_rounding_factors
                params_base_new_rt[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]
                #[5-5]: Apply newly randomized base parameters
                seeker['_params_base'][bsps_indices_repopTarget] = params_base_new_rt
                seeker['_momentum'][bsps_indices_repopTarget]    = 0.0
                seeker['_velocity'][bsps_indices_repopTarget]    = 0.0

        #[10]: Finally
        return (False, nRepetition_current, currentStep, bestResults_thisRep[-1])

    def __processBatch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #Data
        _size_paramsBatch = params.size(dim = 0)
        _size_dataLen     = self.__data_price_open.size(dim = 0)
        _params_trade = params[:,:2]
        _params_model = params[:,2:]

        #Params Length Padding
        _params_lToPad = (16-(_params_model.size(dim=1)%16))%16
        _params_lToPad = 0
        if (0 < _params_lToPad): _params_model = torch.cat([_params_model, torch.zeros((_size_paramsBatch, _params_lToPad), device='cuda', dtype=_TORCHDTYPE)], dim=1).contiguous()

        #Result Buffers
        _balance_finals               = torch.empty(size = (_size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        _balance_bestFit_intercepts   = torch.empty(size = (_size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        _balance_bestFit_growthRates  = torch.empty(size = (_size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        _balance_bestFit_volatilities = torch.empty(size = (_size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        if self.isSeeker:
            _balance_wallet_history  = torch.empty(size = (1,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
            _balance_margin_history  = torch.empty(size = (1,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        else:
            _balance_wallet_history  = torch.empty(size = (_size_paramsBatch, self.__data_nValidSamples), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
            _balance_margin_history  = torch.empty(size = (_size_paramsBatch, self.__data_nValidSamples), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        _balance_ftIndexes = torch.full(size = (_size_paramsBatch,), fill_value = -1, device = 'cuda', dtype = torch.int32, requires_grad = False)

        #Processing
        _grid = (_size_paramsBatch,)
        processBatch_triton_kernel[_grid](#Constants
                                          leverage        = self.leverage,
                                          allocationRatio = ALLOCATIONRATIO,
                                          tradingFee      = TRADINGFEE,
                                          #Base Data
                                          data_price_open     = self.__data_price_open,
                                          data_price_high     = self.__data_price_high,
                                          data_price_low      = self.__data_price_low,
                                          data_price_close    = self.__data_price_close,
                                          data_volume         = self.__data_volume,
                                          data_volume_tb      = self.__data_volume_tb,
                                          data_pip_lst        = self.__data_pip_lst,
                                          data_pip_lsp        = self.__data_pip_lsp,
                                          data_pip_nna        = self.__data_pip_nna,
                                          data_pip_csf        = self.__data_pip_csf,
                                          data_pip_ivp        = self.__data_pip_ivp,
                                          data_pip_ivp_stride = self.__data_pip_ivp.stride(dim=0),
                                          params_trade_fslImmed   = _params_trade[:,0].contiguous(),
                                          params_trade_fslClose   = _params_trade[:,1].contiguous(),
                                          params_trade_pslReentry = self.pslReentry,
                                          params_model          = _params_model,
                                          params_model_stride   = _params_model.stride(dim = 0),
                                          #Result Buffers
                                          balance_finals               = _balance_finals,
                                          balance_bestFit_intercepts   = _balance_bestFit_intercepts,
                                          balance_bestFit_growthRates  = _balance_bestFit_growthRates,
                                          balance_bestFit_volatilities = _balance_bestFit_volatilities,
                                          balance_wallet_history       = _balance_wallet_history,
                                          balance_margin_history       = _balance_margin_history,
                                          balance_ftIndexes            = _balance_ftIndexes,
                                          #Sizes
                                          size_paramsBatch = _size_paramsBatch,
                                          size_dataLen     = _size_dataLen,
                                          size_loop        = self.__data_nValidSamples,
                                          #Mode
                                          MODELNAME  = self.MODELNAME,
                                          SEEKERMODE = self.isSeeker,
                                          #Triton Config
                                          num_warps  = 1,
                                          num_stages = 2
                                         )

        #Return Result
        if self.isSeeker: return _balance_finals, _balance_bestFit_growthRates, _balance_bestFit_volatilities
        else:
            _indexGrid         = torch.arange(self.__data_nValidSamples, device='cuda', dtype=_TORCHDTYPE).unsqueeze(0)
            _ftIndexes_bc      = _balance_ftIndexes.unsqueeze(1)
            _mask_validRegion  = (_indexGrid >= _ftIndexes_bc) & (_ftIndexes_bc != -1)
            _balance_bestFit_x = _indexGrid - _ftIndexes_bc
            _balance_bestFit_history_raw = torch.exp(_balance_bestFit_growthRates.unsqueeze(1)*_balance_bestFit_x + _balance_bestFit_intercepts.unsqueeze(1))
            _balance_bestFit_history = torch.where(_mask_validRegion, _balance_bestFit_history_raw, float('nan'))
            return _balance_wallet_history, _balance_margin_history, _balance_bestFit_history, _balance_bestFit_growthRates, _balance_bestFit_volatilities

    @BPST_Timer
    def __performOnParams_Timed(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__processBatch(params = params) 

    def performOnParams(self, params: list) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__processBatch(params = torch.tensor(data = params, device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False))
# =======================================================================================================================================================================================================================================================