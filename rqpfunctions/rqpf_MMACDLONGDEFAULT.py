import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_ld
from . import simulatorFunctions as sf

KLINEINDEX_OPENTIME:        tl.constexpr = sf.KLINEINDEX_OPENTIME
KLINEINDEX_OPENPRICE:       tl.constexpr = sf.KLINEINDEX_OPENPRICE
KLINEINDEX_HIGHPRICE:       tl.constexpr = sf.KLINEINDEX_HIGHPRICE
KLINEINDEX_LOWPRICE:        tl.constexpr = sf.KLINEINDEX_LOWPRICE
KLINEINDEX_CLOSEPRICE:      tl.constexpr = sf.KLINEINDEX_CLOSEPRICE
KLINEINDEX_VOLBASE:         tl.constexpr = sf.KLINEINDEX_VOLBASE
KLINEINDEX_VOLBASETAKERBUY: tl.constexpr = sf.KLINEINDEX_VOLBASETAKERBUY

"""
FUNCTION MODEL: MMACDLONGDEFAULT (MMACDLONG Default)
 * The first two parameters are required by the system, and must always be included in the format as they are.
"""
MODEL = [{'PRECISION': 4, 'LIMIT': (0.0001,   5.0000)},   #Alpha
         {'PRECISION': 2, 'LIMIT': (1.00,     10.00)},    #Beta
         {'PRECISION': 4, 'LIMIT': (-1.0000,  1.0000)},   #Delta
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - LONG
        ]

INPUTDATAKEYS = ['MMACDLONG_MSDELTAABSMAREL',]

def PROCESSBATCH(**kwargs):
    sf.processBatch(tkf = processBatch, **kwargs)

"""
<Triton Kernel Function>
 * This is an RQP value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate RQP value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
#Batch Processing Triton Kernel Function ================================================================================================================================================================================================================
@triton.autotune(configs=sf.TRITON_AUTOTUNE_CONFIGURATIONS, key=sf.TRITON_AUTOTUNE_KEY)
@triton.jit
def processBatch(
    #Constants
    leverage:        tl.constexpr,
    allocationRatio: tl.constexpr,
    tradingFee:      tl.constexpr,
    #Base Data
    data_klines,
    data_klines_stride:   tl.constexpr,
    data_analysis,
    data_analysis_stride: tl.constexpr,
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
    nTrades_rb,
    #Sizes
    size_paramsBatch: tl.constexpr,
    size_dataLen:     tl.constexpr,
    size_block:       tl.constexpr,
    #Mode
    SEEKERMODE: tl.constexpr
    ):
    (offsets,
     mask,
     tp_fsl_immed,
     tp_fsl_close,
     balance_wallet,
     balance_allocated,
     balance_margin,
     balance_ftIndex,
     quantity,
     entryPrice,
     forceExited,
     nTrades,
     bt_sum,
     bt_sum_xy,
     bt_sum_squared
    ) = sf.initializeSimulation_triton_kernel(
        params_trade_fslImmed = params_trade_fslImmed,
        params_trade_fslClose = params_trade_fslClose,
        allocationRatio       = allocationRatio,
        size_paramsBatch      = size_paramsBatch, 
        size_block            = size_block
        )

    #Model Parameters
    mp_base_ptr = params_model + (offsets * params_model_stride)
    mp_alpha      = tl.load(mp_base_ptr + 0, mask = mask)
    mp_beta       = tl.load(mp_base_ptr + 1, mask = mask)
    mp_delta      = tl.load(mp_base_ptr + 2, mask = mask)
    mp_strength_S = tl.load(mp_base_ptr + 3, mask = mask)
    mp_strength_L = tl.load(mp_base_ptr + 4, mask = mask)
    
    #Model State Trackers
    st_rqpVal_prev                    = tl.full([size_block,], 0.0, dtype=tl.float32)
    st_mmacdlong_msDeltaAbsMARel_prev = 0.0

    #Loop
    for loop_index in range(0, size_dataLen):
        #[1]: RQP Values  <!!! EDIT HERE FOR MODEL ADDITION !!!> --------------------------------------------------------------------------------------------------------------------------------------
        (rqpDir_this,
         rqpVal_this, 
         st_rqpVal_prev,
         st_mmacdlong_msDeltaAbsMARel_prev,
        ) = getRQPValue(
            #Process
            size_block = size_block,
            #Base Data
            loop_index           = loop_index,
            data_klines          = data_klines, 
            data_klines_stride   = data_klines_stride,
            data_analysis        = data_analysis, 
            data_analysis_stride = data_analysis_stride,
            #Model Parameters
            mp_alpha      = mp_alpha,
            mp_beta       = mp_beta,
            mp_delta      = mp_delta,
            mp_strength_S = mp_strength_S,
            mp_strength_L = mp_strength_L,
            #Model State Trackers
            st_rqpVal_prev                    = st_rqpVal_prev,
            st_mmacdlong_msDeltaAbsMARel_prev = st_mmacdlong_msDeltaAbsMARel_prev
            )
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        #[2]: Trade Simulation
        (balance_wallet, 
         balance_allocated, 
         balance_margin,
         balance_ftIndex,
         quantity, 
         entryPrice, 
         forceExited, 
         nTrades,
         bt_sum, 
         bt_sum_xy,
         bt_sum_squared
        ) = sf.processTrade_triton_kernel(
            #Process
            offsets      = offsets, 
            mask         = mask, 
            size_dataLen = size_dataLen,
            SEEKERMODE   = SEEKERMODE,
            #Constants
            leverage        = leverage,
            allocationRatio = allocationRatio,
            tradingFee      = tradingFee,
            #Base Data
            loop_index              = loop_index,
            data_klines             = data_klines,
            data_klines_stride      = data_klines_stride,
            tp_fsl_immed            = tp_fsl_immed, 
            tp_fsl_close            = tp_fsl_close,
            params_trade_pslReentry = params_trade_pslReentry,
            #State & Result Tensors
            rqpDir_this            = rqpDir_this,
            rqpVal_this            = rqpVal_this, 
            balance_wallet         = balance_wallet, 
            balance_allocated      = balance_allocated, 
            balance_margin         = balance_margin,
            quantity               = quantity, 
            entryPrice             = entryPrice, 
            forceExited            = forceExited, 
            nTrades                = nTrades, 
            balance_ftIndex        = balance_ftIndex,
            bt_sum                 = bt_sum, 
            bt_sum_xy              = bt_sum_xy,
            bt_sum_squared         = bt_sum_squared, 
            balance_wallet_history = balance_wallet_history, 
            balance_margin_history = balance_margin_history
            )
    #Balance Trend Evaluation
    sf.evaluateBalanceTrend_triton_kernel(
        size_dataLen                 = size_dataLen,
        offsets                      = offsets,
        mask                         = mask,
        balance_wallet               = balance_wallet,
        nTrades                      = nTrades,
        balance_ftIndex              = balance_ftIndex,
        bt_sum                       = bt_sum,
        bt_sum_xy                    = bt_sum_xy,
        bt_sum_squared               = bt_sum_squared,
        balance_finals               = balance_finals,
        balance_bestFit_intercepts   = balance_bestFit_intercepts,
        balance_bestFit_growthRates  = balance_bestFit_growthRates,
        balance_bestFit_volatilities = balance_bestFit_volatilities,
        balance_ftIndexes            = balance_ftIndexes,
        nTrades_rb                   = nTrades_rb
        )
# =======================================================================================================================================================================================================================================================





"""
<Triton Kernel Function>
 * This is an RQP value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate RQP value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
@triton.jit
def getRQPValue(
    #Process
    size_block: tl.constexpr,
    #Base Data
    loop_index,
    data_klines, 
    data_klines_stride: tl.constexpr,
    data_analysis, 
    data_analysis_stride: tl.constexpr,
    #Model Parameters
    mp_alpha,
    mp_beta,
    mp_delta,
    mp_strength_S,
    mp_strength_L,
    #Model State Trackers
    st_rqpVal_prev,
    st_mmacdlong_msDeltaAbsMARel_prev,
    ):

    #[1]: Base Data - Kline
    """
    <Not Used For This Model>
    kline_base_ptr_this = data_klines + (loop_index * data_klines_stride)
    kline_price_close   = tl.load(kline_base_ptr_this + KLINEINDEX_CLOSEPRICE)
    """

    #[2]: Base Data - Analysis
    analysis_base_ptr_this = data_analysis + (loop_index * data_analysis_stride)
    analysis_mmacdlong_msDeltaAbsMARel = tl.load(analysis_base_ptr_this + 0)

    #[3]: Nan Check
    isNan = (analysis_mmacdlong_msDeltaAbsMARel != analysis_mmacdlong_msDeltaAbsMARel)
    analysis_mmacdlong_msDeltaAbsMARel = tl.where(isNan, 0.0, analysis_mmacdlong_msDeltaAbsMARel)

    #[4]: ABSMARel Cycle
    isShort_prev = (st_mmacdlong_msDeltaAbsMARel_prev  < mp_delta)
    isShort_this = (analysis_mmacdlong_msDeltaAbsMARel < mp_delta)
    cycleReset   = (isShort_prev ^ isShort_this)

    #[5]: RQP Value Calculation
    #---[5-1]: Effective Params
    mp_strength_eff = tl.where(isShort_this, mp_strength_S, mp_strength_L)
    #---[5-2]: MSDeltaAbsMARel Normalization
    x_sign = tl.where(analysis_mmacdlong_msDeltaAbsMARel < 0, -1.0, 1.0)
    x_abs  = tl_ld.pow(tl.abs(analysis_mmacdlong_msDeltaAbsMARel/mp_alpha), mp_beta)
    y_norm = tl_ld.tanh(x_abs)*x_sign
    #---[5-3]: RQP Value
    width = tl.where(isShort_this, mp_delta+1.0, 1.0-mp_delta)
    dist  = tl.abs(y_norm-mp_delta)
    rqpVal_this_abs = tl.maximum(1-(dist/tl.maximum(width, 1e-9)), 0.0)*mp_strength_eff
    rqpVal_this_abs = tl.where(width == 0.0, 0.0, rqpVal_this_abs)
    #---[5-4]: Cyclic Minimum
    rqpVal_this_abs = tl.where(cycleReset, rqpVal_this_abs, tl.minimum(rqpVal_this_abs, tl.abs(st_rqpVal_prev)))
    #---[5-5]: Direction
    rqpVal_this = tl.where(isShort_this, -rqpVal_this_abs, rqpVal_this_abs)
    #---[5-6]: Nan Check
    rqpDir_this = tl.where(isNan, 0.0, tl.where(isShort_this, -1.0, 1.0))
    rqpVal_this = tl.where(isNan, 0.0, rqpVal_this)

    #[6]: State Trackers Update
    st_rqpVal_prev                    = rqpVal_this
    st_mmacdlong_msDeltaAbsMARel_prev = analysis_mmacdlong_msDeltaAbsMARel
    
    #[7]: Return RQP Value & States
    return (rqpDir_this,
            rqpVal_this,
            st_rqpVal_prev,
            st_mmacdlong_msDeltaAbsMARel_prev)