import triton
import triton.language as tl
from . import simulatorFunctions as sf

KLINEINDEX_OPENTIME:        tl.constexpr = sf.KLINEINDEX_OPENTIME
KLINEINDEX_OPENPRICE:       tl.constexpr = sf.KLINEINDEX_OPENPRICE
KLINEINDEX_HIGHPRICE:       tl.constexpr = sf.KLINEINDEX_HIGHPRICE
KLINEINDEX_LOWPRICE:        tl.constexpr = sf.KLINEINDEX_LOWPRICE
KLINEINDEX_CLOSEPRICE:      tl.constexpr = sf.KLINEINDEX_CLOSEPRICE
KLINEINDEX_VOLBASE:         tl.constexpr = sf.KLINEINDEX_VOLBASE
KLINEINDEX_VOLBASETAKERBUY: tl.constexpr = sf.KLINEINDEX_VOLBASETAKERBUY

"""
FUNCTION MODEL: SPDDEFAULT (Swing Price Deviation Default)
 * The first two parameters are required by the system, and must always be included in the format as they are.
"""
MODEL = [#TRADE PARAMETERS <NECESSARY>
         {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed
         {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close
         #MODEL PARAMETERS
         {'PRECISION': 6, 'LIMIT': (-1.000000, 1.000000)}, #Delta    - SHORT
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Length   - SHORT
         {'PRECISION': 6, 'LIMIT': (-1.000000, 1.000000)}, #Delta    - LONG
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Strength - LONG
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Length   - LONG
        ]
INPUTDATAKEYS = ['SWING_0_LSPRICE',
                 'SWING_0_LSTYPE']

def PROCESSBATCH(**kwargs):
    grid = lambda META: (triton.cdiv(kwargs['size_paramsBatch'], META['size_block']),)
    processBatch_triton_kernel[grid](**kwargs)

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
def processBatch_triton_kernel(#Constants
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
     rqp_prev,
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
     bt_sum_squared,
     bt_sum_xy
    ) = sf.initializeSimulation_triton_kernel(
        params_trade_fslImmed = params_trade_fslImmed,
        params_trade_fslClose = params_trade_fslClose,
        allocationRatio       = allocationRatio,
        size_paramsBatch      = size_paramsBatch, 
        size_block            = size_block
        )

    #Model Parameters
    mp_base_ptr = params_model + (offsets * params_model_stride)
    mp_delta_S    = tl.load(mp_base_ptr + 0, mask = mask)
    mp_strength_S = tl.load(mp_base_ptr + 1, mask = mask)
    mp_length_S   = tl.load(mp_base_ptr + 2, mask = mask)
    mp_delta_L    = tl.load(mp_base_ptr + 3, mask = mask)
    mp_strength_L = tl.load(mp_base_ptr + 4, mask = mask)
    mp_length_L   = tl.load(mp_base_ptr + 5, mask = mask)
    
    #Model State Trackers
    st_lsp_prev = tl.full([size_block,], -1.0, dtype=tl.float32)
    st_lst_prev = tl.full([size_block,], -1.0, dtype=tl.float32)

    #Loop
    for loopIndex in range(0, size_dataLen):
        #[1]: RQP Values  <!!! EDIT HERE FOR MODEL ADDITION !!!> --------------------------------------------------------------------------------------------------------------------------------------
        (rqp_val, 
         st_lsp_prev,
         st_lst_prev
        ) = getRQPValue(
            #Process
            loopIndex = loopIndex,
            #Base Data
            data_klines          = data_klines, 
            data_klines_stride   = data_klines_stride,
            data_analysis        = data_analysis, 
            data_analysis_stride = data_analysis_stride,
            #Model Parameters
            mp_delta_S    = mp_delta_S,
            mp_strength_S = mp_strength_S,
            mp_length_S   = mp_length_S,
            mp_delta_L    = mp_delta_L,
            mp_strength_L = mp_strength_L,
            mp_length_L   = mp_length_L,
            #Model State Trackers
            st_lsp_prev = st_lsp_prev,
            st_lst_prev = st_lst_prev,
            blockSize = size_block,
            #Previous RQP Value Reference
            rqp_prev = rqp_prev
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
         rqp_prev, 
         bt_sum, 
         bt_sum_squared, 
         bt_sum_xy
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
            data_klines             = data_klines,
            data_klines_stride      = data_klines_stride,
            loopIndex               = loopIndex,
            tp_fsl_immed            = tp_fsl_immed, 
            tp_fsl_close            = tp_fsl_close,
            params_trade_pslReentry = params_trade_pslReentry,
            #State & Result Tensors
            rqp_val                = rqp_val, 
            rqp_prev               = rqp_prev,
            balance_wallet         = balance_wallet, 
            balance_allocated      = balance_allocated, 
            balance_margin         = balance_margin,
            quantity               = quantity, 
            entryPrice             = entryPrice, 
            forceExited            = forceExited, 
            nTrades                = nTrades, 
            balance_ftIndex        = balance_ftIndex,
            bt_sum                 = bt_sum, 
            bt_sum_squared         = bt_sum_squared, 
            bt_sum_xy              = bt_sum_xy,
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
    loopIndex,
    blockSize: tl.constexpr,
    #Base Data
    data_klines, 
    data_klines_stride: tl.constexpr,
    data_analysis, 
    data_analysis_stride: tl.constexpr,
    #Model Parameters
    mp_delta_S,
    mp_strength_S,
    mp_length_S,
    mp_delta_L,
    mp_strength_L,
    mp_length_L,
    #Model State Trackers
    st_lsp_prev,
    st_lst_prev,
    #Previous RQP Value Reference
    rqp_prev
    ):
    #[1]: Original State Trackers
    st_lsp_prev_original = st_lsp_prev
    st_lst_prev_original = st_lst_prev

    #[2]: Base Data - Kline
    kline_base_ptr_this = data_klines + (loopIndex * data_klines_stride)
    kline_price_close   = tl.load(kline_base_ptr_this + KLINEINDEX_CLOSEPRICE)

    #[3]: Base Data - Analysis
    analysis_base_ptr_this = data_analysis + (loopIndex * data_analysis_stride)
    analysis_lsp = tl.load(analysis_base_ptr_this + 0)
    analysis_lst = tl.load(analysis_base_ptr_this + 1)

    #[4]: Nan Check
    isNan = (analysis_lst != analysis_lst)
    analysis_lsp = tl.where(isNan, -1.0, analysis_lsp)
    analysis_lst = tl.where(isNan, -1.0, analysis_lst)

    #[5]: PIP Swing Cycle
    isShort_prev = (st_lst_prev_original == 1.0)
    isShort_this = (analysis_lst         == 1.0)
    cycleReset   = (isShort_prev ^ isShort_this)

    #[6]: RQP Value Calculation
    #---[6-1]: Effective Params
    mp_delta_eff    = tl.where(isShort_this, mp_delta_S,    mp_delta_L)
    mp_strength_eff = tl.where(isShort_this, mp_strength_S, mp_strength_L)
    mp_length_eff   = tl.where(isShort_this, mp_length_S,   mp_length_L)
    #---[6-2]: RQP Value
    pd   = tl.where(isShort_this, 1-kline_price_close/analysis_lsp, kline_price_close/analysis_lsp-1)
    dist = pd-mp_delta_eff
    rqpVal_abs = tl.where(mp_delta_eff <= pd,
                          tl.maximum((1-dist/tl.maximum(mp_length_eff, 1e-6))*mp_strength_eff, 0.0),
                          0.0)
    rqpVal_abs = tl.where(mp_length_eff == 0.0, 0.0, rqpVal_abs)
    #---[6-3]: Cyclic Minimum
    rqpVal_abs = tl.where(cycleReset, rqpVal_abs, tl.minimum(rqpVal_abs, tl.abs(rqp_prev)))
    #---[6-4]: Direction
    rqpVal = tl.where(isShort_this, -rqpVal_abs, rqpVal_abs)

    #[7]: States Update
    st_lsp_prev = tl.full(shape = [blockSize,], value = analysis_lsp, dtype = tl.float32)
    st_lst_prev = tl.full(shape = [blockSize,], value = analysis_lst, dtype = tl.float32)
    
    #[8]: Return RQP Value & States
    rqpVal_return          = tl.where(isNan, 0.0,                  rqpVal)
    st_pip_lsp_prev_return = tl.where(isNan, st_lsp_prev_original, st_lsp_prev)
    st_pip_lst_prev_return = tl.where(isNan, st_lst_prev_original, st_lst_prev)
    
    return (rqpVal_return,
            st_pip_lsp_prev_return,
            st_pip_lst_prev_return)