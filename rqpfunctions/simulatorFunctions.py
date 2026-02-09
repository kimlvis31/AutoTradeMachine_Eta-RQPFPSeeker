import triton
import triton.language as tl

TRITON_AUTOTUNE_CONFIGURATIONS = [triton.Config({'size_block': 32},  num_warps= 1, num_stages=2),
                                  triton.Config({'size_block': 32},  num_warps= 1, num_stages=3),
                                  triton.Config({'size_block': 32},  num_warps= 1, num_stages=4),

                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=2),
                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=3),
                                  triton.Config({'size_block': 64},  num_warps= 2, num_stages=4),

                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=2),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=3),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=4),
                                  triton.Config({'size_block': 128}, num_warps= 4, num_stages=5),

                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=2),
                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=3),
                                  triton.Config({'size_block': 256}, num_warps= 8, num_stages=4),

                                  triton.Config({'size_block': 512}, num_warps=16, num_stages=2)
                                 ]
TRITON_AUTOTUNE_KEY = ['size_paramsBatch']

KLINEINDEX_OPENTIME:        tl.constexpr = 0
KLINEINDEX_OPENPRICE:       tl.constexpr = 1
KLINEINDEX_HIGHPRICE:       tl.constexpr = 2
KLINEINDEX_LOWPRICE:        tl.constexpr = 3
KLINEINDEX_CLOSEPRICE:      tl.constexpr = 4
KLINEINDEX_VOLBASE:         tl.constexpr = 5
KLINEINDEX_VOLBASETAKERBUY: tl.constexpr = 6

def processBatch(tkf, **kwargs):
    if kwargs['SEEKERMODE']:
        grid = lambda META: (triton.cdiv(kwargs['size_paramsBatch'], META['size_block']),)
        tkf[grid](**kwargs)
    else:
        kwargs['size_block'] = 32
        kwargs['num_warps']  = 1
        kwargs['num_stages'] = 2
        grid = (triton.cdiv(kwargs['size_paramsBatch'], kwargs['size_block']),)
        tkf.fn[grid](**kwargs)

@triton.jit
def initializeSimulation_triton_kernel(
    params_trade_fslImmed,
    params_trade_fslClose,
    allocationRatio:  tl.constexpr,
    size_paramsBatch: tl.constexpr, 
    size_block:       tl.constexpr
    ):
    #[1]: Offsets
    offsets = tl.program_id(0) * size_block + tl.arange(0, size_block)

    #[2]: Mask
    mask = offsets < size_paramsBatch

    #[3]: Trade Parameters Load
    tp_fsl_immed = tl.load(pointer = params_trade_fslImmed + offsets, mask = mask)
    tp_fsl_close = tl.load(pointer = params_trade_fslClose + offsets, mask = mask)

    #[4]: Trade Simulation State
    balance_wallet    = tl.zeros(shape = [size_block], dtype = tl.float32) + 1.0
    balance_allocated = balance_wallet * allocationRatio
    balance_margin    = tl.zeros(shape = [size_block], dtype = tl.float32) + 1.0
    balance_ftIndex   = tl.full(shape = [size_block], value = -1, dtype = tl.int32)
    quantity          = tl.zeros(shape = [size_block,], dtype = tl.float32)
    entryPrice        = tl.zeros(shape = [size_block,], dtype = tl.float32)
    forceExited       = tl.zeros(shape = [size_block,], dtype = tl.float32)
    nTrades           = tl.zeros(shape = [size_block,], dtype = tl.float32)

    #[5]: Balance Trend
    bt_sum         = tl.zeros(shape = [size_block,], dtype = tl.float32)
    bt_sum_squared = tl.zeros(shape = [size_block,], dtype = tl.float32)
    bt_sum_xy      = tl.zeros(shape = [size_block,], dtype = tl.float32)

    #[6]: Return Initialized Contents
    return (offsets,
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
            )

@triton.jit
def processTrade_triton_kernel(
    #Process
    offsets, 
    mask, 
    size_dataLen,
    SEEKERMODE: tl.constexpr,
    #Constants
    leverage:        tl.constexpr,
    allocationRatio: tl.constexpr,
    tradingFee:      tl.constexpr,
    #Base Data
    loop_index,
    data_klines,
    data_klines_stride,
    tp_fsl_immed, 
    tp_fsl_close,
    params_trade_pslReentry: tl.constexpr,
    #State & Result Tensors
    rqpDir_this,
    rqpVal_this, 
    balance_wallet, 
    balance_allocated, 
    balance_margin,
    quantity, 
    entryPrice, 
    forceExited, 
    nTrades, 
    balance_ftIndex,
    bt_sum, 
    bt_sum_xy,
    bt_sum_squared, 
    balance_wallet_history, 
    balance_margin_history
    ):
    #[1]: Trade Simulation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #RQP Range Control
    rqpVal_this = tl.minimum(rqpVal_this,  1.0)
    rqpVal_this = tl.maximum(rqpVal_this, -1.0)

    #Prices
    kline_base_ptr_this = data_klines + (loop_index * data_klines_stride)
    kline_price_high  = tl.load(kline_base_ptr_this + KLINEINDEX_HIGHPRICE)
    kline_price_low   = tl.load(kline_base_ptr_this + KLINEINDEX_LOWPRICE)
    kline_price_close = tl.load(kline_base_ptr_this + KLINEINDEX_CLOSEPRICE)

    #Position Side & Has #qty_entry
    position_side = tl.where(0 < quantity,  1.0, 0.0)
    position_side = tl.where(quantity < 0, -1.0, position_side)
    position_has = (quantity != 0)

    #Exit Conditions
    price_act_FSLImmed = entryPrice * (1.0 - position_side*tp_fsl_immed)
    price_act_FSLClose = entryPrice * (1.0 - position_side*tp_fsl_close)
    price_liquidation  = entryPrice * (1.0 - position_side/leverage)

    price_worst = tl.where(0 < quantity, kline_price_low,  kline_price_close)
    price_worst = tl.where(quantity < 0, kline_price_high, price_worst)
    
    hit_liquidation = position_has & ((position_side*price_worst)       <= (position_side*price_liquidation))
    hit_fslImmed    = position_has & ((position_side*price_worst)       <= (position_side*price_act_FSLImmed))
    hit_fslClose    = position_has & ((position_side*kline_price_close) <= (position_side*price_act_FSLClose))

    #Exit Execution Price
    price_exit_execution = tl.where(hit_fslImmed,    price_act_FSLImmed, kline_price_close)
    price_exit_execution = tl.where(hit_liquidation, price_liquidation,  price_exit_execution)
    
    #Quantity Reduce
    balance_committed = tl.abs(quantity)  * entryPrice
    balance_toCommit  = balance_allocated * tl.abs(rqpVal_this)

    status_forceExit   = hit_liquidation | hit_fslImmed | hit_fslClose
    status_clear       = (position_side != rqpDir_this)
    status_partialExit = (balance_toCommit - balance_committed) < 0

    quantity_new = tl.where(position_has & status_partialExit, (balance_toCommit / entryPrice) * position_side, quantity)
    quantity_new = tl.where(status_forceExit | status_clear,   0.0,                                             quantity_new)

    quantity_delta = quantity_new - quantity
    profit         = quantity_delta * (entryPrice-price_exit_execution)
    fee            = tl.abs(quantity_delta) * price_exit_execution * tradingFee

    #Wallet Balance Post-Exit Update
    balance_wallet = balance_wallet + (profit - fee) * leverage
    balance_wallet = tl.maximum(balance_wallet, 0.0)

    #Allocated Balance Update
    balance_allocated = tl.where(quantity_new == 0.0, 
                                 balance_wallet*allocationRatio, 
                                 balance_allocated) 
    
    #Force Exit State Update
    if (params_trade_pslReentry == False):
        forceExited = tl.where(status_forceExit,           position_side, forceExited)
        forceExited = tl.where(forceExited != rqpDir_this, 0.0,           forceExited)

    #Quantity Increase
    balance_committed = tl.abs(quantity_new) * entryPrice
    balance_toCommit  = balance_allocated * tl.abs(rqpVal_this)
    balance_toCommit_entry = tl.maximum(balance_toCommit-balance_committed, 0.0)

    quantity_entry = tl.where(forceExited == 0.0,
                              (balance_toCommit_entry / kline_price_close)*tl.where(rqpVal_this < 0, -1.0, 1.0),
                              0.0)
    quantity_final = quantity_new + quantity_entry
    
    #Entry Price Update
    entryPrice_new = tl.where(quantity_final == 0.0, 
                              0.0, 
                              (tl.abs(quantity_new)*entryPrice + tl.abs(quantity_entry)*kline_price_close) / tl.maximum(tl.abs(quantity_final), 1e-6))
    
    #Wallet Balance Post-Entry Update
    fee = tl.abs(quantity_entry) * kline_price_close * tradingFee
    balance_wallet = balance_wallet - fee * leverage
    balance_wallet = tl.maximum(balance_wallet, 0.0)

    #Margin Balance
    balance_margin = balance_wallet + quantity_final * (kline_price_close - entryPrice_new) * leverage
    balance_margin = tl.maximum(balance_margin, 0.0)

    #Update State
    balance_ftIndex = tl.where((balance_ftIndex == -1) & (quantity_final != quantity), loop_index, balance_ftIndex)
    nTrades         = tl.where(quantity_final != quantity, nTrades+1, nTrades)
    quantity   = quantity_final
    entryPrice = entryPrice_new
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #[2]: Balance Trend Trackers ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    first_trade_occurred = (0 <= balance_ftIndex)
    bt_val_x = tl.where(first_trade_occurred, (loop_index-balance_ftIndex).to(tl.float32), 0.0)
    bt_val_y = tl.where(first_trade_occurred, tl.log(tl.maximum(balance_wallet, 1e-9)),   0.0)
    bt_sum         += bt_val_y
    bt_sum_xy      += bt_val_x*bt_val_y
    bt_sum_squared += bt_val_y*bt_val_y
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #[3]: History Record ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if not SEEKERMODE:
        off_write = offsets * size_dataLen + loop_index
        tl.store(pointer = balance_wallet_history  + off_write, value = balance_wallet, mask = mask)
        tl.store(pointer = balance_margin_history  + off_write, value = balance_margin, mask = mask)
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    return (
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
    )

@triton.jit
def evaluateBalanceTrend_triton_kernel(
    size_dataLen: tl.constexpr,
    offsets,
    mask,
    balance_wallet,
    nTrades,
    balance_ftIndex,
    bt_sum,
    bt_sum_xy,
    bt_sum_squared,
    balance_finals,
    balance_bestFit_intercepts,
    balance_bestFit_growthRates,
    balance_bestFit_volatilities,
    balance_ftIndexes,
    nTrades_rb
    ):
    bt_n      = (size_dataLen-balance_ftIndex).to(tl.float32)
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
    tl.store(pointer = balance_bestFit_intercepts   + offsets, value = bt_intercepts,   mask = mask)
    tl.store(pointer = balance_bestFit_growthRates  + offsets, value = bt_growthRate,   mask = mask)
    tl.store(pointer = balance_bestFit_volatilities + offsets, value = bt_volatility,   mask = mask)
    tl.store(pointer = balance_finals               + offsets, value = balance_wallet,  mask = mask)
    tl.store(pointer = balance_ftIndexes            + offsets, value = balance_ftIndex, mask = mask)
    tl.store(pointer = nTrades_rb                   + offsets, value = nTrades,         mask = mask)

