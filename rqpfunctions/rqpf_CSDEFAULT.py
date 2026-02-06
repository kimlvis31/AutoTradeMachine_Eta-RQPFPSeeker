import triton
import triton.language as tl

"""
FUNCTION MODEL: CSDEFAULT (Classical Signal Default)
 * The first two parameters are required by the system, and must always be included in the format as they are.
"""
MODEL = [{'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed <NECESSARY>
         {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close <NECESSARY>

         {'PRECISION': 4, 'LIMIT': (-1.0000,  1.0000)},   #Delta
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - LONG
        ]

INPUTDATAKEYS = ['MMACDLONG_MSDELTAABSMAREL',]

"""
<Triton Kernel Function>
 * This is an RQP value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate RQP value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
@triton.jit
def initializeMST(blockSize: tl.constexpr):
    mst0 = tl.full([blockSize,], -1.0, dtype=tl.float32)
    mst1 = tl.full([blockSize,], -1.0, dtype=tl.float32)
    mst2 = tl.full([blockSize,], -1.0, dtype=tl.float32)

    mst = [mst0, mst1, mst2]
    return mst
@triton.jit
def getRQPValue(kline_base_ptr, analysis_base_ptr, mp, mst, blockSize):
    
    return tl.zeros(shape = [blockSize,], dtype = tl.float32), mst

    #[1]: Record Original Values
    st_pip_csf_prev_original = st_pip_csf_prev

    #[2]: Nan Check
    isNan = (data_pip_csf != data_pip_csf)
    data_pip_csf = tl.where(isNan, 0.0, data_pip_csf)

    #[3]: PIP CSF Cycle
    isShort_prev = st_pip_csf_prev < mp_delta
    isShort_now  = data_pip_csf    < mp_delta
    cycleReset   = (isShort_prev ^ isShort_now)

    #[4]: RQP Value Calculation
    #---[4-1]: Effective Params
    mp_strength_eff = tl.where(isShort_now, mp_strength_S, mp_strength_L)
    #---[4-2]: RQP Value
    width = tl.where(isShort_now, mp_delta+1.0, 1.0-mp_delta)
    dist  = tl.abs(data_pip_csf-mp_delta)
    rqpVal_abs = tl.maximum(1-(dist/tl.maximum(width, 1e-9)), 0.0)*mp_strength_eff
    rqpVal_abs = tl.where(width == 0.0, 0.0, rqpVal_abs)
    #---[4-3]: Cyclic Minimum
    rqpVal_abs = tl.where(cycleReset, rqpVal_abs, tl.minimum(rqpVal_abs, tl.abs(rqpVal_prev)))
    #---[4-4]: Direction
    rqpVal = tl.where(isShort_now, -rqpVal_abs, rqpVal_abs)

    #[5]: States Update
    st_pip_csf_prev = data_pip_csf

    #[6]: Return RQP Value & States
    rqpVal_return               = tl.where(isNan, 0.0,                      rqpVal)
    st_pip_csf_prev_return      = tl.where(isNan, st_pip_csf_prev_original, st_pip_csf_prev)
    return rqpVal_return, st_pip_csf_prev_return