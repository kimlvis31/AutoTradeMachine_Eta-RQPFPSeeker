import triton
import triton.language as tl

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

"""
<Triton Kernel Function>
 * This is an RQP value calculation function written in Triton.
 * It simply takes in model parameters, model state trackers, and base data, and calculate RQP value for trading simulation in the base Triton Kernel Function.
 * This is an example and is recommended to be kept without edits for reference. The user may add similar .py files following the general structure in this file to test their customized strategies. In order for the trade simulator function to be able to 
   recognize and call this function, the user must implement the model parameter import, state trackers initialization, and function call parts for the new specific model. Check 'processBatch_triton_kernel' function in 'exitFunction_base.py'
"""
@triton.jit
def getRQPValue(#Model Parameters <UNIQUE>
                mp_delta_S,
                mp_strength_S,
                mp_length_S,
                mp_delta_L,
                mp_strength_L,
                mp_length_L,
                #Model State Trackers <UNIQUE>
                st_pip_lst_prev,
                st_pip_lsp_prev,
                #Base Data <COMMON>
                data_price_close,
                data_pip_lst,
                data_pip_lsp,
                rqpVal_prev):
    #[1]: Original Values
    st_pip_lst_prev_original = st_pip_lst_prev
    st_pip_lsp_prev_original = st_pip_lsp_prev

    #[2]: Nan Check
    isNan = (data_pip_lst != data_pip_lst)
    data_pip_lst = tl.where(isNan, -1.0, data_pip_lst)
    data_pip_lsp = tl.where(isNan, -1.0, data_pip_lsp)

    #[3]: PIP Swing Cycle
    isShort_prev = (st_pip_lst_prev == 1.0)
    isShort_now  = (data_pip_lst    == 1.0)
    cycleReset   = (isShort_prev ^ isShort_now)

    #[4]: RQP Value Calculation
    #---[4-1]: Effective Params
    mp_delta_eff    = tl.where(isShort_now, mp_delta_S,    mp_delta_L)
    mp_strength_eff = tl.where(isShort_now, mp_strength_S, mp_strength_L)
    mp_length_eff   = tl.where(isShort_now, mp_length_S,   mp_length_L)
    #---[4-2]: RQP Value
    pd   = tl.where(isShort_now, 1-data_price_close/data_pip_lsp, data_price_close/data_pip_lsp-1)
    dist = pd-mp_delta_eff
    rqpVal_abs = tl.where(mp_delta_eff <= pd,
                          tl.maximum((1-dist/tl.maximum(mp_length_eff, 1e-6))*mp_strength_eff, 0.0),
                          0.0)
    rqpVal_abs = tl.where(mp_length_eff == 0.0, 0.0, rqpVal_abs)
    #---[4-3]: Cyclic Minimum
    rqpVal_abs = tl.where(cycleReset, rqpVal_abs, tl.minimum(rqpVal_abs, tl.abs(rqpVal_prev)))
    #---[4-4]: Direction
    rqpVal = tl.where(isShort_now, -rqpVal_abs, rqpVal_abs)

    #[5]: States Update
    st_pip_lst_prev = data_pip_lst
    st_pip_lsp_prev = data_pip_lsp

    #[6]: Return RQP Value & States
    rqpVal_return          = tl.where(isNan, 0.0,                      rqpVal)
    st_pip_lst_prev_return = tl.where(isNan, st_pip_lst_prev_original, st_pip_lst_prev)
    st_pip_lsp_prev_return = tl.where(isNan, st_pip_lsp_prev_original, st_pip_lsp_prev)
    return rqpVal_return, st_pip_lst_prev_return, st_pip_lsp_prev_return