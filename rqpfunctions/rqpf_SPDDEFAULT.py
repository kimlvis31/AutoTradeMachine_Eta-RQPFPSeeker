import triton
import triton.language as tl

MODEL = [{'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed <NECESSARY>
         {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close <NECESSARY>

         {'PRECISION': 6, 'LIMIT': (-1.000000, 1.000000)}, #Delta    - SHORT
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Length   - SHORT
         {'PRECISION': 6, 'LIMIT': (-1.000000, 1.000000)}, #Delta    - LONG
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Strength - LONG
         {'PRECISION': 6, 'LIMIT': ( 0.000000, 1.000000)}, #Length   - LONG
        ]
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

    #Original Values
    st_pip_lst_prev_original = st_pip_lst_prev
    st_pip_lsp_prev_original = st_pip_lsp_prev

    #Nan Check
    isNan = (data_pip_lst != data_pip_lst)
    data_pip_lst = tl.where(isNan, -1.0, data_pip_lst)
    data_pip_lsp = tl.where(isNan, -1.0, data_pip_lsp)

    #PIP CSF Cycle
    isShort_prev = (st_pip_lst_prev == 1)
    isShort_now  = (data_pip_lst    == 1)
    cycleReset   = (isShort_prev ^ isShort_now)

    #RQP Value Calculation
    #---Effective Params
    mp_delta_eff    = tl.where(isShort_now, mp_delta_S,    mp_delta_L)
    mp_strength_eff = tl.where(isShort_now, mp_strength_S, mp_strength_L)
    mp_length_eff   = tl.where(isShort_now, mp_length_S,   mp_length_L)
    #---RQP Value
    pd = tl.where(isShort_now, 1-data_price_close/data_pip_lsp, data_price_close/data_pip_lsp-1)
    rqpVal_abs = tl.where(mp_delta_eff <= pd,
                          tl.maximum((1-(pd-mp_delta_eff)/tl.maximum(mp_length_eff, 1e-9))*mp_strength_eff, 0.0),
                          0.0)
    rqpVal_abs = tl.where(mp_length_eff == 0.0, 0.0, rqpVal_abs)
    rqpVal_abs = tl.where(cycleReset, rqpVal_abs, tl.minimum(rqpVal_abs, tl.abs(rqpVal_prev)))
    rqpVal = tl.where(isShort_now, -rqpVal_abs, rqpVal_abs)

    #States Update
    st_pip_lst_prev = data_pip_lst
    st_pip_lsp_prev = data_pip_lsp

    #Return RQP Value & States
    rqpVal_return          = tl.where(isNan, 0.0,                      rqpVal)
    st_pip_lst_prev_return = tl.where(isNan, st_pip_lst_prev_original, st_pip_lst_prev)
    st_pip_lsp_prev_return = tl.where(isNan, st_pip_lsp_prev_original, st_pip_lsp_prev)
    return rqpVal_return, st_pip_lst_prev_return, st_pip_lsp_prev_return