import triton
import triton.language as tl

MODEL = [{'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed <NECESSARY>
         {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close <NECESSARY>

         {'PRECISION': 4, 'LIMIT': (-1.0000,  1.0000)},   #Delta
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - SHORT
         {'PRECISION': 6, 'LIMIT': (0.000000, 1.000000)}, #Strength - LONG
        ]

@triton.jit
def getRQPValue(#Model Parameters <UNIQUE>
                mp_delta,
                mp_strength_S,
                mp_strength_L,
                #Model State Trackers <UNIQUE>
                st_pip_csf_prev,
                #Base Data <COMMON>
                data_pip_csf,
                rqpVal_prev):

    #Original Values
    st_pip_csf_prev_original = st_pip_csf_prev

    #Nan Check
    isNan = (data_pip_csf != data_pip_csf)
    data_pip_csf = tl.where(isNan, 0.0, data_pip_csf)

    #PIP CSF Cycle
    isShort_prev = st_pip_csf_prev < mp_delta
    isShort_now  = data_pip_csf    < mp_delta
    cycleReset   = (isShort_prev ^ isShort_now)

    #RQP Value Calculation
    #---Effective Params
    mp_strength_eff = tl.where(isShort_now, mp_strength_S, mp_strength_L)
    #---Inputs
    width = tl.where(isShort_now, mp_delta+1.0, 1.0-mp_delta)
    dist  = tl.abs(data_pip_csf-mp_delta)
    rqpVal_abs = tl.maximum(1-(dist/tl.maximum(width, 1e-9)), 0.0)*mp_strength_eff
    rqpVal_abs = tl.where(width == 0.0, 0.0, rqpVal_abs)
    #---Cyclic Minimum
    rqpVal_abs = tl.where(cycleReset, rqpVal_abs, tl.minimum(rqpVal_abs, tl.abs(rqpVal_prev)))
    #---Direction
    rqpVal = tl.where(isShort_now, -rqpVal_abs, rqpVal_abs)

    #States Update
    st_pip_csf_prev = data_pip_csf

    #Return RQP Value & States
    rqpVal_return               = tl.where(isNan, 0.0,                      rqpVal)
    st_pip_csf_prev_return      = tl.where(isNan, st_pip_csf_prev_original, st_pip_csf_prev)
    return rqpVal_return, st_pip_csf_prev_return