import numpy
import torch
import termcolor
import sys
import math
import time
from functools import wraps

from exitFunction_models import RQPFUNCTIONS_MODEL, RQPFUNCTIONS_INPUTDATAKEY, RQPFUNCTIONS_BATCHPROCESSFUNCTION
import rqpfunctions.simulatorFunctions as sf

ALLOCATIONRATIO = 0.90
TRADINGFEE      = 0.0005

BPST_KVALUE        = 2/(100+1)
BPST_PRINTINTERVAL = 100e6

_TORCHDTYPE = torch.float32

KLINEINDEX_OPENTIME        = sf.KLINEINDEX_OPENTIME
KLINEINDEX_OPENPRICE       = sf.KLINEINDEX_OPENPRICE
KLINEINDEX_HIGHPRICE       = sf.KLINEINDEX_HIGHPRICE
KLINEINDEX_LOWPRICE        = sf.KLINEINDEX_LOWPRICE
KLINEINDEX_CLOSEPRICE      = sf.KLINEINDEX_CLOSEPRICE
KLINEINDEX_VOLBASE         = sf.KLINEINDEX_VOLBASE
KLINEINDEX_VOLBASETAKERBUY = sf.KLINEINDEX_VOLBASETAKERBUY

TRADEPARAMS = [{'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Immed <NECESSARY>
               {'PRECISION': 4, 'LIMIT': (0.0000, 1.0000)}, #FSL Close <NECESSARY>
               ]

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

#Exit Function Model ====================================================================================================================================================================================================================================
class exitFunction():
    def __init__(self, modelName, isSeeker, leverage, pslReentry):
        self.MODELNAME                 = modelName
        self.model                     = RQPFUNCTIONS_MODEL[self.MODELNAME]
        self.inputDataKeys             = RQPFUNCTIONS_INPUTDATAKEY[self.MODELNAME]
        self.modelBatchProcessFunction = RQPFUNCTIONS_BATCHPROCESSFUNCTION[self.MODELNAME]
        self.isSeeker           = isSeeker
        self.leverage           = leverage
        self.pslReentry         = pslReentry
        self.parameterBatchSize = 32

        #Data Set
        self.__data_klines   = None
        self.__data_analysis = None

        #Seeker
        self.__seeker = None

    def preprocessData(self, linearizedAnalysis: numpy.ndarray, indexIdentifier: dict) -> None:
        #[1]: Tensor Conversion
        linearizedAnalysis = torch.from_numpy(linearizedAnalysis).to(device='cuda', dtype=torch.float64)
        dataLen = linearizedAnalysis.size(dim = 0)

        #[2]: Klines Tensor
        #---[2-1]: Klines Data Initialization & Data Load
        data_klines = torch.full(size = (dataLen, 7), fill_value = torch.nan, device='cuda', dtype=torch.float64)
        for lIndex, klKey in ((KLINEINDEX_OPENTIME,        'KLINE_OPENTIME'), 
                              (KLINEINDEX_OPENPRICE,       'KLINE_OPENPRICE'), 
                              (KLINEINDEX_HIGHPRICE,       'KLINE_HIGHPRICE'), 
                              (KLINEINDEX_LOWPRICE,        'KLINE_LOWPRICE'), 
                              (KLINEINDEX_CLOSEPRICE,      'KLINE_CLOSEPRICE'), 
                              (KLINEINDEX_VOLBASE,         'KLINE_VOLBASE'), 
                              (KLINEINDEX_VOLBASETAKERBUY, 'KLINE_VOLBASETAKERBUY')):
            if klKey not in indexIdentifier:
                print(termcolor.colored(f"      - [WARNING] Kline Data Key '{klKey}' Not Found In The Index Identifier. The Corresponding Data Will Stay Empty", 'light_red'))
                continue
            aIndex = indexIdentifier[klKey]
            data_klines[:,lIndex] = linearizedAnalysis[:,aIndex]
        #---[2-2]: Klines Data Normalization
        closePrice_initial = data_klines[0,KLINEINDEX_CLOSEPRICE].item()
        nonzero_indices = (data_klines[:,KLINEINDEX_VOLBASE] != 0).nonzero()
        if 0 < nonzero_indices.size(0):
            first_nonzero_idx       = nonzero_indices[0, 0]
            baseAssetVolume_initial = data_klines[first_nonzero_idx, KLINEINDEX_VOLBASE]
            if first_nonzero_idx != 0: print(termcolor.colored(f"      - [WARNING] None Zero-Index Volume Used During The Volume Normalization. Used Index: {first_nonzero_idx.item()}", 'light_red'))
        else:
            baseAssetVolume_initial = 1.0
            print(termcolor.colored("      - [WARNING] No Non-Zero Volume Found During The Volume Normalization. Assuming The Initial Value Of 1.0", 'light_red'))
        data_klines[:,KLINEINDEX_OPENPRICE]  = data_klines[:,KLINEINDEX_OPENPRICE] /closePrice_initial
        data_klines[:,KLINEINDEX_HIGHPRICE]  = data_klines[:,KLINEINDEX_HIGHPRICE] /closePrice_initial
        data_klines[:,KLINEINDEX_LOWPRICE]   = data_klines[:,KLINEINDEX_LOWPRICE]  /closePrice_initial
        data_klines[:,KLINEINDEX_CLOSEPRICE] = data_klines[:,KLINEINDEX_CLOSEPRICE]/closePrice_initial
        data_klines[:,KLINEINDEX_VOLBASE]         = data_klines[:,KLINEINDEX_VOLBASE]        /baseAssetVolume_initial
        data_klines[:,KLINEINDEX_VOLBASETAKERBUY] = data_klines[:,KLINEINDEX_VOLBASETAKERBUY]/baseAssetVolume_initial
        #---[2-3]: Update Klines Data
        self.__data_klines = data_klines.to(dtype=_TORCHDTYPE).contiguous()

        #[3]: Analysis Tensor
        #---[2-1]: Analysis Data Load
        data_analysis = torch.full(size = (dataLen, len(self.inputDataKeys)), fill_value = torch.nan, device='cuda', dtype=torch.float64)
        for lIndex, laKey in enumerate(self.inputDataKeys):
            #[2-1-1]: Linearized Analysis Key Check
            if laKey not in indexIdentifier:
                print(termcolor.colored(f"      - [WARNING] Linearized Analysis Key '{laKey}' Not Found In The Index Identifier. The Corresponding Data Will Stay Empty", 'light_red'))
                continue
            aIndex = indexIdentifier[laKey]
            #[2-1-2]: Analysis Data Preprocessing (Recognition & Normalization) & Update
            data_analysis[:,lIndex] = self.__preprocessAnalysisData(linearizedAnalysisKey   = laKey,
                                                                    analysisDataLine        = linearizedAnalysis[:,aIndex],
                                                                    closePrice_initial      = closePrice_initial,
                                                                    baseAssetVolume_initial = baseAssetVolume_initial)
        #---[2-2]: Save Contiguous Analysis Data
        self.__data_analysis = data_analysis.to(dtype=_TORCHDTYPE).contiguous()

    def __preprocessAnalysisData(self, linearizedAnalysisKey, analysisDataLine, closePrice_initial, baseAssetVolume_initial):
        #[1]: Linearized Analysis Key Check
        laKeySplit    = linearizedAnalysisKey.split("_")
        laKeySplitLen = len(laKeySplit)
        if laKeySplitLen == 2: 
            aType, valType = laKeySplit
            aLine = None
        elif laKeySplitLen == 3: 
            aType, aLine, valType = laKeySplit
        else:
            print(termcolor.colored(f"      - [WARNING] Unexpected Format Of Linearized Analysis Key Detected - '{linearizedAnalysisKey}'. User Attention Strongly Advised", 'light_red'))
            return analysisDataLine
        
        #[2]: Type-Dependent Normalization
        aType_unrecognized   = False
        valType_unrecognized = False
        adl_pp = analysisDataLine

        #---[2-1]: Analysis Type - SMA
        if aType == 'SMA':
            if valType == 'SMA':
                adl_pp = analysisDataLine/closePrice_initial
            else: 
                valType_unrecognized = True



        #---[2-2]: Analysis Type - WMA
        elif aType == 'WMA':
            if valType == 'WMA':
                adl_pp = analysisDataLine/closePrice_initial
            else: 
                valType_unrecognized = True



        #---[2-3]: Analysis Type - EMA
        elif aType == 'EMA':
            if valType == 'EMA':
                adl_pp = analysisDataLine/closePrice_initial
            else: 
                valType_unrecognized = True



        #---[2-4]: Analysis Type - PSAR
        elif aType == 'PSAR':
            if valType == 'PSAR':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'DCC':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-5]: Analysis Type - BOL
        elif aType == 'BOL':
            if valType == 'BOLLOW':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'BOLHIGH':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'MA':
                adl_pp = analysisDataLine/closePrice_initial
            else: 
                valType_unrecognized = True



        #---[2-6]: Analysis Type - IVP
        elif aType == 'IVP':
            if valType.startswith("NB") and valType[2:].isdigit():
                adl_pp = analysisDataLine/closePrice_initial
            else: 
                valType_unrecognized = True



        #---[2-7]: Analysis Type - SWING
        elif aType == 'SWING':
            if valType == 'LSTIMESTAMP':
                adl_pp = analysisDataLine
            elif valType == 'LSPRICE':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'LSTYPE':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-8]: Analysis Type - VOL
        elif aType == 'VOL':
            if valType == 'MABASE':
                adl_pp = analysisDataLine/baseAssetVolume_initial
            elif valType == 'MABASETB':
                adl_pp = analysisDataLine/baseAssetVolume_initial
            else: 
                valType_unrecognized = True



        #---[2-9]: Analysis Type - NNA
        elif aType == 'NNA':
            if valType == 'NNA':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-10]: Analysis Type - MMACDSHORT
        elif aType == 'MMACDSHORT':
            if valType == 'MSDELTA':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'MSDELTAABSMA':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'MSDELTAABSMAREL':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-11]: Analysis Type - MMACDLONG
        elif aType == 'MMACDLONG':
            if valType == 'MSDELTA':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'MSDELTAABSMA':
                adl_pp = analysisDataLine/closePrice_initial
            elif valType == 'MSDELTAABSMAREL':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-12]: Analysis Type - DMIxADX
        elif aType == 'DMIxADX':
            if valType == 'ABSATHREL':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-13]: Analysis Type - MFI
        elif aType == 'MFI':
            if valType == 'ABSATHREL':
                adl_pp = analysisDataLine
            else: 
                valType_unrecognized = True



        #---[2-14]: Unrecognizable Analysis Type
        else:
            aType_unrecognized = True



        #[3]: Result Interpretation
        if aType_unrecognized:   print(termcolor.colored(f"      - [WARNING] Unrecognized Analysis Type Detected During Linearized Analysis Data Normalization - '{aType}'. User Attention Strongly Advised", 'light_red'))
        if valType_unrecognized: print(termcolor.colored(f"      - [WARNING] Unrecognized Value Type Detected During Linearized Analysis Data Normalization - '{valType}'. User Attention Strongly Advised", 'light_red'))
        return adl_pp

    def initializeSeeker(self,
                         tradeParamConfig:         list,
                         modelParamConfig:         list, 
                         nSeekerPoints:            int, 
                         parameterBatchSize:       int, 
                         nRepetition:              int,
                         learningRate:             float, 
                         deltaRatio:               float,
                         beta_velocity:            float,
                         beta_momentum:            float,
                         repopulationRatio:        float, 
                         repopulationInterval:     int,
                         repopulationGuideRatio:   float,
                         repopulationDecayRate:    float,
                         scoring:                  str,
                         scoring_maxMDD:           int | float,
                         scoring_growthRateWeight: int | float,
                         scoring_growthRateScaler: int | float,
                         scoring_volatilityWeight: int | float,
                         scoring_volatilityScaler: int | float,
                         scoring_nTradesWeight:    int | float,
                         scoring_nTradesScaler:    int | float,
                         scoringSamples:           int,
                         terminationThreshold:     float) -> None:

        #[1]: Seeker Parameters Check
        #---[1-1]: tradeParamConfig
        if type(tradeParamConfig) not in (list, tuple): tradeParamConfig = [None,]*nParams_trade
        nParams_trade       = len(TRADEPARAMS)
        nParams_tradeConfig = len(tradeParamConfig)
        if nParams_tradeConfig < nParams_trade: 
            tradeParamConfig += [None,]*(nParams_trade-nParams_tradeConfig)
        elif (nParams_trade < nParams_tradeConfig):
            tradeParamConfig = tradeParamConfig[:nParams_trade]
        tradeParamConfig = tuple(tradeParamConfig)
        #---[1-2]: modelParamConfig
        if type(modelParamConfig) not in (list, tuple): modelParamConfig = [None,]*nParams_model
        nParams_model       = len(self.model)
        nParams_modelConfig = len(modelParamConfig)
        if nParams_modelConfig < nParams_model: 
            modelParamConfig += [None,]*(nParams_model-nParams_modelConfig)
        elif (nParams_model < nParams_modelConfig):
            modelParamConfig = modelParamConfig[:nParams_model]
        modelParamConfig = tuple(modelParamConfig)
        #---[1-3]: nSeekerPoints
        if type(nSeekerPoints) is not int: nSeekerPoints = 10
        if not (1 <= nSeekerPoints):       nSeekerPoints = 10
        #---[1-4]: parameterBatchSize
        nParamsSet_max     = ((len(TRADEPARAMS)+len(self.model))*2*nSeekerPoints)
        paramBatchSize_max = math.ceil(nParamsSet_max/32)*32
        if type(parameterBatchSize) is not int:            parameterBatchSize = paramBatchSize_max
        if not (1 <= parameterBatchSize):                  parameterBatchSize = paramBatchSize_max
        if not (parameterBatchSize <= paramBatchSize_max): parameterBatchSize = paramBatchSize_max
        if not (parameterBatchSize%32 == 0):               parameterBatchSize = math.ceil(parameterBatchSize/32)*32
        #---[1-5]: nRepetition
        if type(nRepetition) is not int: nRepetition = 10
        if not (1 <= nRepetition):       nRepetition = 10
        #---[1-6]: learningRate
        if type(learningRate) not in (float, int): learningRate = 0.001
        if not (0.0 < learningRate <= 1.0):        learningRate = 0.001
        #---[1-7]: deltaRatio
        if type(deltaRatio) not in (float, int): deltaRatio = 0.1
        if not (0.0 < deltaRatio < 1.0):         deltaRatio = 0.1
        #---[1-8]: beta_velocity
        if type(beta_velocity) not in (float, int): beta_velocity = 0.9
        if not (0.0 <= beta_velocity < 1.0):        beta_velocity = 0.9
        #---[1-9]: beta_momentum
        if type(beta_momentum) not in (float, int): beta_momentum = 0.99
        if not (0.0 <= beta_momentum < 1.0):         beta_momentum = 0.99
        #---[1-10]: repopulationRatio
        if type(repopulationRatio) not in (float, int): repopulationRatio = 0.1
        if not (0.0 <= repopulationRatio <= 1.0):       repopulationRatio = 0.1
        #---[1-11]: repopulationInterval
        if type(repopulationInterval) is not int: repopulationInterval = 10
        if not (1 <= repopulationInterval):       repopulationInterval = 10
        #---[1-12]: repopulationGuideRatio
        if type(repopulationGuideRatio) not in (float, int): repopulationGuideRatio = 0.5
        if not (0.0 <= repopulationGuideRatio <= 1.0):       repopulationGuideRatio = 0.5
        #---[1-13]: repopulationDecayRate
        if type(repopulationDecayRate) not in (float, int): repopulationDecayRate = 0.1
        if not (0.0 <  repopulationDecayRate <= 1.0):       repopulationDecayRate = 0.1
        #---[1-14]: scoring
        if scoring not in ('FINALBALANCE', 'GROWTHRATE', 'VOLATILITY', 'SHARPERATIO'): scoring = 'FINALBALANCE'
        #---[1-15]: scoring_maxMDD
        if type(scoring_maxMDD) not in (float, int): scoring_maxMDD = 1.0
        if not (0.0 <= scoring_maxMDD <= 1.0):       scoring_maxMDD = 1.0
        #---[1-16]: scoring_growthRateWeight
        if type(scoring_growthRateWeight) not in (float, int): scoring_growthRateWeight = 1.0
        if not (0.0 <= scoring_growthRateWeight):              scoring_growthRateWeight = 1.0
        #---[1-17]: scoring_growthRateScaler
        if type(scoring_growthRateScaler) not in (float, int): scoring_growthRateScaler = 1e5
        if not (0.0 <= scoring_growthRateScaler):              scoring_growthRateScaler = 1e5
        #---[1-18]: scoring_volatilityWeight
        if type(scoring_volatilityWeight) not in (float, int): scoring_volatilityWeight = 1.0
        if not (0.0 <= scoring_volatilityWeight):              scoring_volatilityWeight = 1.0
        #---[1-19]: scoring_volatilityScaler
        if type(scoring_volatilityScaler) not in (float, int): scoring_volatilityScaler = 0.1
        if not (0.0 <= scoring_volatilityScaler):              scoring_volatilityScaler = 0.1
        #---[1-20]: scoring_nTradesWeight
        if type(scoring_nTradesWeight) not in (float, int): scoring_nTradesWeight = 1.0
        if not (0.0 <= scoring_nTradesWeight):              scoring_nTradesWeight = 1.0
        #---[1-21]: scoring_nTradesScaler
        if type(scoring_nTradesScaler) not in (float, int): scoring_nTradesScaler = 0.0001
        if not (0.0 <= scoring_nTradesScaler):              scoring_nTradesScaler = 0.0001
        #---[1-22]: scoringSamples
        if type(scoringSamples) is not int: scoringSamples = 20
        if not (1 <= scoringSamples):       scoringSamples = 20
        #---[1-23]: terminationThreshold
        if type(terminationThreshold) not in (float, int): terminationThreshold = 0.0001
        if not (0.0 <= terminationThreshold <= 1.0):       terminationThreshold = 0.0001

        #[2]: Trade & Model Parameters
        paramDescriptions_full = TRADEPARAMS+self.model
        paramConfig_full       = tradeParamConfig+modelParamConfig
        nParams                = len(paramConfig_full)

        #[3]: Rounding Tensor
        params_rounding_factors = 10.0 ** torch.tensor([pDesc['PRECISION'] for pDesc in paramDescriptions_full], device='cuda', dtype=_TORCHDTYPE).unsqueeze(0)

        #[4]: Parameter Configuration Fixed Value Mask Generation
        params_fixed_mask   = torch.zeros(nParams, dtype=torch.bool,    device='cuda')
        params_fixed_values = torch.zeros(nParams, dtype=torch.float32, device='cuda')
        for pIndex, val in enumerate(paramConfig_full):
            if val is None: continue
            params_fixed_mask[pIndex]   = True
            params_fixed_values[pIndex] = val
        params_fixed_values = (torch.round(params_fixed_values * params_rounding_factors) / params_rounding_factors).squeeze(0)

        #[5]: Parameter Range Tensors
        params_min = torch.tensor([[pDesc['LIMIT'][0] for pDesc in paramDescriptions_full]], device='cuda', dtype = _TORCHDTYPE)
        params_max = torch.tensor([[pDesc['LIMIT'][1] for pDesc in paramDescriptions_full]], device='cuda', dtype = _TORCHDTYPE)
        params_min = torch.round(params_min * params_rounding_factors) / params_rounding_factors
        params_max = torch.round(params_max * params_rounding_factors) / params_rounding_factors

        #[6]: Base Tensors
        params_base = torch.rand(size = (nSeekerPoints, nParams), device = 'cuda', dtype = _TORCHDTYPE)
        velocity    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)
        momentum    = torch.zeros_like(params_base, device = 'cuda', dtype = _TORCHDTYPE)

        params_base = params_base * (params_max - params_min) + params_min                         #Range Mapping
        params_base = torch.round(params_base * params_rounding_factors) / params_rounding_factors #Rounding
        params_base[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]                 #Fixed Parameters Overwrite

        #[7]: Seeker Update
        self.__seeker = {#Seeker Parameters
                         'tradeParamConfig':         tradeParamConfig,
                         'modelParamConfig':         modelParamConfig,
                         'parameterBatchSize':       parameterBatchSize,
                         'nSeekerPoints':            nSeekerPoints,
                         'nRepetition':              nRepetition,
                         'learningRate':             learningRate,
                         'deltaRatio':               deltaRatio,
                         'beta_velocity':            beta_velocity,
                         'beta_momentum':            beta_momentum,
                         'repopulationRatio':        repopulationRatio,
                         'repopulationInterval':     repopulationInterval,
                         'repopulationGuideRatio':   repopulationGuideRatio,
                         'repopulationDecayRate':    repopulationDecayRate,
                         'scoring':                  scoring,
                         'scoring_maxMDD':           scoring_maxMDD,
                         'scoring_growthRateWeight': scoring_growthRateWeight,
                         'scoring_growthRateScaler': scoring_growthRateScaler,
                         'scoring_volatilityWeight': scoring_volatilityWeight,
                         'scoring_volatilityScaler': scoring_volatilityScaler,
                         'scoring_nTradesWeight':    scoring_nTradesWeight,
                         'scoring_nTradesScaler':    scoring_nTradesScaler,
                         'scoringSamples':           scoringSamples,
                         'terminationThreshold':     terminationThreshold,
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
        self.parameterBatchSize = self.__seeker['parameterBatchSize']
        
        #[8]: Applied Seeker Parameters
        asp = dict()
        asp['tradeParamConfig']         = self.__seeker['tradeParamConfig']
        asp['modelParamConfig']         = self.__seeker['modelParamConfig']
        asp['parameterBatchSize']       = self.__seeker['parameterBatchSize']
        asp['nSeekerPoints']            = self.__seeker['nSeekerPoints']
        asp['nRepetition']              = self.__seeker['nRepetition']
        asp['learningRate']             = self.__seeker['learningRate']
        asp['deltaRatio']               = self.__seeker['deltaRatio']
        asp['beta_velocity']            = self.__seeker['beta_velocity']
        asp['beta_momentum']            = self.__seeker['beta_momentum']
        asp['repopulationRatio']        = self.__seeker['repopulationRatio']
        asp['repopulationInterval']     = self.__seeker['repopulationInterval']
        asp['repopulationGuideRatio']   = self.__seeker['repopulationGuideRatio']
        asp['repopulationDecayRate']    = self.__seeker['repopulationDecayRate']
        asp['scoring']                  = self.__seeker['scoring']
        asp['scoring_maxMDD']           = self.__seeker['scoring_maxMDD']
        asp['scoring_growthRateWeight'] = self.__seeker['scoring_growthRateWeight']
        asp['scoring_growthRateScaler'] = self.__seeker['scoring_growthRateScaler']
        asp['scoring_volatilityWeight'] = self.__seeker['scoring_volatilityWeight']
        asp['scoring_volatilityScaler'] = self.__seeker['scoring_volatilityScaler']
        asp['scoring_nTradesWeight']    = self.__seeker['scoring_nTradesWeight']
        asp['scoring_nTradesScaler']    = self.__seeker['scoring_nTradesScaler']
        asp['scoringSamples']           = self.__seeker['scoringSamples']
        asp['terminationThreshold']     = self.__seeker['terminationThreshold']

        return asp
    
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
    
    def __scoreResults(self, balance_finals, balance_bestFit_growthRates, balance_bestFit_volatilities, nTrades):
        scoring                  = self.__seeker['scoring']
        scoring_maxMDD           = self.__seeker['scoring_maxMDD']
        scoring_growthRateWeight = self.__seeker['scoring_growthRateWeight']
        scoring_growthRateScaler = self.__seeker['scoring_growthRateScaler']
        scoring_volatilityWeight = self.__seeker['scoring_volatilityWeight']
        scoring_volatilityScaler = self.__seeker['scoring_volatilityScaler']
        scoring_nTradesWeight    = self.__seeker['scoring_nTradesWeight']
        scoring_nTradesScaler    = self.__seeker['scoring_nTradesScaler']

        #[1]: TYPE - 'FINALBALANCE'
        if (scoring == 'FINALBALANCE'):
            scores = 1-torch.exp(-balance_finals)

        #[2]: TYPE - 'GROWTHRATE'
        elif (scoring == 'GROWTHRATE'):
            gr_scaled = balance_bestFit_growthRates * scoring_growthRateScaler
            scores = torch.where(balance_bestFit_growthRates < 0,
                                 1/(1-gr_scaled),
                                 1+gr_scaled)

        #[3]: TYPE - 'VOLATILITY'
        elif (scoring == 'VOLATILITY'):
            vol_scaled     = balance_bestFit_volatilities * scoring_volatilityScaler
            nTrades_scaled = nTrades                      * scoring_nTradesScaler
            scores_volatility = torch.exp(-vol_scaled)         ** scoring_volatilityWeight
            scores_nTrades    = (1-torch.exp(-nTrades_scaled)) ** scoring_nTradesWeight
            scores = scores_volatility * scores_nTrades

        #[4]: TYPE - 'SHARPERATIO'
        elif (scoring == 'SHARPERATIO'):
            gr_scaled      = balance_bestFit_growthRates  * scoring_growthRateScaler
            vol_scaled     = balance_bestFit_volatilities * scoring_volatilityScaler
            nTrades_scaled = nTrades                      * scoring_nTradesScaler
            scores_gr = torch.where(balance_bestFit_growthRates < 0,
                                    1/(1-gr_scaled),
                                    1+gr_scaled
                                    ) ** scoring_growthRateWeight
            scores_volatility = torch.exp(-vol_scaled)         ** scoring_volatilityWeight
            scores_nTrades    = (1-torch.exp(-nTrades_scaled)) ** scoring_nTradesWeight
            scores = scores_gr * scores_volatility * scores_nTrades
            
        #[5]: Maximum Drawdown Filtering
        volatility_tMin_997 = torch.exp(-balance_bestFit_volatilities*3)-1
        scores = torch.where(scoring_maxMDD < -volatility_tMin_997, 0.0, scores)

        #[6]: Finally
        return scores

    def runSeeker(self) -> tuple[bool, tuple]:
        #[1]: Timer
        t_cpu_beg_ns = time.perf_counter_ns()

        #[2]: Parameters
        seeker = self.__seeker
        nSeekerPoints = seeker['nSeekerPoints']
        nParameters   = len(seeker['tradeParamConfig'])+len(seeker['modelParamConfig'])
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

        #[3]: Get Test Parameters & Split into batches
        params_test, params_plus, params_minus = self.__getTestParams()
        params_test_batches = torch.split(params_test, self.parameterBatchSize)

        #[4]: Process Batches
        bestResults = []
        scores      = []
        t_elapsed_gpu_simulation_total_ms = 0
        for params_test_batch in params_test_batches:
            #[4-1]: Batch Processing
            balances, t_elapsed_gpu_ms = self.__performOnParams_Timed(params = params_test_batch)
            (balance_finals, 
             balance_bestFit_growthRates, 
             balance_bestFit_volatilities,
             nTrades) = balances
            t_elapsed_gpu_simulation_total_ms += t_elapsed_gpu_ms
            
            #[4-2]: Scoring
            scores_batch = self.__scoreResults(balance_finals               = balance_finals,
                                               balance_bestFit_growthRates  = balance_bestFit_growthRates,
                                               balance_bestFit_volatilities = balance_bestFit_volatilities,
                                               nTrades                      = nTrades)
            scores.append(scores_batch)
            
            #[4-3]: Best Result Record
            _, max_idx = torch.max(scores_batch, dim = 0)
            max_idx = max_idx.item()
            bestParams = params_test_batch[max_idx].detach().cpu().numpy().tolist()
            bestParams = tuple(round(bestParams[pIndex], pDesc['PRECISION']) for pIndex, pDesc in enumerate(TRADEPARAMS+self.model))
            bestParams_trade = bestParams[:len(TRADEPARAMS)]
            bestParams_model = bestParams[len(TRADEPARAMS):]
            bestResult = (bestParams_trade,                                        #Trade Parameters
                          bestParams_model,                                        #Model Parameters
                          round(float(balance_finals[max_idx]),               12), #Final Wallet Balance
                          round(float(balance_bestFit_growthRates[max_idx]),  12), #Growth Rate
                          round(float(balance_bestFit_volatilities[max_idx]), 12), #Volatility
                          round(float(scores_batch[max_idx]),                 12), #Score
                          round(int(nTrades[max_idx])))                            #nTrades
            bestResults.append(bestResult)
        t_processing_sim_paramsSet_ms = t_elapsed_gpu_simulation_total_ms/len(params_test)

        #[5]: Best Reulst Record
        bestResult          = max(bestResults, key=lambda x: x[5])
        bestResults         = seeker['_bestResults']
        bestResults_thisRep = bestResults[nRepetition_current]
        if bestResults_thisRep:
            if bestResults_thisRep[-1][5] < bestResult[5]: bestResult_eff = bestResult
            else:                                          bestResult_eff = bestResults_thisRep[-1]
        else:
            bestResult_eff = bestResult
        bestResults_thisRep.append(bestResult_eff)

        #[6]: Compute Gradients
        scores      = torch.cat(scores)
        scores_view = scores.view(nSeekerPoints, 2, nParameters)
        dx = torch.diagonal(params_plus-params_minus, dim1=-2, dim2=-1)
        dx = torch.where(dx == 0, torch.tensor(1e-9, device='cuda'), dx)
        dy = scores_view[:,0,:]-scores_view[:,1,:]
        gradients = dy / (dx + 1e-12)

        #[7]: Update Velocity, Momentum, and Parameters Base
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

        #[8]: Compute Best Score Delta EMA
        scoringSamples           = seeker['scoringSamples']
        bestScore                = bestResult[5]
        bestScore_delta_ema      = None
        bestScore_delta_ema_prev = seeker['_bestScore_delta_ema']
        if (scoringSamples+1 <= currentStep):
            #[8-1]: Calculate SMA (the first value)
            if bestScore_delta_ema_prev is None:
                bestScore_deltas_sum = sum((bestResults_thisRep[rIndex][5]/max(bestResults_thisRep[rIndex-1][5], 1e-12))-1 for rIndex in range (1, len(bestResults_thisRep)))
                bestScore_deltas_sma = bestScore_deltas_sum/scoringSamples
                bestScore_delta_ema = bestScore_deltas_sma
            #[8-2]: Calculate EMA
            else:
                bestScore_delta = (bestScore/max(bestResults_thisRep[-2][5], 1e-12))-1
                ema_k = 2/(scoringSamples+1)
                bestScore_delta_ema = (bestScore_delta*ema_k) + (bestScore_delta_ema_prev*(1-ema_k))
            #[8-4]: Update EMA
            seeker['_bestScore_delta_ema'] = bestScore_delta_ema

        #[9]: Check Termination
        terminationThreshold = seeker['terminationThreshold']
        nRepetition_total    = seeker['nRepetition']
        if bestScore_delta_ema is not None and bestScore_delta_ema < terminationThreshold:
            nRepetition_next             = nRepetition_current + 1
            seeker['_currentRepetition'] = nRepetition_next
            if nRepetition_next == nRepetition_total:
                return (True, 
                        nRepetition_current, 
                        currentStep, 
                        bestResults_thisRep[-1], 
                        t_processing_sim_paramsSet_ms,
                        (time.perf_counter_ns()-t_cpu_beg_ns)/1e6/len(params_test)-t_processing_sim_paramsSet_ms)
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
                return (False, 
                        nRepetition_current, 
                        currentStep, 
                        bestResults_thisRep[-1], 
                        t_processing_sim_paramsSet_ms,
                        (time.perf_counter_ns()-t_cpu_beg_ns)/1e6/len(params_test)-t_processing_sim_paramsSet_ms)

        #[10]: Step Count Update
        seeker['_currentStep'] = currentStep+1

        #[11]: Repopulate (If needed)
        repop_interval   = seeker['repopulationInterval']
        repop_ratio      = seeker['repopulationRatio']
        repop_guideRatio = seeker['repopulationGuideRatio']
        repop_decayRate  = seeker['repopulationDecayRate']
        if (currentStep % repop_interval == 0):
            #[11-1]: Number of seekers to repopulate (Randomize)
            n_toRepopulate = int(nSeekerPoints * repop_ratio)
            if 0 < n_toRepopulate:
                scores_flat    = scores_view.view(nSeekerPoints, -1).max(dim=1)[0]
                sorted_indices = torch.argsort(scores_flat, descending=True)
                n_survived       = max(1, nSeekerPoints-n_toRepopulate)
                survived_indices = sorted_indices[:n_survived]
                repop_indices    = sorted_indices[-n_toRepopulate:]

                n_guided         = int(n_toRepopulate * repop_guideRatio)
                n_completeRandom = n_toRepopulate-n_guided
                
                #[11-2]: Guided Randomization
                if 0 < n_guided:
                    #[11-2-1]: Survived mean and STD
                    survived = seeker['_params_base'][survived_indices]
                    survived_mean = torch.mean(survived, dim=0)
                    survived_std  = torch.std(survived, dim=0)*math.exp(-(currentStep//repop_interval-1)*repop_decayRate)
                    survived_std  = torch.max(survived_std, 2.0/params_rounding_factors)
                    
                    #[11-2-2]: Generate random params using normal distribution
                    p_guided = torch.normal(mean = survived_mean.repeat(n_guided, 1), 
                                            std  = survived_std.repeat(n_guided,  1))
                    p_guided = torch.max(torch.min(p_guided, params_max), params_min)
                    p_guided = torch.round(p_guided * params_rounding_factors) / params_rounding_factors
                    p_guided[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]
                    
                    #[11-2-3]: Apply new base parameters
                    target_indices = repop_indices[n_completeRandom:]
                    seeker['_params_base'][target_indices] = p_guided
                    seeker['_velocity'][target_indices]    = 0.0
                    seeker['_momentum'][target_indices]    = 0.0

                #[11-3]: Complete Randomization
                if 0 < n_completeRandom:
                    #[11-3-1]: Generate completely random params
                    p_rand = torch.rand((n_completeRandom, nParameters), device='cuda', dtype=_TORCHDTYPE)
                    p_rand = p_rand * (params_max - params_min) + params_min
                    p_rand = torch.round(p_rand * params_rounding_factors) / params_rounding_factors
                    p_rand[:, params_fixed_mask] = params_fixed_values[params_fixed_mask]

                    #[11-3-2]: Apply new base parameters
                    seeker['_params_base'][repop_indices[:n_completeRandom]] = p_rand
                    seeker['_velocity'][repop_indices[:n_completeRandom]]    = 0.0
                    seeker['_momentum'][repop_indices[:n_completeRandom]]    = 0.0

        #[12]: Finally
        return (False, 
                nRepetition_current, 
                currentStep, 
                bestResults_thisRep[-1], 
                t_processing_sim_paramsSet_ms,
                (time.perf_counter_ns()-t_cpu_beg_ns)/1e6/len(params_test)-t_processing_sim_paramsSet_ms)

    def __processBatch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        #[1]: Data
        size_paramsBatch = params.size(dim = 0)
        size_dataLen     = self.__data_klines.size(dim = 0)
        params_trade = params[:,:len(TRADEPARAMS)]
        params_model = params[:,len(TRADEPARAMS):]

        #[2]: Model Parameters Length Padding
        params_lToPad = (16-(params_model.size(dim=1)%16))%16
        if 0 < params_lToPad: params_model = torch.cat([params_model, torch.zeros((size_paramsBatch, params_lToPad), device='cuda', dtype=_TORCHDTYPE)], dim=1)
        params_model = params_model.contiguous()

        #[3]: Result Buffers Initialization
        balance_finals               = torch.empty(size = (size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        balance_bestFit_intercepts   = torch.empty(size = (size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        balance_bestFit_growthRates  = torch.empty(size = (size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        balance_bestFit_volatilities = torch.empty(size = (size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        if self.isSeeker:
            balance_wallet_history  = torch.empty(size = (1,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
            balance_margin_history  = torch.empty(size = (1,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        else:
            balance_wallet_history  = torch.empty(size = (size_paramsBatch, size_dataLen), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
            balance_margin_history  = torch.empty(size = (size_paramsBatch, size_dataLen), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)
        balance_ftIndexes = torch.full(size = (size_paramsBatch,), fill_value = -1, device = 'cuda', dtype = torch.int32, requires_grad = False)
        nTrades = torch.zeros(size = (size_paramsBatch,), device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False)

        #[4]: Processing
        self.modelBatchProcessFunction(#Constants
                                       leverage        = self.leverage,
                                       allocationRatio = ALLOCATIONRATIO,
                                       tradingFee      = TRADINGFEE,
                                       #Base Data
                                       data_klines             = self.__data_klines,
                                       data_klines_stride      = self.__data_klines.stride(dim=0),
                                       data_analysis           = self.__data_analysis,
                                       data_analysis_stride    = self.__data_analysis.stride(dim=0),
                                       params_trade_fslImmed   = params_trade[:,0].contiguous(),
                                       params_trade_fslClose   = params_trade[:,1].contiguous(),
                                       params_trade_pslReentry = self.pslReentry,
                                       params_model            = params_model,
                                       params_model_stride     = params_model.stride(dim = 0),
                                       #Result Buffers
                                       balance_finals               = balance_finals,
                                       balance_bestFit_intercepts   = balance_bestFit_intercepts,
                                       balance_bestFit_growthRates  = balance_bestFit_growthRates,
                                       balance_bestFit_volatilities = balance_bestFit_volatilities,
                                       balance_wallet_history       = balance_wallet_history,
                                       balance_margin_history       = balance_margin_history,
                                       balance_ftIndexes            = balance_ftIndexes,
                                       nTrades_rb                   = nTrades,
                                       #Sizes
                                       size_paramsBatch = size_paramsBatch,
                                       size_dataLen     = size_dataLen,
                                       #Mode
                                       SEEKERMODE = self.isSeeker
                                       )

        #[5]: Return Result
        if self.isSeeker: 
            return balance_finals, balance_bestFit_growthRates, balance_bestFit_volatilities, nTrades
        else:
            indexGrid         = torch.arange(size_dataLen, device='cuda', dtype=_TORCHDTYPE).unsqueeze(0)
            ftIndexes_bc      = balance_ftIndexes.unsqueeze(1)
            mask_validRegion  = (indexGrid >= ftIndexes_bc) & (ftIndexes_bc != -1)
            balance_bestFit_x = indexGrid - ftIndexes_bc
            balance_bestFit_history_raw = torch.exp(balance_bestFit_growthRates.unsqueeze(1)*balance_bestFit_x + balance_bestFit_intercepts.unsqueeze(1))
            balance_bestFit_history = torch.where(mask_validRegion, balance_bestFit_history_raw, float('nan'))
            return balance_wallet_history, balance_margin_history, balance_bestFit_history, balance_bestFit_growthRates, balance_bestFit_volatilities, nTrades

    @BPST_Timer
    def __performOnParams_Timed(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__processBatch(params = params) 

    def performOnParams(self, params: list) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__processBatch(params = torch.tensor(data = params, device = 'cuda', dtype = _TORCHDTYPE, requires_grad = False))
# =======================================================================================================================================================================================================================================================