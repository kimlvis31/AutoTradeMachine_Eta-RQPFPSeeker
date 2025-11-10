import time
import numpy
import datetime
import torch
import json
import os
import matplotlib
import matplotlib.pyplot
import pprint
import termcolor
import pickle
import math
import pandas
import sys

_DEVICE     = 'cuda'
_NUMPYDTYPE = numpy.float32
_TORCHDTYPE = torch.float32

_TRADINGFEE = 0.0005
_RQPFUNCTION_PRICEPERCENTAGEPRECISION = 3
_RQPFUNCTION_SIGNALSTRENGTHPRECISION  = 2
_RQPFUNCTION_UNITPRICEPERCENTAGE      = round(pow(10, -_RQPFUNCTION_PRICEPERCENTAGEPRECISION), _RQPFUNCTION_PRICEPERCENTAGEPRECISION)
_RQPFUNCTION_UNITSIGNALSTRENGTH       = round(pow(10, -_RQPFUNCTION_SIGNALSTRENGTHPRECISION),  _RQPFUNCTION_SIGNALSTRENGTHPRECISION)

_BATCHPROCESSINGSPEEDTRACKER_KVALUE = 2/(100+1)

LOOPERSIZE         = 128
PARAMETERBATCHSIZE = 128

ALLOCATIONRATIO = 0.90

def preprocessCycleData(cycleRecords):
    _maxCycleLen = max([len(_cRec['history']) for _cRec in cycleRecords])
    _cRecs_preprocessed = numpy.empty(shape = (len(cycleRecords), _maxCycleLen, 5), dtype = _NUMPYDTYPE)
    for _cIndex, _cRec in enumerate (cycleRecords):
        _cRec_type     = _cRec['type']
        _cRec_history  = _cRec['history']
        _nThisCycleLen = len(_cRec_history)
        for _i in range (_maxCycleLen):
            if (_i < _nThisCycleLen): 
                _cRecs_preprocessed[_cIndex][_i][0] = _i                   #Continuation Index
                _cRecs_preprocessed[_cIndex][_i][1] = _cRec_history[_i][0] #pDPerc
                _cRecs_preprocessed[_cIndex][_i][2] = _cRec_history[_i][1] #pDPerc_lastSwing
                _cRecs_preprocessed[_cIndex][_i][3] = _cRec_history[_i][2] #pDPerc_reverseMax
                _cRecs_preprocessed[_cIndex][_i][4] = _cRec_history[_i][3] #Signal Strength
            else:                     
                _cRecs_preprocessed[_cIndex][_i][0] = None
                _cRecs_preprocessed[_cIndex][_i][1] = None
                _cRecs_preprocessed[_cIndex][_i][2] = None
                _cRecs_preprocessed[_cIndex][_i][3] = None
                _cRecs_preprocessed[_cIndex][_i][4] = None
        _cRecs_preprocessed[_cIndex][0][3] = 0
    return _cRecs_preprocessed

def removeConsoleLines(nLinesToRemove):
    for _ in range (nLinesToRemove): 
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()

def timeStringFormatter(time_seconds):
    if   (time_seconds < 60):    return "00:{:02d}".format(time_seconds)                                                                                                                                  #Less than a minute
    elif (time_seconds < 3600):  return "{:02d}:{:02d}".format(int(time_seconds/60), time_seconds%60)                                                                                                     #Less than an hour
    elif (time_seconds < 86400): return "{:02d}:{:02d}:{:02d}".format(int(time_seconds/3600), int((time_seconds-int(time_seconds/3600)*3600)/60), time_seconds%60)                                        #Less than a day
    else: return "{:d}:{:02d}:{:02d}:{:02d}".format(int(time_seconds/86400), int((time_seconds-int(time_seconds/86400)*86400)/3600), int((time_seconds-int(time_seconds/3600)*3600)/60), time_seconds%60) #More than a day

#Exit Function Model
class exitFunction():
    def __init__(self, entryMode, isSeeker, leverage):
        self.entryMode = entryMode
        self.leverage  = leverage
        self.isSeeker  = isSeeker

        #Remaining Quantity Percentage Maps
        self.__params_batches      = None
        self.__params_nValidParams = None
        self.__params_nBatches     = None

        #Data Set
        self.__dataSet = None

        #Forward Function Variables
        self.__stepperFunction                    = None
        self.__stepperBuffer_rqpVals              = None
        self.__stepperBuffer_price                = None
        self.__stepperBuffer_price_rm             = None
        self.__stepperBuffer_balance              = None
        self.__stepperBuffer_allocatedBalance     = None
        self.__stepperBuffer_quantity_zeroIndex   = None
        self.__stepperBuffer_entryPrice_zeroIndex = None
        self.__stepperBuffer_tradeParams          = None

        #Optimized Parameters Seeker
        self._optimizedParametersSeeker = None
        self._ops_resultsByParams_prev = dict()
        self._ops_resultsByParams_this = dict()

    def loadRQPFunctionParamsSet(self, rqpFunctionParamsSet):
        _paramsSet = torch.tensor(data = rqpFunctionParamsSet, device = _DEVICE, dtype = _TORCHDTYPE, requires_grad = False)
        self.__params_nValidParams = _paramsSet.size(dim = 0)
        self.__params_nBatches     = math.ceil(self.__params_nValidParams/PARAMETERBATCHSIZE)
        #---Padding
        _nToPad = self.__params_nBatches*PARAMETERBATCHSIZE-self.__params_nValidParams
        if (0 < _nToPad):
            _pad = torch.zeros((_nToPad, _paramsSet.size(dim = 1)), dtype=_TORCHDTYPE, device=_DEVICE)
            _paramsSet = torch.cat([_paramsSet, _pad], dim=0)
        #---Batching
        self.__params_batches = [_paramsSet[_batchIndex*PARAMETERBATCHSIZE:(_batchIndex+1)*PARAMETERBATCHSIZE,:] for _batchIndex in range (self.__params_nBatches)]

    def getRQPValues(self, continuations, priceDeltaPercentanges, priceDeltaPercentanges_lastSwing, signalStrengths):
        _rqpVals = torch.zeros(size = (PARAMETERBATCHSIZE,)+continuations.size(), device = _DEVICE, dtype = _TORCHDTYPE)
        return _rqpVals

    def loadDataSet(self, dataSet):
        _dataSet = torch.tensor(data = dataSet, device = _DEVICE, dtype = _TORCHDTYPE, requires_grad = False)
        self.__dataSet_nValidSamples = _dataSet.size(dim = 0)
        #Batching
        #---nSamples
        _nToPad = math.ceil(self.__dataSet_nValidSamples/32)*32-self.__dataSet_nValidSamples
        if (0 < _nToPad):
            _pad = torch.zeros((_nToPad, _dataSet.size(dim = 1), _dataSet.size(dim = 2)), dtype=_TORCHDTYPE, device=_DEVICE)
            _dataSet = torch.cat([_dataSet, _pad], dim=0)
        #---cycleLength
        _targetLength_LOOPER = math.ceil((_dataSet.size(dim=1)-1)/(LOOPERSIZE-1))*(LOOPERSIZE-1)
        _targetLength_LENGTH = math.ceil(_targetLength_LOOPER/LOOPERSIZE)*LOOPERSIZE
        _lengthDelta = _targetLength_LENGTH-_dataSet.size(dim = 1)
        if (0 < _lengthDelta): 
            _pad     = torch.zeros((_dataSet.size(dim = 0), _lengthDelta, _dataSet.size(dim = 2)), dtype=_TORCHDTYPE, device=_DEVICE)
            _dataSet = torch.cat([_dataSet, _pad], dim=1)
        elif (_lengthDelta < 0): _dataSet = _dataSet[:,:_targetLength_LENGTH,:]
        #---Finally
        self.__dataSet = _dataSet
        #Stepper Functions
        if   (self.entryMode == 'SHORT'): _sf_entryDir = -1
        elif (self.entryMode == 'LONG'):  _sf_entryDir =  1
        _leverage      = self.leverage
        _dataSetSize_0 = self.__dataSet.size(dim = 0)
        _dataSetSize_1 = self.__dataSet.size(dim = 1)
        def stepperFunction(rqpVals, price, price_reverseMax, balance, allocatedBalance, quantity_zeroIndex, entryPrice_zeroIndex, tradeParams):
            #Updates
            _quantity_prev    = quantity_zeroIndex.clone()
            _entryPrice_prev  = entryPrice_zeroIndex.clone()
            _price            = price.unsqueeze(0).expand(PARAMETERBATCHSIZE, -1, -1)
            _price_reverseMax = price_reverseMax.unsqueeze(0).expand_as(_price)
            _tradeParams_fsl  = tradeParams[:,0].unsqueeze(-1).expand(-1, _dataSetSize_0)
            _balance          = balance.clone()
            _allocatedBalance = allocatedBalance.clone()
            #Loop
            for _eIndex in range (1, LOOPERSIZE):
                #Tensor Views
                _rqpVals_this          = rqpVals[:,:,_eIndex]
                _price_this            = _price[:,:,_eIndex]
                _price_reverseMax_this = _price_reverseMax[:,:,_eIndex]
                #Trade Execution Types
                _fslPrice = _entryPrice_prev*(1-_sf_entryDir*_tradeParams_fsl)
                _liqPrice = _entryPrice_prev-_sf_entryDir*_entryPrice_prev/_leverage
                _holdingQuantity = (0 < _quantity_prev)
                _fslHit          = _holdingQuantity & (_sf_entryDir*_price_reverseMax_this <= _sf_entryDir*_fslPrice)
                _liquidated      = _holdingQuantity & (_sf_entryDir*_price_reverseMax_this <= _sf_entryDir*_liqPrice)
                #Trade Prices
                _tradePrice = torch.where(_liquidated,
                                          _liqPrice,
                                          torch.where(_fslHit, _fslPrice, _price_this)) #Liquidation
                #Trade Quantities
                _targetCommittedBalance  = _allocatedBalance*_rqpVals_this
                _currentCommittedBalance = _quantity_prev*_entryPrice_prev
                _balanceToCommit         = _targetCommittedBalance-_currentCommittedBalance
                _quantity_trade          = torch.where(0 < _balanceToCommit, _balanceToCommit/_price_this, _balanceToCommit/_entryPrice_prev.clamp_min(min = 1e-12))
                _quantity_trade = torch.where(_fslHit | _liquidated, -_quantity_prev, _quantity_trade)
                _quantity_trade = torch.where(_quantity_trade < -_quantity_prev, -_quantity_prev, _quantity_trade)
                #New Quantity & Entry Price
                _quantity_entry = torch.relu( _quantity_trade)
                _quantity_exit  = torch.relu(-_quantity_trade)
                _quantity_new   = (_quantity_prev+_quantity_entry-_quantity_exit).clamp_min(min = 0.0)
                _entryPrice_new = torch.where(_quantity_new == 0, 0.0, (_quantity_prev*_entryPrice_prev+_quantity_entry*_tradePrice-_quantity_exit*_entryPrice_prev)/_quantity_new.clamp_min(min = 1e-12))
                #Profit
                _profit_this     = _quantity_exit*_sf_entryDir*(_tradePrice-_entryPrice_prev)
                _tradingFee_this = (_quantity_entry+_quantity_exit)*_tradePrice*_TRADINGFEE
                #Apply Updates
                _quantity_prev   = _quantity_new
                _entryPrice_prev = _entryPrice_new
                _balance.add_(torch.nan_to_num((_profit_this-_tradingFee_this)*_leverage, nan = 0.0))
                #New Balance Allocation
                _allocatedBalance = torch.where(_quantity_new == 0, _balance*ALLOCATIONRATIO, _allocatedBalance)
            return _balance, _allocatedBalance, _quantity_prev, _entryPrice_prev
        self.__stepperFunction = torch.compile(model = stepperFunction, mode="max-autotune", dynamic=False, fullgraph=True)
        self.__stepperBuffer_rqpVals              = torch.zeros(size = (PARAMETERBATCHSIZE, _dataSetSize_0, _dataSetSize_1), device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_price                = torch.zeros(size =                     (_dataSetSize_0, _dataSetSize_1), device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_price_rm             = torch.zeros(size =                     (_dataSetSize_0, _dataSetSize_1), device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_balance              = torch.zeros(size = (PARAMETERBATCHSIZE, _dataSetSize_0),                 device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_allocatedBalance     = torch.zeros(size = (PARAMETERBATCHSIZE, _dataSetSize_0),                 device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_quantity_zeroIndex   = torch.zeros(size = (PARAMETERBATCHSIZE, _dataSetSize_0),                 device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_entryPrice_zeroIndex = torch.zeros(size = (PARAMETERBATCHSIZE, _dataSetSize_0),                 device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperBuffer_tradeParams          = torch.zeros(size = (PARAMETERBATCHSIZE, 1),                              device=_DEVICE, dtype=_TORCHDTYPE)
        self.__stepperFunction(rqpVals              = self.__stepperBuffer_rqpVals.narrow(dim =2,start=0,length=LOOPERSIZE),
                               price                = self.__stepperBuffer_price.narrow(dim   =1,start=0,length=LOOPERSIZE),
                               price_reverseMax     = self.__stepperBuffer_price_rm.narrow(dim=1,start=0,length=LOOPERSIZE),
                               balance              = self.__stepperBuffer_balance,
                               allocatedBalance     = self.__stepperBuffer_allocatedBalance,
                               quantity_zeroIndex   = self.__stepperBuffer_quantity_zeroIndex, 
                               entryPrice_zeroIndex = self.__stepperBuffer_entryPrice_zeroIndex,
                               tradeParams          = self.__stepperBuffer_tradeParams)

    def performSeeker(self):
        #[1]: Generate Parameters Set
        _paramsSetToTest = self.__performSeeker_getParametersSets()
        self.loadRQPFunctionParamsSet(rqpFunctionParamsSet = _paramsSetToTest)

        #[2]: Batches
        _ce_beg = torch.cuda.Event(enable_timing=True)
        _ce_end = torch.cuda.Event(enable_timing=True)
        _avgBatchProcessingTime = None
        _bsPrinted          = False
        _bsTrackingInterval = 5
        _outputs = torch.zeros(size = (PARAMETERBATCHSIZE*self.__params_nBatches,), dtype = _TORCHDTYPE, device = _DEVICE)
        for _batchIndex_beg in range (0, self.__params_nBatches, _bsTrackingInterval):
            _ce_beg.record()
            if (_batchIndex_beg+_bsTrackingInterval <= self.__params_nBatches): _nBatches_thisChunk = _bsTrackingInterval
            else:                                                               _nBatches_thisChunk = self.__params_nBatches-_batchIndex_beg
            for _batchIndex in range (_batchIndex_beg, _batchIndex_beg+_nBatches_thisChunk): _outputs[_batchIndex*PARAMETERBATCHSIZE:(_batchIndex+1)*PARAMETERBATCHSIZE].copy_(self.processBatch(params = self.__params_batches[_batchIndex]))
            _ce_end.record()
            _ce_end.synchronize()
            _t_elapsed_perBatch = _ce_beg.elapsed_time(_ce_end)/1e3/_nBatches_thisChunk
            if (0 < _batchIndex_beg):
                if (_avgBatchProcessingTime is None): _avgBatchProcessingTime = _t_elapsed_perBatch
                else:                                 _avgBatchProcessingTime = _t_elapsed_perBatch*_BATCHPROCESSINGSPEEDTRACKER_KVALUE + _avgBatchProcessingTime*(1-_BATCHPROCESSINGSPEEDTRACKER_KVALUE)
                _ps_batch        = PARAMETERBATCHSIZE/_avgBatchProcessingTime
                _t_remaining     = _avgBatchProcessingTime*(self.__params_nBatches-_batchIndex-1)
                _t_remaining_str = timeStringFormatter(time_seconds = int(_t_remaining))
                if (_bsPrinted == True): removeConsoleLines(nLinesToRemove = 1)
                print(f"   * PARAMBATCH: {_batchIndex_beg+1}~{_batchIndex_beg+_nBatches_thisChunk}/{self.__params_nBatches} <{_ps_batch:.1f} Parameter Sets / Second> <ETC: {_t_remaining_str}>")
                _bsPrinted = True
        _outputs = _outputs[:len(_paramsSetToTest)].detach().cpu()
        if (_bsPrinted == True): removeConsoleLines(nLinesToRemove = 1)

        #[3]: Results Scoring
        for _index, _paramsSet in enumerate (_paramsSetToTest): 
            self._ops_resultsByParams_this[_paramsSet] = float(_outputs[_index])

        #[4]: Best Results Search
        _bestResult = None
        for _params, _finalBalance in self._ops_resultsByParams_this.items():
            if ((_bestResult is None) or (_bestResult[1] < _finalBalance)): _bestResult = (_params, _finalBalance)

        #[5]: Pivot Params Update
        if (_bestResult[0] == self._optimizedParametersSeeker['currentPivotParams']):
            if (self._optimizedParametersSeeker['currentPrecisionStep'] < self._optimizedParametersSeeker['maxPrecisionStep']):
                self._optimizedParametersSeeker['currentPrecisionStep'] += 1
                _moreToGo = True
            else: _moreToGo = False
        else: _moreToGo = True
        self._optimizedParametersSeeker['currentPivotParams']       = _bestResult[0]
        self._optimizedParametersSeeker['currentPivotParamsResult'] = {'finalBalance': _bestResult[1]}
        self._ops_resultsByParams_prev = self._ops_resultsByParams_this.copy()
        self._ops_resultsByParams_this.clear()
        return _moreToGo

    def __performSeeker_getParametersSets(self):
        _cps = self._optimizedParametersSeeker['currentPrecisionStep']
        _ep = list()
        for _paramIndex in range (len(self._optimizedParametersSeeker['currentPivotParams'])):
            _ppuConfig = self._optimizedParametersSeeker['pivotParamsUpdateConfiguration'][_paramIndex]
            if ('absolutePrecision' in _ppuConfig): _ep.append(_ppuConfig['absolutePrecision'])
            else:                                   _ep.append(_ppuConfig['relativePrecision']+_cps)
        _res = [pow(10, -(_ep[_paramIndex])) for _paramIndex in range (len(self._optimizedParametersSeeker['currentPivotParams']))]
        _paramsSetToTest = set()
        _pv_list = list(self._optimizedParametersSeeker['currentPivotParams'])
        for _baseIndex in range (3**self._optimizedParametersSeeker['nUpdateTargets']):
            _baseIndex_forDivmod = _baseIndex
            _pv = _pv_list.copy()
            for _ppuConfig in self._optimizedParametersSeeker['pivotParamsUpdateConfiguration']:
                _ppuConfig_index = _ppuConfig['index']
                _ppuConfig_limit = _ppuConfig['limit']
                if (_ppuConfig['update'] == True): 
                    _baseIndex_forDivmod, _tIndex = divmod(_baseIndex_forDivmod, 3)
                    if   (_tIndex == 0): _newVal = round(_pv[_ppuConfig_index],                        _ep[_ppuConfig_index])
                    elif (_tIndex == 1): _newVal = round(_pv[_ppuConfig_index]-_res[_ppuConfig_index], _ep[_ppuConfig_index])
                    elif (_tIndex == 2): _newVal = round(_pv[_ppuConfig_index]+_res[_ppuConfig_index], _ep[_ppuConfig_index])
                    if ((_ppuConfig_limit is None) or ((_ppuConfig_limit[0] <= _newVal) and (_newVal <= _ppuConfig_limit[1]))): _pv[_ppuConfig_index] = _newVal
            _pv_tuple = tuple(_pv)
            if (_pv_tuple in self._ops_resultsByParams_prev): self._ops_resultsByParams_this[_pv_tuple] = self._ops_resultsByParams_prev[_pv_tuple]
            else:                                             _paramsSetToTest.add(_pv_tuple)
        return list(_paramsSetToTest)

    def processBatch(self, params = None):
        if (self.isSeeker == True): _targetParams = [params,];             _outputs = None
        else:                       _targetParams = self.__params_batches; _outputs = torch.zeros(size = (PARAMETERBATCHSIZE*self.__params_nBatches, self.__dataSet.size(dim=0)), dtype = _TORCHDTYPE, device = _DEVICE)
        with torch.no_grad():
            for _index, _params in enumerate(_targetParams):
                #Samples Nan & Last Valid
                _nanMask   = torch.isnan(self.__dataSet[:,:,0])
                _hasNan    = _nanMask.any(dim = 1)
                _lastValid = torch.amax(torch.where(_nanMask, 0, self.__dataSet[:,:,0]), dim = 1).int()
                #RQP Maps Generation
                _rqpVals = self.getRQPValues(params = _params,
                                             continuations                    = self.__dataSet.select(dim=2,index=0),
                                             priceDeltaPercentanges           = self.__dataSet.select(dim=2,index=1),
                                             priceDeltaPercentanges_lastSwing = self.__dataSet.select(dim=2,index=2),
                                             signalStrengths                  = self.__dataSet.select(dim=2,index=4))
                #Output
                #---RQP Value Zeroing on Cycle Ends
                _rows = torch.arange(self.__dataSet.size(dim = 0), device=_DEVICE)
                _rqpVals[:,_rows[ _hasNan], _lastValid[_hasNan]]            = 0 #If has nan, set the rqpVal to 0 where the last non-nan value appears
                _rqpVals[:,_rows[~_hasNan], self.__dataSet.size(dim = 1)-1] = 0 #If has no nan, set the rqpVal to 0 where the last non-nan value appears
                #---Stepper Buffer
                self.__stepperBuffer_rqpVals.zero_()
                self.__stepperBuffer_price.zero_()
                self.__stepperBuffer_price_rm.zero_()
                self.__stepperBuffer_rqpVals.copy_(_rqpVals)
                self.__stepperBuffer_price.copy_(self.__dataSet.select(dim=2,index=1)).add_(1)
                self.__stepperBuffer_price_rm.copy_(self.__dataSet.select(dim=2,index=3)).add_(1)
                self.__stepperBuffer_tradeParams.copy_(_params.narrow(dim=1,start=0,length=1))
                self.__stepperBuffer_balance.zero_().add_(1)
                self.__stepperBuffer_allocatedBalance.copy_(self.__stepperBuffer_balance*ALLOCATIONRATIO)
                #---Index 0
                self.__stepperBuffer_quantity_zeroIndex.zero_()
                self.__stepperBuffer_entryPrice_zeroIndex.zero_()
                _targetCommittedBalance  = self.__stepperBuffer_allocatedBalance*_rqpVals.select(dim=2,index=0)
                _currentCommittedBalance = self.__stepperBuffer_quantity_zeroIndex*self.__stepperBuffer_entryPrice_zeroIndex
                _balanceToCommit         = (_targetCommittedBalance-_currentCommittedBalance).clamp_min(min = 0)
                _quantitiesToEnter       = _balanceToCommit/self.__stepperBuffer_price.select(dim=1,index=0)[None,]
                self.__stepperBuffer_quantity_zeroIndex.copy_(_quantitiesToEnter)
                self.__stepperBuffer_entryPrice_zeroIndex.copy_(torch.where(0 < self.__stepperBuffer_quantity_zeroIndex, 1.0, 0.0))
                self.__stepperBuffer_balance.subtract_(_quantitiesToEnter*1.0*_TRADINGFEE*self.leverage)
                #---Stepper Function & Net Profit Computation
                _eIndex_beg = 0
                _eIndex_end = _eIndex_beg+LOOPERSIZE
                while (_eIndex_end <= self.__dataSet.size(dim = 1)):
                    _balance_looped, _allocatedBalance_looped, _quantity_looperLast, _entryPrice_looperLast = self.__stepperFunction(rqpVals              = self.__stepperBuffer_rqpVals.narrow(dim =2,start=_eIndex_beg,length=LOOPERSIZE),
                                                                                                                                     price                = self.__stepperBuffer_price.narrow(dim   =1,start=_eIndex_beg,length=LOOPERSIZE),
                                                                                                                                     price_reverseMax     = self.__stepperBuffer_price_rm.narrow(dim=1,start=_eIndex_beg,length=LOOPERSIZE),
                                                                                                                                     balance              = self.__stepperBuffer_balance,
                                                                                                                                     allocatedBalance     = self.__stepperBuffer_allocatedBalance,
                                                                                                                                     quantity_zeroIndex   = self.__stepperBuffer_quantity_zeroIndex,
                                                                                                                                     entryPrice_zeroIndex = self.__stepperBuffer_entryPrice_zeroIndex,
                                                                                                                                     tradeParams          = self.__stepperBuffer_tradeParams)
                    self.__stepperBuffer_balance.copy_(_balance_looped)
                    self.__stepperBuffer_allocatedBalance.copy_(_allocatedBalance_looped)
                    self.__stepperBuffer_quantity_zeroIndex.copy_(_quantity_looperLast)
                    self.__stepperBuffer_entryPrice_zeroIndex.copy_(_entryPrice_looperLast)
                    _eIndex_beg += (LOOPERSIZE-1)
                    _eIndex_end = _eIndex_beg + LOOPERSIZE
                if (self.isSeeker == False): _outputs[_index*PARAMETERBATCHSIZE:(_index+1)*PARAMETERBATCHSIZE,:].copy_(self.__stepperBuffer_balance)
        #---Finally
        if (self.isSeeker == True): return torch.prod(input = self.__stepperBuffer_balance[:,:self.__dataSet_nValidSamples], dim = 1)
        else:                       return _outputs.detach().cpu()
        
    def getCurrentPivotParams(self):
        return self._optimizedParametersSeeker['currentPivotParams']
    def getCurrentPivotParamsResult(self):
        return self._optimizedParametersSeeker['currentPivotParamsResult'].copy()

#Exit Function Model - Rotational Gaussian
class exitFunction_RotationalGaussian(exitFunction):
    def __init__(self, entryMode, isSeeker, leverage):
        super().__init__(entryMode = entryMode, isSeeker = isSeeker, leverage = leverage)

    def getRQPValues(self, params, continuations, priceDeltaPercentanges, priceDeltaPercentanges_lastSwing, signalStrengths):
        _angle0  = params.select(dim=1,index=1)[...,None,None]*2*torch.pi
        _angle1  = params.select(dim=1,index=2)[...,None,None]*2*torch.pi
        _angle2  = params.select(dim=1,index=3)[...,None,None]*2*torch.pi
        _c0, _s0 = torch.cos(_angle0), torch.sin(_angle0)
        _c1, _s1 = torch.cos(_angle1), torch.sin(_angle1)
        _c2, _s2 = torch.cos(_angle2), torch.sin(_angle2)
        _a0_0 = continuations[None]*0.001             +params.select(dim=1,index=4)[...,None,None]
        _a1_0 = priceDeltaPercentanges[None]          +params.select(dim=1,index=5)[...,None,None]
        _a2_0 = priceDeltaPercentanges_lastSwing[None]+params.select(dim=1,index=6)[...,None,None]
        _a3_0 = signalStrengths[None]                 +params.select(dim=1,index=7)[...,None,None]
        _a0_1   =  _c0*_a0_0+_s0*_a1_0
        _a1_rot = -_s0*_a0_0+_c0*_a1_0
        _a0_2   =  _c1*_a0_1+_s1*_a2_0
        _a2_rot = -_s1*_a0_1+_c1*_a2_0
        _a0_rot =  _c2*_a0_2+_s2*_a3_0
        _a3_rot = -_s2*_a0_2+_c2*_a3_0
        _q = 0.5 * (torch.square(_a0_rot/params.select(dim=1,index= 8)[...,None,None]) \
                  + torch.square(_a1_rot/params.select(dim=1,index= 9)[...,None,None]) \
                  + torch.square(_a2_rot/params.select(dim=1,index=10)[...,None,None]) \
                  + torch.square(_a3_rot/params.select(dim=1,index=11)[...,None,None]))
        _rqpVals = torch.exp(-_q**100).round(decimals = 4)
        return _rqpVals

    def initializeOptimizedParametersSeeker(self, begPrecisionStep = 0, maxPrecisionStep = 3, pivotParams = None):
        self._optimizedParametersSeeker = {'maxPrecisionStep':     maxPrecisionStep,
                                           'currentPrecisionStep': begPrecisionStep,
                                           'pivotParamsUpdateConfiguration': [{'index':  0, 'update': True, 'absolutePrecision': 2, 'limit': ( 0.0000, 1.0000)},  #FSL
                                                                              {'index':  1, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0000, 1.0000)},  #Theta0
                                                                              {'index':  2, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0000, 1.0000)},  #Theta1
                                                                              {'index':  3, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0000, 1.0000)},  #Theta2
                                                                              {'index':  4, 'update': True, 'relativePrecision': 1, 'limit': (-1.0000, 1.0000)},  #Delta0
                                                                              {'index':  5, 'update': True, 'relativePrecision': 1, 'limit': (-1.0000, 1.0000)},  #Delta1
                                                                              {'index':  6, 'update': True, 'relativePrecision': 1, 'limit': (-1.0000, 1.0000)},  #Delta2
                                                                              {'index':  7, 'update': True, 'relativePrecision': 1, 'limit': (-1.0000, 1.0000)},  #Delta3
                                                                              {'index':  8, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0001, 1.0000)},  #Sigma0
                                                                              {'index':  9, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0001, 1.0000)},  #Sigma1
                                                                              {'index': 10, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0001, 1.0000)},  #Sigma2
                                                                              {'index': 11, 'update': True, 'relativePrecision': 1, 'limit': ( 0.0001, 1.0000)}   #Sigma3
                                                                              ], 
                                           'currentPivotParams': (0.05, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000),
                                           'currentPivotParamsResult': None,
                                           'nUpdateTargets':           None}
        self._optimizedParametersSeeker['nUpdateTargets'] = len([1 for _ppuConfig in self._optimizedParametersSeeker['pivotParamsUpdateConfiguration'] if (_ppuConfig['update'] == True)])
        if (pivotParams is not None): self._optimizedParametersSeeker['currentPivotParams'] = pivotParams
        self._ops_resultsByParams_prev = dict()
        self._ops_resultsByParams_this = dict()

#Exit Function Model - Rotational Gaussian 2
class exitFunction_RotationalGaussian2(exitFunction):
    def __init__(self, entryMode, isSeeker, leverage):
        super().__init__(entryMode = entryMode, isSeeker = isSeeker, leverage = leverage)

    def getRQPValues(self, params, continuations, priceDeltaPercentanges, priceDeltaPercentanges_lastSwing, signalStrengths):
        _a0 = priceDeltaPercentanges[None]          +params.select(dim=1,index=1)[...,None,None]
        #_a1 = priceDeltaPercentanges_lastSwing[None]+params.select(dim=1,index=1)[...,None,None]
        #_a2 = signalStrengths[None]                 +params.select(dim=1,index=1)[...,None,None]
        _q0 = torch.exp(-0.5*(_a0/params.select(dim=1,index=2)[...,None,None])**2)
        _rqpVals = _q0.round(decimals = 4)
        return _rqpVals

    def initializeOptimizedParametersSeeker(self, begPrecisionStep = 0, maxPrecisionStep = 3, pivotParams = None):
        self._optimizedParametersSeeker = {'maxPrecisionStep':     maxPrecisionStep,
                                           'currentPrecisionStep': begPrecisionStep,
                                           'pivotParamsUpdateConfiguration': [{'index': 0, 'update': True, 'absolutePrecision': 2, 'limit': ( 0.000000, 1.000000)},  #FSL
                                                                              {'index': 1, 'update': True, 'relativePrecision': 1, 'limit': (-1.000000, 1.000000)},  #Delta0
                                                                              {'index': 2, 'update': True, 'relativePrecision': 1, 'limit': ( 0.000001, 1.000000)},  #Sigma0
                                                                              ],
                                           'currentPivotParams': (0.05, 0.0000, 0.0001),
                                           'currentPivotParamsResult': None,
                                           'nUpdateTargets':           None}
        self._optimizedParametersSeeker['nUpdateTargets'] = len([1 for _ppuConfig in self._optimizedParametersSeeker['pivotParamsUpdateConfiguration'] if (_ppuConfig['update'] == True)])
        if (pivotParams is not None): self._optimizedParametersSeeker['currentPivotParams'] = pivotParams
        self._ops_resultsByParams_prev = dict()
        self._ops_resultsByParams_this = dict()

_EXITFUNCTIONS = {'ROTATIONALGAUSSIAN':  exitFunction_RotationalGaussian,
                  'ROTATIONALGAUSSIAN2': exitFunction_RotationalGaussian2,
                  }

if __name__ == "__main__":
    print(termcolor.colored("<RQP MAP PARAMETERS GRID SEARCH PROCESS>\n", 'light_green'))

    _PROCESSBEGINTIME = int(time.time())
    
    _SHOWCYCLEDATAGRAPHICS = True
    _CYCLEDATATOPROCESS = [{'cycleData':        'pf\cycleData_BTCUSDT.json',
                            'exitFunctionType': 'ROTATIONALGAUSSIAN2',
                            'entryMode':        'LONG',
                            'leverage':         1},
                            ]
    #_CYCLEDATATOPROCESS = None
    _RCODETOREAD = 'rqpmf_gsr_1758180490'
    _PARAMETERTEST = {'cycleData':        'pf\cycleData_BTCUSDT.json',
                      'exitFunctionType': 'ROTATIONALGAUSSIAN',
                      'entryMode':        'LONG',
                      'leverage':         10,
                      'params':           (0.07, 0.5, 0.4, 0.6, 0.2, 0.1, 0.4, 0.0, 0.1, 0.2, 0.3, 0.5)
                      }

    #Data Folders Setup
    _path_folder_cycles  = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles')
    _path_folder_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if (os.path.isdir(_path_folder_cycles)  == False): os.mkdir(_path_folder_cycles)
    if (os.path.isdir(_path_folder_results) == False): os.mkdir(_path_folder_results)

    #[1]: Generate Data
    if (_CYCLEDATATOPROCESS is not None):
        print(termcolor.colored("[Grid Search]", 'light_blue'))
        #[1]: Result Folder Generation
        _rCode = f"rqpmf_gsr_{int(_PROCESSBEGINTIME)}"
        _thisResult_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', _rCode)
        os.mkdir(_thisResult_path)
        print(f" * Result Code: {_rCode}")
        #[2]: Cycles Data Identification
        print(f" * Target Cycle Data [{len(_CYCLEDATATOPROCESS)} Targets]")
        _processResults = dict()
        for _index, _cdtp in enumerate(_CYCLEDATATOPROCESS):
            if (os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles', _cdtp['cycleData'])) == True):
                _processResults[_index] = {'final':   None,
                                           'records': list()}
                #File Open
                _file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles', _cdtp['cycleData']), 'r')
                _cRecs = json.loads(_file.read())
                _file.close()
                #Cycle Lengths
                _nCycles     = len(_cRecs)
                _avgCycleLen = sum(len(_cRec['history']) for _cRec in _cRecs)/_nCycles
                _minCycleLen = min(len(_cRec['history']) for _cRec in _cRecs)
                _maxCycleLen = max(len(_cRec['history']) for _cRec in _cRecs)
                #Terminal Price Percentages
                _termPercs_HF = list()
                _termPercs_LF = list()
                for _cRec in _cRecs:
                    if   (_cRec['type'] == 'HF'): _termPercs_HF.append(_cRec['history'][-1][0])
                    elif (_cRec['type'] == 'LF'): _termPercs_LF.append(_cRec['history'][-1][0])
                _termPerc_HF_avg = sum(_termPercs_HF)/len(_termPercs_HF)
                _termPerc_HF_min = min(_termPercs_HF)
                _termPerc_HF_max = max(_termPercs_HF)
                _termPerc_LF_avg = sum(_termPercs_LF)/len(_termPercs_LF)
                _termPerc_LF_min = min(_termPercs_LF)
                _termPerc_LF_max = max(_termPercs_LF)
                #Data Formatting & Print
                print(f"  - [{_index+1}/{len(_CYCLEDATATOPROCESS)}] '{_cdtp['cycleData']}'\n"\
                     +f"   * nCycles:             {_nCycles}\n"\
                     +f"   * CycleLength_Average: {_avgCycleLen:.3f}\n"\
                     +f"   * CycleLength_Minimum: {_minCycleLen}\n"\
                     +f"   * CycleLength_Maximum: {_maxCycleLen}\n"\
                     +f"   * TerminalPriceDelta_SHORT_Average: {_termPerc_HF_avg*100:.3f} %\n"\
                     +f"   * TerminalPriceDelta_SHORT_Minimum: {_termPerc_HF_min*100:.3f} %\n"\
                     +f"   * TerminalPriceDelta_SHORT_Maximum: {_termPerc_HF_max*100:.3f} %\n"\
                     +f"   * TerminalPriceDelta_LONG_Average:  {_termPerc_LF_avg*100:.3f} %\n"\
                     +f"   * TerminalPriceDelta_LONG_Minimum:  {_termPerc_LF_min*100:.3f} %\n"\
                     +f"   * TerminalPriceDelta_LONG_Maximum:  {_termPerc_LF_max*100:.3f} %")
                #Matplot Drawing
                if (_SHOWCYCLEDATAGRAPHICS == True):
                    fig, axs = matplotlib.pyplot.subplots(3, constrained_layout=True)
                    axs[0].set_title("HIGH-FIRST", fontsize=8)
                    axs[1].set_title("LOW-FIRST",  fontsize=8)
                    axs[2].set_title("ALL",   fontsize=8)
                    for _ax in axs:
                        _ax.grid(True)
                        _ax.set_xlim(-int(_maxCycleLen)*0.05, int(_maxCycleLen)*1.05)
                        _ax.set_xlabel("N Candles",       fontsize=6)
                        _ax.set_ylabel("Price Delta [%]", fontsize=6)
                        _ax.tick_params(axis='both', labelsize=6)
                    matplotlib.pyplot.suptitle("Cycle Data - '{:s}'".format(_cdtp['cycleData']), fontsize=10)
                    for _cRec in _cRecs:
                        _cRec_type   = _cRec['type']
                        _cRec_pPercs = [_data[0] for _data in _cRec['history']]
                        _x = list(range(0, len(_cRec_pPercs)))
                        _y = numpy.array(_cRec_pPercs)*100
                        if   (_cRec_type == 'HF'): axs[0].plot(_x, _y, color=(1.0, 0.0, 0.0, 0.2), linestyle='-', linewidth=1); axs[2].plot(_x, _y, color=(1.0, 0.0, 0.0, 0.2), linestyle='-', linewidth=1)
                        elif (_cRec_type == 'LF'): axs[1].plot(_x, _y, color=(0.0, 1.0, 0.0, 0.2), linestyle='-', linewidth=1); axs[2].plot(_x, _y, color=(0.0, 1.0, 0.0, 0.2), linestyle='-', linewidth=1)
                    matplotlib.pyplot.show()
            else: 
                print(f"  - [{_index+1}/{len(_CYCLEDATATOPROCESS)}] '{_cdtp['cycleData']}'\n"\
                     +f"   * DATA DOES NOT EXIST")
        #[3]: Results Generation
        print(" * Grid Search Process")
        for _index, _cdtp in enumerate(_CYCLEDATATOPROCESS):
            if (_index in _processResults): 
                #File Open
                _file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles', _cdtp['cycleData']), 'r')
                _cRecs = json.loads(_file.read())
                _file.close()
                #eFunction Initialization & Data Load
                _t_0 = time.perf_counter_ns()
                _eFunction = _EXITFUNCTIONS[_cdtp['exitFunctionType']](entryMode = _cdtp['entryMode'], isSeeker = True, leverage = _cdtp['leverage'])
                if   (_PARAMETERTEST['entryMode'] == 'SHORT'): _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'HF')]
                elif (_PARAMETERTEST['entryMode'] == 'LONG'):  _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'LF')]
                _eFunction.loadDataSet(dataSet = preprocessCycleData(cycleRecords = _cycleRecs))
                _eFunction.initializeOptimizedParametersSeeker(begPrecisionStep = 3,
                                                               maxPrecisionStep = 5,
                                                               pivotParams      = None)
                _t_1 = time.perf_counter_ns()
                print(f"  - eFunction Initialization Complete! <{(_t_1-_t_0)/1e9:.3f} s>")
                #Computation
                print(f"  - Seeking eFunction Optimized Parameters...")
                _firstPrint = True
                while (True):
                    _moreToGo = _eFunction.performSeeker()
                    _cpps = _eFunction.getCurrentPivotParams()
                    _cppr = _eFunction.getCurrentPivotParamsResult()
                    _bestResults = {'functionParams': _cpps, 
                                    'finalBalance':   _cppr['finalBalance']}
                    if (_firstPrint == False): removeConsoleLines(nLinesToRemove = 2); _firstPrint = False
                    _firstPrint = False
                    print(f"   * Params:        {_bestResults['functionParams']}")
                    print(f"   * Final Balance: {_bestResults['finalBalance']:.8f}")
                    _processResults[_index]['records'].append(_bestResults)
                    if (_moreToGo == False): break
                #Result Save
                _processResults[_index]['final'] = _bestResults
                print(f"  - eFunction Optimized Parameters Seeking Process Complete!")
        #[4]: Results Save
        _resultData = {'rCode':     _rCode,
                       'time':      datetime.datetime.fromtimestamp(_PROCESSBEGINTIME).strftime("%Y/%m/%d %H:%M:%S"),
                       'cycleData': _CYCLEDATATOPROCESS,
                       'results':   _processResults}
        _resultData_path = os.path.join(_thisResult_path, f"{_rCode}_result.txt")
        with open(_resultData_path, "w") as f: f.write(json.dumps(_resultData))
        print(" * Grid Search Result Saved")
        #Finally
        print(termcolor.colored("[Grid Search Complete]\n", 'light_blue'))
        _RCODETOREAD = _rCode

    #[2]: Read Data
    if (_RCODETOREAD is not None):
        print(termcolor.colored("[Grid Search Result Read]", 'light_blue'))
        #Load Result Data
        _result_path     = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', _RCODETOREAD)
        _resultData_path = os.path.join(_result_path, f"{_RCODETOREAD}_result.txt")
        with open(_resultData_path, 'r') as f: _resultData = json.loads(f.read())
        #Result Data Display
        print(f"  * Result Code: {_resultData['rCode']}")
        print(f"  * Time:        {_resultData['time']}")
        print(f"  * Results")
        for _targetIndex in _resultData['results']:
            _cycleData = _resultData['cycleData'][int(_targetIndex)]
            _result    = _resultData['results'][_targetIndex]
            print(f"    * Cycle Data: {_cycleData['cycleData']}")
            print(f"     - Exit Function Type: {_cycleData['exitFunctionType']}")
            print(f"     - Entry Mode:         {_cycleData['entryMode']}")
            print(f"     - Leverage:           {_cycleData['leverage']}")
            print(f"     - Best Result")
            print(f"      1. Function Params: {_result['final']['functionParams']}")
            print(f"      2. Final Balance:   {_result['final']['finalBalance']:.8f}")
            #Cycle Data
            _file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles', _cycleData['cycleData']), 'r')
            _cRecs = json.loads(_file.read())
            _file.close()
            #eFunction Initialization & Data Load
            _eFunction = _EXITFUNCTIONS[_cycleData['exitFunctionType']](entryMode = _cycleData['entryMode'], isSeeker = False, leverage = _cycleData['leverage'])
            if   (_PARAMETERTEST['entryMode'] == 'SHORT'): _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'HF')]
            elif (_PARAMETERTEST['entryMode'] == 'LONG'):  _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'LF')]
            _eFunction.loadRQPFunctionParamsSet(rqpFunctionParamsSet = [_seekerRecord['functionParams'] for _seekerRecord in _result['records']])
            _eFunction.loadDataSet(dataSet = preprocessCycleData(cycleRecords = _cycleRecs))
            _nSeekerRecords = len(_result['records'])
            #Outputs
            _outputs       = _eFunction.processBatch()[:_nSeekerRecords,:len(_cycleRecs)]
            _cumProd       = torch.cumprod(input = _outputs, dim = 1)
            _alpha_ideal   = (torch.exp(torch.log(_cumProd[-1,-1].clamp(min = 0))/(_outputs.size(dim=1)-1))-1)
            _cumProd_ideal = torch.pow(1+_alpha_ideal, exponent = torch.arange(start = 0, end = _outputs.size(dim=1), step = 1, dtype = _TORCHDTYPE, device = 'cpu'))
            for _ in range (_nSeekerRecords-1): matplotlib.pyplot.plot(_cumProd[_,:], color=(0.0, 1.0, 0.5, round((_+1)/_nSeekerRecords*0.5+0.5,3)), linestyle='-', linewidth=1)
            matplotlib.pyplot.plot(_cumProd[-1,:], color=(0.0, 0.5, 1.0, 1.0), linestyle='-', linewidth=2)
            matplotlib.pyplot.plot(_cumProd_ideal, color=(1.0, 0.4, 0.0, 1.0), linestyle='-', linewidth=2)
            matplotlib.pyplot.title(f"Best Result Cumulative Product Profit - {_PARAMETERTEST['entryMode']}")
            matplotlib.pyplot.show()
        print(termcolor.colored("[Grid Search Result Read Complete]", 'light_blue'))

    #[3]: Parameter Test
    if (_PARAMETERTEST is not None):
        print(termcolor.colored("[PARAMETER TEST]", 'light_blue'))
        #Cycle Data
        _file = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cycles', _PARAMETERTEST['cycleData']), 'r')
        _cRecs = json.loads(_file.read())
        _file.close()
        #eFunction Initialization
        _eFunction = _EXITFUNCTIONS[_PARAMETERTEST['exitFunctionType']](entryMode = _PARAMETERTEST['entryMode'], isSeeker = False, leverage = _PARAMETERTEST['leverage'])
        if   (_PARAMETERTEST['entryMode'] == 'SHORT'): _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'HF')]
        elif (_PARAMETERTEST['entryMode'] == 'LONG'):  _cycleRecs = [_cRec for _cRec in _cRecs if (_cRec['type'] == 'LF')]
        _eFunction.loadRQPFunctionParamsSet(rqpFunctionParamsSet = [_PARAMETERTEST['params'],])
        _eFunction.loadDataSet(dataSet = preprocessCycleData(cycleRecords = _cycleRecs))
        #Outputs
        _outputs = _eFunction.processBatch()[0,:len(_cycleRecs)]
        _cumProd = torch.cumprod(input = _outputs, dim = 0)
        _alpha_ideal   = (torch.exp(torch.log(_cumProd[-1].clamp(min = 0))/(_outputs.size(dim=0)-1))-1)[...,None]
        _cumProd_ideal = torch.pow(1+_alpha_ideal, exponent = torch.arange(start = 0, end = _outputs.size(dim=0), step = 1, dtype = _TORCHDTYPE, device = 'cpu'))
        #Matplot Drawing
        matplotlib.pyplot.plot(_cumProd)
        matplotlib.pyplot.plot(_cumProd_ideal)
        matplotlib.pyplot.title(f"Best Result Cumulative Product Profit - {_PARAMETERTEST['entryMode']}")
        matplotlib.pyplot.show()
        print(termcolor.colored("[PARAMETER TEST COMPLETE]", 'light_blue'))

    print(termcolor.colored("\n<RQP MAP PARAMETERS GRID SEARCH PROCESS COMPLETE>", 'light_green'))
    while (True): time.sleep(1)