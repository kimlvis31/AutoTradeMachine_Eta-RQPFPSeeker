import time
import numpy
import datetime
import json
import os
import matplotlib
import matplotlib.pyplot
import termcolor
import sys
import math
import config

from exitFunction_base import exitFunction

def removeConsoleLines(nLinesToRemove: int) -> None:
    for _ in range (nLinesToRemove): 
        sys.stdout.write("\x1b[1A\x1b[2K")
        sys.stdout.flush()

if __name__ == "__main__":
    print(termcolor.colored("<RQP MAP PARAMETERS SEEKER PROCESS>\n", 'light_green'))

    _PROCESSBEGINTIME = int(time.time())

    #[1]: Data Folders Setup
    _path_folder_ppips   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ppips')
    _path_folder_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if (os.path.isdir(_path_folder_ppips)   == False): os.mkdir(_path_folder_ppips)
    if (os.path.isdir(_path_folder_results) == False): os.mkdir(_path_folder_results)

    #[2]: Read Configuration
    PARAMETERTEST  = None
    PPIPSTOPROCESS = None
    RCODETOREAD    = None
    MODE           = config.MODE
    if   MODE == 'TEST': PARAMETERTEST  = config.PARAMETERTEST
    elif MODE == 'SEEK': PPIPSTOPROCESS = config.PPIPSTOPROCESS
    elif MODE == 'READ': RCODETOREAD    = config.RCODETOREAD

    #[3]: Parameter Test
    if (PARAMETERTEST is not None):
        print(termcolor.colored("[PARAMETER TEST]", 'light_blue'))

        #Directories
        _dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ppips')
        _dir_file_descriptor = os.path.join(_dir_file_base, PARAMETERTEST['ppips']+'_descriptor.json')
        _dir_file_data       = os.path.join(_dir_file_base, PARAMETERTEST['ppips']+'_data.npy')

        #eFunction Initialization
        _eFunction = exitFunction(modelName  = PARAMETERTEST['exitFunctionType'],
                                  isSeeker   = False, 
                                  leverage   = PARAMETERTEST['leverage'], 
                                  pslReentry = PARAMETERTEST['pslReentry'])
        _eFunction.preprocessData(data = numpy.load(_dir_file_data))

        #eFunction Processing
        (
            _balance_wallet_history, 
            _balance_margin_history, 
            _balance_bestFit_history,
            _balance_bestFit_growthRates,
            _balance_bestFit_volatilities,
        ) = _eFunction.performOnParams(params = [PARAMETERTEST['params'],])

        #Matplot Drawing
        matplotlib.pyplot.plot(_balance_wallet_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1)
        matplotlib.pyplot.plot(_balance_margin_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1)
        matplotlib.pyplot.plot(_balance_bestFit_history[0,:].cpu(), color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=1)

        #Summary
        _balance_final = _balance_wallet_history[0,-1].item()
        _growthRate_interval = _balance_bestFit_growthRates[0].item()
        _growthRate_daily    = math.exp(_growthRate_interval*96)-1
        _growthRate_monthly  = math.exp(_growthRate_daily   *30.4167)-1
        _volatility = _balance_bestFit_volatilities[0].item()
        _volatility_tMin_997 = math.exp(-_volatility*3)-1
        _volatility_tMax_997 = math.exp( _volatility*3)-1
        print(f" * Final Balance: {_balance_final:.8f}")
        if   (_growthRate_daily < 0):  print(f" * Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", 'light_red')} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", 'light_red')} [Monthly]")
        elif (_growthRate_daily == 0): print(f" * Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", None)} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", None)} [Monthly]")
        else:                          print(f" * Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"+{_growthRate_daily*100:.3f} %", 'light_green')} [Daily] / {termcolor.colored(f"+{_growthRate_monthly*100:.3f} %", 'light_green')} [Monthly]")
        print(f" * Volatility:    {_volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f"{_volatility_tMin_997*100:.3f} %", 'light_magenta')} / {termcolor.colored(f"+{_volatility_tMax_997*100:.3f} %", 'light_blue')}]")

        #Matplot Show
        matplotlib.pyplot.title(f"[PARAMETER TEST] Wallet & Margin Balance History")
        matplotlib.pyplot.show()

        #Finally
        print(termcolor.colored("[PARAMETER TEST COMPLETE]", 'light_blue'))

    #[4]: Seeker
    if (PPIPSTOPROCESS is not None):
        print(termcolor.colored("[Seeker]", 'light_blue'))

        #[4-1]: Process Preparation
        if (True):
            #Analysis Code
            _rCode = f"rqpfpResult_{int(_PROCESSBEGINTIME)}"
            print(f" * Result Code: {_rCode}")

            #Results Buffer
            _processes = dict()
            for _index, _ptp in enumerate(PPIPSTOPROCESS):
                #Files Existence Check
                _dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ppips')
                _dir_file_descriptor = os.path.join(_dir_file_base, _ptp['ppips']+'_descriptor.json')
                _dir_file_data       = os.path.join(_dir_file_base, _ptp['ppips']+'_data.npy')
                if not os.path.isfile(_dir_file_descriptor) or not (_dir_file_data): continue

                #Descriptor Read
                with open(_dir_file_descriptor, 'r') as _f: 
                    _descriptor = json.loads(_f.read())

                #Processes
                _processes[_index] = {'dir_file_descriptor': _dir_file_descriptor,
                                      'dir_file_data':       _dir_file_data,
                                      'descriptor': _descriptor.copy(),
                                      'bestResult': None,
                                      'records':    list()}
              
        #[4-2]: Results Generation
        if (True):
            print(" * Seeker Process")
            for _pIndex in _processes:
                _ptp = PPIPSTOPROCESS[_pIndex]
                print(f"  [{_pIndex+1} / {len(_processes)}] <PPIPS - '{_ptp['ppips']}'>")
                #Directories
                _dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ppips')
                _dir_file_descriptor = os.path.join(_dir_file_base, _ptp['ppips']+'_descriptor.json')
                _dir_file_data       = os.path.join(_dir_file_base, _ptp['ppips']+'_data.npy')

                #eFunction Initialization & Data Load
                _eFunction = exitFunction(modelName          = _ptp['exitFunctionType'],
                                          isSeeker           = True, 
                                          leverage           = _ptp['leverage'],
                                          pslReentry         = _ptp['pslReentry'],
                                          parameterBatchSize = _ptp['parameterBatchSize'])
                print(f"    - Preprocessing PPIPS...")
                _t_0 = time.perf_counter_ns()
                _eFunction.preprocessData(data = numpy.load(_dir_file_data))
                _t_1 = time.perf_counter_ns()
                print(f"    - PPIPS Preprocessing Complete! <{(_t_1-_t_0)/1e6:.3f} ms>")
                asp = _eFunction.initializeSeeker(paramConfig          = _ptp['paramConfig'], 
                                                  nSeekerPoints        = _ptp['nSeekerPoints'],
                                                  nRepetition          = _ptp['nRepetition'],
                                                  learningRate         = _ptp['learningRate'],
                                                  deltaRatio           = _ptp['deltaRatio'],
                                                  beta_velocity        = _ptp['beta_velocity'],
                                                  beta_momentum        = _ptp['beta_momentum'],
                                                  repopulationRatio    = _ptp['repopulationRatio'],
                                                  repopulationInterval = _ptp['repopulationInterval'],
                                                  scoring              = _ptp['scoring'], 
                                                  scoringSamples       = _ptp['scoringSamples'], 
                                                  terminationThreshold = _ptp['terminationThreshold'], 
                                                  )
                print(f"    - eFunction Initialization Complete!")
                if asp['nSeekerPoints']        != _ptp['nSeekerPoints']:        print(f"      - Number of Seeker Points: {_ptp['nSeekerPoints']} -> {asp['nSeekerPoints']}")
                else:                                                           print(f"      - Number of Seeker Points: {asp['nSeekerPoints']}")
                if asp['nRepetition']          != _ptp['nRepetition']:          print(f"      - Number of Repetition:    {_ptp['nRepetition']} -> {asp['nRepetition']}")
                else:                                                           print(f"      - Number of Repetition:    {asp['nRepetition']}")
                if asp['learningRate']         != _ptp['learningRate']:         print(f"      - Learning Rate:           {_ptp['learningRate']} -> {asp['learningRate']}")
                else:                                                           print(f"      - Learning Rate:           {asp['learningRate']}")
                if asp['deltaRatio']           != _ptp['deltaRatio']:           print(f"      - Delta Ratio:             {_ptp['deltaRatio']} -> {asp['deltaRatio']}")
                else:                                                           print(f"      - Delta Ratio:             {asp['deltaRatio']}")
                if asp['beta_velocity']        != _ptp['beta_velocity']:        print(f"      - Velocity Beta:           {_ptp['beta_velocity']} -> {asp['beta_velocity']}")
                else:                                                           print(f"      - Velocity Beta:           {asp['beta_velocity']}")
                if asp['beta_momentum']        != _ptp['beta_momentum']:        print(f"      - Momentum Beta:           {_ptp['beta_momentum']} -> {asp['beta_momentum']}")
                else:                                                           print(f"      - Momentum Beta:           {asp['beta_momentum']}")
                if asp['repopulationRatio']    != _ptp['repopulationRatio']:    print(f"      - Repopulation Ratio:      {_ptp['repopulationRatio']} -> {asp['repopulationRatio']}")
                else:                                                           print(f"      - Repopulation Ratio:      {asp['repopulationRatio']}")
                if asp['repopulationInterval'] != _ptp['repopulationInterval']: print(f"      - Repopulation Interval:   {_ptp['repopulationInterval']} -> {asp['repopulationInterval']}")
                else:                                                           print(f"      - Repopulation Interval:   {asp['repopulationInterval']}")
                if asp['scoring']              != _ptp['scoring']:              print(f"      - Scoring:                 {_ptp['scoring']} -> {asp['scoring']}")
                else:                                                           print(f"      - Scoring:                 {asp['scoring']}")
                if asp['terminationThreshold'] != _ptp['terminationThreshold']: print(f"      - Termination Threshold:   {_ptp['terminationThreshold']} -> {asp['terminationThreshold']}")
                else:                                                           print(f"      - Termination Threshold:   {asp['terminationThreshold']}")
                _ptp['nSeekerPoints']        = asp['nSeekerPoints']
                _ptp['nRepetition']          = asp['nRepetition']
                _ptp['learningRate']         = asp['learningRate']
                _ptp['deltaRatio']           = asp['deltaRatio']
                _ptp['beta_velocity']        = asp['beta_velocity']
                _ptp['beta_momentum']        = asp['beta_momentum']
                _ptp['repopulationRatio']    = asp['repopulationRatio']
                _ptp['repopulationInterval'] = asp['repopulationInterval']
                _ptp['scoring']              = asp['scoring']
                _ptp['scoringSamples']       = asp['scoringSamples']
                _ptp['terminationThreshold'] = asp['terminationThreshold']

                #Seeker
                print(f"    - Seeking eFunction Optimized Parameters...")
                nPrintedLines = None
                complete      = False
                repIndex_last = None
                bestResult    = None
                try:
                    while not(complete):
                        #Processing
                        complete, _repetitionIndex, _step, _bestResult = _eFunction.runSeeker()
                        _bestResult = {'repetitionIndex': _repetitionIndex,
                                       'functionParams':  _bestResult[0], 
                                       'finalBalance':    _bestResult[1],
                                       'growthRate':      _bestResult[2],
                                       'volatility':      _bestResult[3],
                                       'score':           _bestResult[4]}
                        #Best Result Check
                        if (bestResult is None) or (bestResult['score'] < _bestResult['score']): 
                            bestResult = _bestResult
                            _processes[_pIndex]['bestResult'] = bestResult
                            _processes[_pIndex]['records'].append(_bestResult)
                        #Progress Print
                        if (nPrintedLines is not None): removeConsoleLines(nLinesToRemove = nPrintedLines)
                        nPrintedLines = 6
                        print(f"      - Progress:      <Repetition: {_repetitionIndex+1}/{_ptp['nRepetition']}> <Step: {_step}>")
                        print(f"      - Params:        {bestResult['functionParams']}")
                        print(f"      - Final Balance: {bestResult['finalBalance']:.8f}")
                        _growthRate_interval = bestResult['growthRate']
                        _growthRate_daily    = math.exp(_growthRate_interval*96)-1
                        _growthRate_monthly  = math.exp(_growthRate_daily   *30.4167)-1
                        _volatility = bestResult['volatility']
                        _volatility_tMin_997 = math.exp(-_volatility*3)-1
                        _volatility_tMax_997 = math.exp( _volatility*3)-1
                        if   (_growthRate_daily < 0):  print(f"      - Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", 'light_red')} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", 'light_red')} [Monthly]")
                        elif (_growthRate_daily == 0): print(f"      - Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", None)} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", None)} [Monthly]")
                        else:                          print(f"      - Growth Rate:   {_growthRate_interval:.8f} / {termcolor.colored(f"+{_growthRate_daily*100:.3f} %", 'light_green')} [Daily] / {termcolor.colored(f"+{_growthRate_monthly*100:.3f} %", 'light_green')} [Monthly]")
                        print(f"      - Volatility:    {_volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f"{_volatility_tMin_997*100:.3f} %", 'light_magenta')} / {termcolor.colored(f"+{_volatility_tMax_997*100:.3f} %", 'light_blue')}]")
                        print(f"      - Score:         {bestResult['score']:.8f}")
                except KeyboardInterrupt:
                    print("    - Keyboard Interruption Detected, Terminating...")

                print(f"    - eFunction Optimized Parameters Seeking Process Complete!")

        #[4-3]: Results Save
        if (True):
            #Folder Generation
            _thisResult_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', _rCode)
            os.mkdir(_thisResult_path)

            #Results Save
            _pResults = dict()
            for _pIndex in _processes:
                _process = _processes[_pIndex]
                _pResults[_pIndex] = {'genTime_ns':     _process['descriptor']['genTime_ns'],
                                      'simulationCode': _process['descriptor']['simulationCode'],
                                      'positionSymbol': _process['descriptor']['positionSymbol'],
                                      'bestResult': _process['bestResult'],
                                      'records':    _process['records']}
            _resultData = {'rCode':   _rCode,
                           'time':    datetime.datetime.fromtimestamp(_PROCESSBEGINTIME).strftime("%Y/%m/%d %H:%M:%S"),
                           'ppips':   PPIPSTOPROCESS,
                           'results': _pResults}
            _resultData_path = os.path.join(_thisResult_path, f"{_rCode}_result.json")
            with open(_resultData_path, "w") as f: f.write(json.dumps(_resultData))
            print(" * Seeker Result Saved")

            #Finally
            print(termcolor.colored("[Seeker Complete]\n", 'light_blue'))
            RCODETOREAD = _rCode

    #[5]: Read Data
    if (RCODETOREAD is not None):
        print(termcolor.colored("[Seeker Result Read]", 'light_blue'))

        #Directories
        _dir_result_folder   = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', RCODETOREAD)
        _dir_result_file     = os.path.join(_dir_result_folder, f"{RCODETOREAD}_result.json")
        _dir_ppips_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ppips')

        #Result Data
        with open(_dir_result_file, 'r') as _f: _resultData = json.loads(_f.read())
        print(f" * Result Code: {_resultData['rCode']}")
        print(f" * Time:        {_resultData['time']}")
        print(f" * Results:")

        #Result Data Display
        for _pIndex_str in _resultData['results']:
            _ptp     = _resultData['ppips'][int(_pIndex_str)]
            _pResult = _resultData['results'][_pIndex_str]

            #PPIPS Directories
            _dir_ppips_file_descriptor = os.path.join(_dir_ppips_file_base, _ptp['ppips']+'_descriptor.json')
            _dir_ppips_file_data       = os.path.join(_dir_ppips_file_base, _ptp['ppips']+'_data.npy')

            #Process Descriptor
            print(f"  [{int(_pIndex_str)+1} / {len(_resultData['results'])}] <PPIPS - '{_ptp['ppips']}'>")
            print(f"    - Exit Function Type:      {_ptp['exitFunctionType']}")
            print(f"    - Leverage:                {_ptp['leverage']}")
            print(f"    - PSL Re-Entry:            {_ptp['pslReentry']}")
            print(f"    - Parameter Batch Size:    {_ptp['parameterBatchSize']}")
            print(f"    - Parameter Configuration: {_ptp['paramConfig']}")
            print(f"    - Number of Seeker Points: {_ptp['nSeekerPoints']}")
            print(f"    - Number of Repetition:    {_ptp['nRepetition']}")
            print(f"    - Learning Rate:           {_ptp['learningRate']}")
            print(f"    - Delta Ratio:             {_ptp['deltaRatio']}")
            print(f"    - Velocity Beta:           {_ptp['beta_velocity']}")
            print(f"    - Momentum Beta:           {_ptp['beta_momentum']}")
            print(f"    - Repopulation Ratio:      {_ptp['repopulationRatio']}")
            print(f"    - Repopulation Interval:   {_ptp['repopulationInterval']}")
            print(f"    - Scoring:                 {_ptp['scoring']}")
            print(f"    - Termination Threshold:   {_ptp['terminationThreshold']}")

            #PPIPS Match Check
            with open(_dir_ppips_file_descriptor, 'r') as f: _ppips_descriptor = json.loads(f.read())
            if ((_pResult['genTime_ns']     == _ppips_descriptor['genTime_ns'])     & \
                (_pResult['simulationCode'] == _ppips_descriptor['simulationCode']) & \
                (_pResult['positionSymbol'] == _ppips_descriptor['positionSymbol'])):
                print("    - PPIPS Data Match:  ", termcolor.colored("TRUE", 'light_green'))
            else:
                print("    - PPIPS Data Match:  ", termcolor.colored("FAlSE", 'light_green'))

            #Seeker Best Result
            print(f"    - Seeker Best Result:")
            print(f"      - Function Params: {_pResult['bestResult']['functionParams']}")
            print(f"      - Final Balance:   {_pResult['bestResult']['finalBalance']:.8f}")
            _growthRate_interval = _pResult['bestResult']['growthRate']
            _growthRate_daily    = math.exp(_growthRate_interval*96)-1
            _growthRate_monthly  = math.exp(_growthRate_daily   *30.4167)-1
            _volatility = _pResult['bestResult']['volatility']
            _volatility_tMin_997 = math.exp(-_volatility*3)-1
            _volatility_tMax_997 = math.exp( _volatility*3)-1
            if   (_growthRate_daily < 0):  print(f"      - Growth Rate:     {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", 'light_red')} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", 'light_red')} [Monthly]")
            elif (_growthRate_daily == 0): print(f"      - Growth Rate:     {_growthRate_interval:.8f} / {termcolor.colored(f"{_growthRate_daily*100:.3f} %", None)} [Daily] / {termcolor.colored(f"{_growthRate_monthly*100:.3f} %", None)} [Monthly]")
            else:                          print(f"      - Growth Rate:     {_growthRate_interval:.8f} / {termcolor.colored(f"+{_growthRate_daily*100:.3f} %", 'light_green')} [Daily] / {termcolor.colored(f"+{_growthRate_monthly*100:.3f} %", 'light_green')} [Monthly]")
            print(f"      - Volatility:      {_volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f"{_volatility_tMin_997*100:.3f} %", 'light_magenta')} / {termcolor.colored(f"+{_volatility_tMax_997*100:.3f} %", 'light_blue')}]")
            print(f"      - Score:           {_pResult['bestResult']['score']:.8f}")

            #eFunction Initialization
            _eFunction = exitFunction(modelName  = _ptp['exitFunctionType'],
                                      isSeeker   = False, 
                                      leverage   = _ptp['leverage'],
                                      pslReentry = _ptp['pslReentry'])
            print(f"    - Preprocessing PPIPS...")
            _t_0 = time.perf_counter_ns()
            _eFunction.preprocessData(data = numpy.load(_dir_ppips_file_data))
            _t_1 = time.perf_counter_ns()
            print(f"    - PPIPS Preprocessing Complete! <{(_t_1-_t_0)/1e6:.3f} ms>")

            #eFunction Processing
            print(f"    - Reconstructing Seeker Records...")
            _t_0 = time.perf_counter_ns()
            _recs    = sorted(_pResult['records'], key = lambda x: x['score'], reverse = True)[:100]
            _params  = [_rec['functionParams'] for _rec in _recs]
            _nParams = len(_params)
            (
            _balance_wallet_history, 
            _balance_margin_history, 
            _balance_bestFit_history,
            _balance_bestFit_growthRates,
            _balance_bestFit_volatilities,
            ) = _eFunction.performOnParams(params = _params)
            _t_1 = time.perf_counter_ns()
            print(f"    - Seeker Records Reconstruction Complete! <{(_t_1-_t_0)/1e6:.3f} ms>")

            #Matplot Drawing
            matplotlib.pyplot.plot(_balance_wallet_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1, zorder = 4)
            matplotlib.pyplot.plot(_balance_margin_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1, zorder = 3)
            matplotlib.pyplot.plot(_balance_bestFit_history[0,:].cpu(), color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=2, zorder = 5)
            for _rIndex in range (1, _nParams):
                matplotlib.pyplot.plot(_balance_wallet_history[_rIndex,:].cpu(), color=(0.5, 0.8, 0.0, round((_rIndex+1)/_nParams*0.10+0.10,3)), linestyle='solid',  linewidth=0.5, zorder = 2)
                matplotlib.pyplot.plot(_balance_margin_history[_rIndex,:].cpu(), color=(0.5, 0.8, 0.0, round((_rIndex+1)/_nParams*0.05+0.05,3)), linestyle='dashed', linewidth=0.5, zorder = 1)

            #Matplot Show
            matplotlib.pyplot.title(f"[RESULT READ] Wallet & Margin Balance History")
            matplotlib.pyplot.show()

            #Line Skip
            print()
        print(termcolor.colored("[Seeker Result Read Complete]", 'light_blue'))

    print(termcolor.colored("\n<RQP MAP PARAMETERS SEEKER PROCESS COMPLETE>", 'light_green'))
    while (True): time.sleep(100)