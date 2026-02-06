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
from rich.console import Console as rConsole
from rich.live    import Live    as rLive

from exitFunction_base import exitFunction

if __name__ == "__main__":
    print(termcolor.colored("<RQP MAP PARAMETERS SEEKER PROCESS>\n", 'light_green'))

    PROCESSBEGINTIME = int(time.time())

    #[1]: Data Folders Setup
    path_folder_analysisData = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
    path_folder_results      = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.isdir(path_folder_analysisData): os.mkdir(path_folder_analysisData)
    if not os.path.isdir(path_folder_results):      os.mkdir(path_folder_results)

    #[2]: Read Configuration
    PARAMETERTEST = None
    SEEKERTARGETS = None
    RCODETOREAD   = None
    MODE          = config.MODE
    if   MODE == 'TEST': PARAMETERTEST = config.PARAMETERTEST
    elif MODE == 'SEEK': SEEKERTARGETS = config.SEEKERTARGETS
    elif MODE == 'READ': RCODETOREAD   = config.RCODETOREAD

    #[3]: Parameter Test
    if (PARAMETERTEST is not None):
        print(termcolor.colored("[PARAMETER TEST]", 'light_blue'))

        #[3-1]: Directories
        dir_file_base       = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
        dir_file_descriptor = os.path.join(dir_file_base, PARAMETERTEST['analysisData']+'_descriptor.json')
        dir_file_data       = os.path.join(dir_file_base, PARAMETERTEST['analysisData']+'_data.npy')

        #[3-2]: Data Load
        with open(dir_file_descriptor, 'r') as _f: 
            descriptor = json.loads(_f.read())
        linearizedAnalysis = numpy.load(dir_file_data)

        #[3-3]: eFunction Initialization
        eFunction = exitFunction(modelName  = PARAMETERTEST['exitFunctionType'],
                                 isSeeker   = False, 
                                 leverage   = PARAMETERTEST['leverage'], 
                                 pslReentry = PARAMETERTEST['pslReentry'])
        eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])

        #[3-4]: eFunction Processing
        (
            balance_wallet_history, 
            balance_margin_history, 
            balance_bestFit_history,
            balance_bestFit_growthRates,
            balance_bestFit_volatilities,
            nTrades
        ) = eFunction.performOnParams(params = [PARAMETERTEST['params'],])

        #[3-5]: Matplot Drawing
        matplotlib.pyplot.plot(balance_wallet_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1)
        matplotlib.pyplot.plot(balance_margin_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1)
        matplotlib.pyplot.plot(balance_bestFit_history[0,:].cpu(), color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=1)

        #[3-6]: Summary
        balance_final = balance_wallet_history[0,-1].item()
        growthRate_interval = balance_bestFit_growthRates[0].item()
        growthRate_daily    = math.exp(growthRate_interval*96)        -1
        growthRate_monthly  = math.exp(growthRate_interval*96*30.4167)-1
        volatility = balance_bestFit_volatilities[0].item()
        volatility_tMin_997 = math.exp(-volatility*3)-1
        volatility_tMax_997 = math.exp( volatility*3)-1
        print(f" * Final Balance: {balance_final:.8f}")
        if   (growthRate_daily < 0):  print(f" * Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', 'light_red')} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', 'light_red')} [Monthly]")
        elif (growthRate_daily == 0): print(f" * Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', None)} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', None)} [Monthly]")
        else:                         print(f" * Growth Rate:   {growthRate_interval:.8f} / {termcolor.colored(f'+{growthRate_daily*100:.3f} %', 'light_green')} [Daily] / {termcolor.colored(f'+{growthRate_monthly*100:.3f} %', 'light_green')} [Monthly]")
        print(f" * Volatility:    {volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f'{volatility_tMin_997*100:.3f} %', 'light_magenta')} / {termcolor.colored(f'+{volatility_tMax_997*100:.3f} %', 'light_blue')}]")
        print(f" * nTrades:       {int(nTrades[0].item())}")

        #[3-7]: Matplot Show
        matplotlib.pyplot.title(f"[PARAMETER TEST] Wallet & Margin Balance History")
        matplotlib.pyplot.show()

        #[3-8]: Finally
        print(termcolor.colored("[PARAMETER TEST COMPLETE]", 'light_blue'))

    #[4]: Seeker
    if (SEEKERTARGETS is not None):
        print(termcolor.colored("[Seeker]", 'light_blue'))

        #[4-1]: Process Preparation
        #---[4-1-1]: Analysis Code
        rCode = f"rqpfpResult_{int(PROCESSBEGINTIME)}"
        print(f" * Result Code: {rCode}")

        #---[4-1-2]: Results Buffer
        processes = dict()
        for pIndex, st in enumerate(SEEKERTARGETS):
            #[4-1-2-1]: Files Existence Check
            dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
            dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
            dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')
            if not os.path.isfile(dir_file_descriptor) or not os.path.isfile(dir_file_data): continue

            #[4-1-2-2]: Descriptor Read
            dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
            dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')
            with open(dir_file_descriptor, 'r') as f: 
                descriptor = json.loads(f.read())

            #[4-1-2-3]: Processes
            processes[pIndex] = {'descriptor': descriptor.copy(),
                                 'bestResult': None,
                                 'records':    list()}
              
        #[4-2]: Results Generation
        print(" * Seeker Process")
        for pIndex in processes:
            process = processes[pIndex]
            st      = SEEKERTARGETS[pIndex]
            print(f"  [{pIndex+1} / {len(processes)}] <ANALYSIS DATA - '{st['analysisData']}'>")
            #Directories
            dir_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')
            dir_file_descriptor = os.path.join(dir_file_base, st['analysisData']+'_descriptor.json')
            dir_file_data       = os.path.join(dir_file_base, st['analysisData']+'_data.npy')

            #Data Load
            with open(dir_file_descriptor, 'r') as f: 
                descriptor = json.loads(f.read())
            linearizedAnalysis = numpy.load(dir_file_data)

            #eFunction Initialization
            eFunction = exitFunction(modelName          = st['exitFunctionType'],
                                     isSeeker           = True, 
                                     leverage           = st['leverage'],
                                     pslReentry         = st['pslReentry'],
                                     parameterBatchSize = st['parameterBatchSize'])
            print(f"    - Preprocessing Analysis Data...")
            t_0 = time.perf_counter_ns()
            eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])
            t_1 = time.perf_counter_ns()
            print(f"    - Analysis Data Preprocessing Complete! <{(t_1-t_0)/1e6:.3f} ms>")
            asp = eFunction.initializeSeeker(paramConfig              = st['paramConfig'], 
                                             nSeekerPoints            = st['nSeekerPoints'],
                                             nRepetition              = st['nRepetition'],
                                             learningRate             = st['learningRate'],
                                             deltaRatio               = st['deltaRatio'],
                                             beta_velocity            = st['beta_velocity'],
                                             beta_momentum            = st['beta_momentum'],
                                             repopulationRatio        = st['repopulationRatio'],
                                             repopulationInterval     = st['repopulationInterval'],
                                             repopulationGuideRatio   = st['repopulationGuideRatio'],
                                             repopulationDecayRate    = st['repopulationDecayRate'],
                                             scoring                  = st['scoring'], 
                                             scoring_maxMDD           = st['scoring_maxMDD'],
                                             scoring_growthRateWeight = st['scoring_growthRateWeight'],
                                             scoring_growthRateScaler = st['scoring_growthRateScaler'],
                                             scoring_volatilityWeight = st['scoring_volatilityWeight'],
                                             scoring_volatilityScaler = st['scoring_volatilityScaler'],
                                             scoring_nTradesWeight    = st['scoring_nTradesWeight'],
                                             scoring_nTradesScaler    = st['scoring_nTradesScaler'],
                                             scoringSamples           = st['scoringSamples'], 
                                             terminationThreshold     = st['terminationThreshold'], 
                                            )
            print(f"    - eFunction Initialization Complete!")
            print(f"      - Exit Function Type:   {st['exitFunctionType']}")
            print(f"      - Leverage:             {st['leverage']}")
            print(f"      - PSL Re-entry:         {st['pslReentry']}")
            print(f"      - Parameter Batch Size: {st['parameterBatchSize']}")
            if asp['nSeekerPoints']            != st['nSeekerPoints']:            print(f"      - Number of Seeker Points:    {st['nSeekerPoints']} -> {asp['nSeekerPoints']}")
            else:                                                                 print(f"      - Number of Seeker Points:    {asp['nSeekerPoints']}")
            if asp['nRepetition']              != st['nRepetition']:              print(f"      - Number of Repetition:       {st['nRepetition']} -> {asp['nRepetition']}")
            else:                                                                 print(f"      - Number of Repetition:       {asp['nRepetition']}")
            if asp['learningRate']             != st['learningRate']:             print(f"      - Learning Rate:              {st['learningRate']} -> {asp['learningRate']}")
            else:                                                                 print(f"      - Learning Rate:              {asp['learningRate']}")
            if asp['deltaRatio']               != st['deltaRatio']:               print(f"      - Delta Ratio:                {st['deltaRatio']} -> {asp['deltaRatio']}")
            else:                                                                 print(f"      - Delta Ratio:                {asp['deltaRatio']}")
            if asp['beta_velocity']            != st['beta_velocity']:            print(f"      - Velocity Beta:              {st['beta_velocity']} -> {asp['beta_velocity']}")
            else:                                                                 print(f"      - Velocity Beta:              {asp['beta_velocity']}")
            if asp['beta_momentum']            != st['beta_momentum']:            print(f"      - Momentum Beta:              {st['beta_momentum']} -> {asp['beta_momentum']}")
            else:                                                                 print(f"      - Momentum Beta:              {asp['beta_momentum']}")
            if asp['repopulationRatio']        != st['repopulationRatio']:        print(f"      - Repopulation Ratio:         {st['repopulationRatio']} -> {asp['repopulationRatio']}")
            else:                                                                 print(f"      - Repopulation Ratio:         {asp['repopulationRatio']}")
            if asp['repopulationInterval']     != st['repopulationInterval']:     print(f"      - Repopulation Interval:      {st['repopulationInterval']} -> {asp['repopulationInterval']}")
            else:                                                                 print(f"      - Repopulation Interval:      {asp['repopulationInterval']}")
            if asp['repopulationGuideRatio']   != st['repopulationGuideRatio']:   print(f"      - Repopulation Guide Ratio:   {st['repopulationGuideRatio']} -> {asp['repopulationGuideRatio']}")
            else:                                                                 print(f"      - Repopulation Guide Ratio:   {asp['repopulationGuideRatio']}")
            if asp['repopulationDecayRate']    != st['repopulationDecayRate']:    print(f"      - Repopulation Decay Rate:    {st['repopulationDecayRate']} -> {asp['repopulationDecayRate']}")
            else:                                                                 print(f"      - Repopulation Decay Rate:    {asp['repopulationDecayRate']}")
            if asp['scoring']                  != st['scoring']:                  print(f"      - Scoring:                    {st['scoring']} -> {asp['scoring']}")
            else:                                                                 print(f"      - Scoring:                    {asp['scoring']}")
            if asp['scoring_maxMDD']           != st['scoring_maxMDD']:           print(f"      - Scoring Maximum MDD:        {st['scoring_maxMDD']} -> {asp['scoring_maxMDD']}")
            else:                                                                 print(f"      - Scoring Maximum MDD:        {asp['scoring_maxMDD']}")
            if asp['scoring_growthRateWeight'] != st['scoring_growthRateWeight']: print(f"      - Scoring Growth Rate Weight: {st['scoring_growthRateWeight']} -> {asp['scoring_growthRateWeight']}")
            else:                                                                 print(f"      - Scoring Growth Rate Weight: {asp['scoring_growthRateWeight']}")
            if asp['scoring_growthRateScaler'] != st['scoring_growthRateScaler']: print(f"      - Scoring Growth Rate Scaler: {st['scoring_growthRateScaler']} -> {asp['scoring_growthRateScaler']}")
            else:                                                                 print(f"      - Scoring Growth Rate Scaler: {asp['scoring_growthRateScaler']}")
            if asp['scoring_volatilityWeight'] != st['scoring_volatilityWeight']: print(f"      - Scoring Volatility Weight:  {st['scoring_volatilityWeight']} -> {asp['scoring_volatilityWeight']}")
            else:                                                                 print(f"      - Scoring Volatility Weight:  {asp['scoring_volatilityWeight']}")
            if asp['scoring_volatilityScaler'] != st['scoring_volatilityScaler']: print(f"      - Scoring Volatility Scaler:  {st['scoring_volatilityScaler']} -> {asp['scoring_volatilityScaler']}")
            else:                                                                 print(f"      - Scoring Volatility Scaler:  {asp['scoring_volatilityScaler']}")
            if asp['scoring_nTradesWeight']    != st['scoring_nTradesWeight']:    print(f"      - Scoring nTrades Weight:     {st['scoring_nTradesWeight']} -> {asp['scoring_nTradesWeight']}")
            else:                                                                 print(f"      - Scoring nTrades Weight:     {asp['scoring_nTradesWeight']}")
            if asp['scoring_nTradesScaler']    != st['scoring_nTradesScaler']:    print(f"      - Scoring nTrades Scaler:     {st['scoring_nTradesScaler']} -> {asp['scoring_nTradesScaler']}")
            else:                                                                 print(f"      - Scoring nTrades Scaler:     {asp['scoring_nTradesScaler']}")
            if asp['terminationThreshold']     != st['terminationThreshold']:     print(f"      - Termination Threshold:      {st['terminationThreshold']} -> {asp['terminationThreshold']}")
            else:                                                                 print(f"      - Termination Threshold:      {asp['terminationThreshold']}")
            st['nSeekerPoints']            = asp['nSeekerPoints']
            st['nRepetition']              = asp['nRepetition']
            st['learningRate']             = asp['learningRate']
            st['deltaRatio']               = asp['deltaRatio']
            st['beta_velocity']            = asp['beta_velocity']
            st['beta_momentum']            = asp['beta_momentum']
            st['repopulationRatio']        = asp['repopulationRatio']
            st['repopulationInterval']     = asp['repopulationInterval']
            st['scoring']                  = asp['scoring']
            st['scoring_maxMDD']           = asp['scoring_maxMDD']
            st['scoring_growthRateWeight'] = asp['scoring_growthRateWeight']
            st['scoring_growthRateScaler'] = asp['scoring_growthRateScaler']
            st['scoring_volatilityWeight'] = asp['scoring_volatilityWeight']
            st['scoring_volatilityScaler'] = asp['scoring_volatilityScaler']
            st['scoring_nTradesWeight']    = asp['scoring_nTradesWeight']
            st['scoring_nTradesScaler']    = asp['scoring_nTradesScaler']
            st['scoringSamples']           = asp['scoringSamples']
            st['terminationThreshold']     = asp['terminationThreshold']

            #Seeker
            print(f"    - Seeking eFunction Optimized Parameters...")
            console       = rConsole(highlight=False)
            complete      = False
            repIndex_last = None
            bestResult    = None
            try:
                with rLive(console = console, refresh_per_second = 4) as rl:
                    while not(complete):
                        #Processing
                        (
                            complete, 
                            repetitionIndex, 
                            step, 
                            _bestResult, 
                            t_processing_paramsSet_sim_ms,
                            t_processing_paramsSet_ppp_ms
                            ) = eFunction.runSeeker()
                        _bestResult = {'repetitionIndex': repetitionIndex,
                                       'functionParams':  _bestResult[0], 
                                       'finalBalance':    _bestResult[1],
                                       'growthRate':      _bestResult[2],
                                       'volatility':      _bestResult[3],
                                       'score':           _bestResult[4],
                                       'nTrades':         _bestResult[5]}
                        #Best Result Check
                        if (bestResult is None) or (bestResult['score'] < _bestResult['score']): 
                            bestResult = _bestResult
                            processes[pIndex]['bestResult'] = bestResult
                            processes[pIndex]['records'].append(_bestResult)
                        #Progress Print
                        growthRate_interval = bestResult['growthRate']
                        growthRate_daily    = math.exp(growthRate_interval*96)        -1
                        growthRate_monthly  = math.exp(growthRate_interval*96*30.4167)-1
                        if   (growthRate_daily < 0):  
                            grStr_color = "bright_red"
                            grStr_sign  = ""
                        elif (growthRate_daily == 0): 
                            grStr_color = "white"
                            grStr_sign  = ""
                        else:                          
                            grStr_color = "bright_green"
                            grStr_sign  = "+"
                        volatility = bestResult['volatility']
                        volatility_tMin_997 = math.exp(-volatility*3)-1
                        volatility_tMax_997 = math.exp( volatility*3)-1
                        ppstr = (f"      - Progress:                       <Repetition: {repetitionIndex+1}/{st['nRepetition']}> <Step: {step}>\n"
                                 f"      - Parameter Set Processing Speed: {t_processing_paramsSet_sim_ms*1e3:.3f} us [SIMULATION], {t_processing_paramsSet_ppp_ms*1e3:.3f} us [PRE/POST PROCESSING]\n"
                                 f"      - Params:                         {bestResult['functionParams']}\n"
                                 f"      - Final Balance:                  {bestResult['finalBalance']:.8f}\n"
                                 f"      - Growth Rate:                    [{grStr_color}]{grStr_sign}{growthRate_interval:.8f} [/]/ [{grStr_color}]{grStr_sign}{growthRate_daily*100:.3f} % [/][Daily] / [{grStr_color}]{grStr_sign}{growthRate_monthly*100:.3f} % [/][Monthly]\n"
                                 f"      - Volatility:                     {volatility:.8f} [Theoretical 99.7%: [bright_magenta]{volatility_tMin_997*100:.3f} % [/]/ [bright_cyan]{volatility_tMax_997*100:.3f} %][/]\n"
                                 f"      - Score:                          {bestResult['score']:.8f}\n"
                                 f"      - nTrades:                        {bestResult['nTrades']}"
                                )
                        rl.update(ppstr)
            except KeyboardInterrupt:
                print("    - Keyboard Interruption Detected, Terminating...")

            print(f"    - eFunction Optimized Parameters Seeking Process Complete!")

        #[4-3]: Results Save
        #---[4-3-1]: Folder Generation
        thisResult_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', rCode)
        os.mkdir(thisResult_path)

        #---[4-3-2]: Results Save
        pResults = dict()
        for pIndex in processes:
            process = processes[pIndex]
            pResults[pIndex] = {'genTime_ns':     process['descriptor']['genTime_ns'],
                                'simulationCode': process['descriptor']['simulationCode'],
                                'positionSymbol': process['descriptor']['positionSymbol'],
                                'bestResult':     process['bestResult'],
                                'records':        process['records']}
        resultData = {'rCode':         rCode,
                      'time':          datetime.datetime.fromtimestamp(PROCESSBEGINTIME).strftime("%Y/%m/%d %H:%M:%S"),
                      'seekerTargets': SEEKERTARGETS,
                      'results':       pResults}
        resultData_path = os.path.join(thisResult_path, f"{rCode}_result.json")
        with open(resultData_path, "w") as f: f.write(json.dumps(resultData))
        print(" * Seeker Result Saved")

        #---[4-3-3]: Finally
        print(termcolor.colored("[Seeker Complete]\n", 'light_blue'))
        RCODETOREAD = rCode

    #[5]: Read Data
    if (RCODETOREAD is not None):
        print(termcolor.colored("[Seeker Result Read]", 'light_blue'))

        #[5-1]: Directories
        dir_result_folder  = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', RCODETOREAD)
        dir_result_file    = os.path.join(dir_result_folder, f"{RCODETOREAD}_result.json")
        dir_data_file_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'analysisData')

        #[5-2]: Result Data
        with open(dir_result_file, 'r') as _f: resultData = json.loads(_f.read())
        print(f" * Result Code: {resultData['rCode']}")
        print(f" * Time:        {resultData['time']}")
        print(f" * Results:")

        #[5-3]: Result Data Display
        for pIndexstr in resultData['results']:
            #[5-3-1]: Instances 
            st      = resultData['seekerTargets'][int(pIndexstr)]
            pResult = resultData['results'][pIndexstr]

            #[5-3-2]: Analysis Data Directories
            dir_file_descriptor = os.path.join(dir_data_file_base, st['analysisData']+'_descriptor.json')
            dir_file_data       = os.path.join(dir_data_file_base, st['analysisData']+'_data.npy')

            #[5-3-3]: Analysis Data Load
            with open(dir_file_descriptor, 'r') as _f: 
                descriptor = json.loads(_f.read())
            linearizedAnalysis = numpy.load(dir_file_data)

            #[5-3-4]: Analysis Data Match Check
            if ((pResult['genTime_ns']     == descriptor['genTime_ns'])     & \
                (pResult['simulationCode'] == descriptor['simulationCode']) & \
                (pResult['positionSymbol'] == descriptor['positionSymbol'])):
                print("    - Analysis Data Match:  ", termcolor.colored("TRUE", 'light_green'))
            else:
                print("    - Analysis Data Match:  ", termcolor.colored("FALSE", 'light_green'))

            #[5-3-5]: Process Parameters
            print(f"  [{int(pIndexstr)+1} / {len(resultData['results'])}] <ANALYSIS DATA - '{st['analysisData']}'>")
            print(f"    - Exit Function Type:         {st['exitFunctionType']}")
            print(f"    - Leverage:                   {st['leverage']}")
            print(f"    - PSL Re-Entry:               {st['pslReentry']}")
            print(f"    - Parameter Batch Size:       {st['parameterBatchSize']}")
            print(f"    - Parameter Configuration:    {st['paramConfig']}")
            print(f"    - Number of Seeker Points:    {st['nSeekerPoints']}")
            print(f"    - Number of Repetition:       {st['nRepetition']}")
            print(f"    - Learning Rate:              {st['learningRate']}")
            print(f"    - Delta Ratio:                {st['deltaRatio']}")
            print(f"    - Velocity Beta:              {st['beta_velocity']}")
            print(f"    - Momentum Beta:              {st['beta_momentum']}")
            print(f"    - Repopulation Ratio:         {st['repopulationRatio']}")
            print(f"    - Repopulation Interval:      {st['repopulationInterval']}")
            print(f"    - Repopulation Guide Ratio:   {st['repopulationGuideRatio']}")
            print(f"    - Repopulation Decay Rate:    {st['repopulationDecayRate']}")
            print(f"    - Scoring:                    {st['scoring']}")
            print(f"    - Scoring Maximum MDD:        {st['scoring_maxMDD']}")
            print(f"    - Scoring Growth Rate Weight: {st['scoring_growthRateWeight']}")
            print(f"    - Scoring Growth Rate Scaler: {st['scoring_growthRateScaler']}")
            print(f"    - Scoring Volatility Weight:  {st['scoring_volatilityWeight']}")
            print(f"    - Scoring Volatility Scaler:  {st['scoring_volatilityScaler']}")
            print(f"    - Scoring nTrades Weight:     {st['scoring_nTradesWeight']}")
            print(f"    - Scoring nTrades Scaler:     {st['scoring_nTradesScaler']}")
            print(f"    - Termination Threshold:      {st['terminationThreshold']}")

            #[5-3-6]: Seeker Best Result
            print(f"    - Seeker Best Result:")
            print(f"      - Function Params: {pResult['bestResult']['functionParams']}")
            print(f"      - Final Balance:   {pResult['bestResult']['finalBalance']:.8f}")
            growthRate_interval = pResult['bestResult']['growthRate']
            growthRate_daily    = math.exp(growthRate_interval*96)        -1
            growthRate_monthly  = math.exp(growthRate_interval*96*30.4167)-1
            volatility = pResult['bestResult']['volatility']
            volatility_tMin_997 = math.exp(-volatility*3)-1
            volatility_tMax_997 = math.exp( volatility*3)-1
            if   (growthRate_daily < 0):  print(f"      - Growth Rate:     {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', 'light_red')} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', 'light_red')} [Monthly]")
            elif (growthRate_daily == 0): print(f"      - Growth Rate:     {growthRate_interval:.8f} / {termcolor.colored(f'{growthRate_daily*100:.3f} %', None)} [Daily] / {termcolor.colored(f'{growthRate_monthly*100:.3f} %', None)} [Monthly]")
            else:                         print(f"      - Growth Rate:     {growthRate_interval:.8f} / {termcolor.colored(f'+{growthRate_daily*100:.3f} %', 'light_green')} [Daily] / {termcolor.colored(f'+{growthRate_monthly*100:.3f} %', 'light_green')} [Monthly]")
            print(f"      - Volatility:      {volatility:.8f} [Theoretical 99.7%: {termcolor.colored(f'{volatility_tMin_997*100:.3f} %', 'light_magenta')} / {termcolor.colored(f'+{volatility_tMax_997*100:.3f} %', 'light_blue')}]")
            print(f"      - Score:           {pResult['bestResult']['score']:.8f}")
            print(f"      - nTrades:         {pResult['bestResult']['nTrades']}")

            #[5-3-7]: eFunction Initialization
            eFunction = exitFunction(modelName  = st['exitFunctionType'],
                                     isSeeker   = False, 
                                     leverage   = st['leverage'],
                                     pslReentry = st['pslReentry'])
            print(f"    - Preprocessing Analysis Data...")
            t_0 = time.perf_counter_ns()
            eFunction.preprocessData(linearizedAnalysis = linearizedAnalysis, indexIdentifier = descriptor['indexIdentifier'])
            t_1 = time.perf_counter_ns()
            print(f"    - Analysis Data Preprocessing Complete! <{(t_1-t_0)/1e6:.3f} ms>")

            #[5-3-8]: eFunction Processing
            print(f"    - Reconstructing Seeker Records...")
            t_0 = time.perf_counter_ns()
            recs    = sorted(pResult['records'], key = lambda x: x['score'], reverse = True)[:100]
            params  = [rec['functionParams'] for rec in recs]
            nParams = len(params)
            (
            balance_wallet_history, 
            balance_margin_history, 
            balance_bestFit_history,
            balance_bestFit_growthRates,
            balance_bestFit_volatilities,
            nTrades,
            ) = eFunction.performOnParams(params = params)
            t_1 = time.perf_counter_ns()
            print(f"    - Seeker Records Reconstruction Complete! <{(t_1-t_0)/1e6:.3f} ms>")

            #[5-3-9]: Matplot Drawing
            matplotlib.pyplot.plot(balance_wallet_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 1.0), linestyle='solid',  linewidth=1, zorder = 4)
            matplotlib.pyplot.plot(balance_margin_history[0,:].cpu(),  color=(0.0, 0.7, 1.0, 0.5), linestyle='dashed', linewidth=1, zorder = 3)
            matplotlib.pyplot.plot(balance_bestFit_history[0,:].cpu(), color=(0.8, 0.5, 0.8, 1.0), linestyle='solid',  linewidth=2, zorder = 5)
            for _rIndex in range (1, nParams):
                matplotlib.pyplot.plot(balance_wallet_history[_rIndex,:].cpu(), color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.10+0.10,3)), linestyle='solid',  linewidth=0.5, zorder = 2)
                matplotlib.pyplot.plot(balance_margin_history[_rIndex,:].cpu(), color=(0.5, 0.8, 0.0, round((_rIndex+1)/nParams*0.05+0.05,3)), linestyle='dashed', linewidth=0.5, zorder = 1)

            #[5-3-10]: Matplot Show
            matplotlib.pyplot.title(f"[RESULT READ] Wallet & Margin Balance History")
            matplotlib.pyplot.show()

            #[5-3-11]: Line Skip
            print()
        print(termcolor.colored("[Seeker Result Read Complete]", 'light_blue'))

    print(termcolor.colored("\n<RQP MAP PARAMETERS SEEKER PROCESS COMPLETE>", 'light_green'))