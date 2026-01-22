#[1]: Parameer Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'ppips':            'USC5_ppips\\USC5_BTCUSDT',
                 'exitFunctionType': 'CSPDRG1',
                 'leverage':         1,
                 'params':           (1.0, 1.0, 0.000000, 0.057000, 0.123000, 0.074000, 0.000200, 10.0, 0.007200, 0.608000, 0.213700, 0.000000, 10.0),
                 'pslReentry':       False,
                }

#[2]: PPIPS to Process
"""
 * This parameter defines the model 
"""
PPIPSTOPROCESS = [{'ppips':            'USC5_ppips\\USC5_BTCUSDT',
                   'exitFunctionType': 'CSPDRG1',
                   'leverage':   1,
                   'pslReentry': False,
                   'parameterBatchSize': 32*4096,
                   'paramConfig': [None,
                                   1.0000,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   None,
                                   ],
                   'nSeekerPoints':            100,
                   'nRepetition':              10,
                   'learningRate':             0.001,
                   'deltaRatio':               0.10,
                   'beta_velocity':            0.999,
                   'beta_momentum':            0.900,
                   'repopulationRatio':        0.95,
                   'repopulationInterval':     10,
                   'repopulationGuideRatio':   0.5,
                   'repopulationDecayRate':    0.1,
                   'scoringSamples':           50,
                   'scoring':                  'SHARPERATIO',
                   'scoring_maxMDD':           1,
                   'scoring_growthRateWeight': 1.0,
                   'scoring_growthRateScaler': 1e5,
                   'scoring_volatilityWeight': 0.1,
                   'scoring_volatilityScaler': 0.1,
                   'scoring_nTradesWeight':    0.1,
                   'scoring_nTradesScaler':    0.00001,
                   'terminationThreshold':     1e-6,
                  },
                 ]


"""
'paramConfig': [None,   #FSL Immed
                1.0000, #FSL Close
                None,   #Side Offset
                None,   #Theta - SHORT
                None,   #Alpha - SHORT
                None,   #Beta0 - SHORT
                None,   #Beta1 - SHORT
                None,   #Gamma - SHORT
                None,   #Theta - LONG
                None,   #Alpha - LONG
                None,   #Beta0 - LONG
                None,   #Beta1 - LONG
                None    #Gamma - LONG
                ],
"""

#[3]: Result Code to Read
"""
 * This parameter defines the target RQP function optimized parameters search process result to read. The target is the result folder name under the 'results' folder.
 * Example: _RCODETOREAD = 'rqpfpResult_1768722056'
"""
RCODETOREAD = 'rqpfpResult_1768759273'

#[4]: Mode
"""
<MODE>
 * 
 * Available Modes: 'TEST'/'SEEK'/'READ'
"""
MODE = 'SEEK'