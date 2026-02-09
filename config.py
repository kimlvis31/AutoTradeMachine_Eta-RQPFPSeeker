#[1]: Parameer Test
"""
 * This parameter defines the model to test with a specific set of parameters.
"""
PARAMETERTEST = {'analysisData':     'USC7_ae\\USC7_BTCUSDT',
                 'exitFunctionType': 'SPDDEFAULT',
                 'leverage':         1,
                 'tradeParams':      (0.0100, 1.0),
                 'modelParams':      (-0.439299, 0.014246, 0.517208, 0.035750, 0.671966, 0.671804),
                 'pslReentry':       False,
                }
"""
PARAMETERTEST = {'analysisData':     'USC7_ae\\USC7_BTCUSDT',
                 'exitFunctionType': 'MMACDLONGDEFAULT',
                 'tradeParams':      (0.0061, 1.0),
                 'modelParams':      (4.9633, 5.61, -0.9308, 0.380649, 1.000000),
                 'pslReentry':       False,
                }
"""

#[2]: Analysis Data to Process
"""
 * This parameter defines the model 
"""
SEEKERTARGETS = [{'analysisData':     'USC7_ae\\USC7_BTCUSDT',
                  'exitFunctionType': 'MMACDLONGDEFAULT',
                  'leverage':         1,
                  'pslReentry':       False,
                  'tradeParamConfig': (None, 1.0000),
                  'modelParamConfig': (None, None, None, None, None),
                  'nSeekerPoints':            10000,
                  'parameterBatchSize':       None,
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