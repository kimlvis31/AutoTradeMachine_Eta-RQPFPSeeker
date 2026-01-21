#[1]: Parameer Test
PARAMETERTEST = {'ppips':            'USC12_ppips\\USC12_BTCUSDT',
                 'exitFunctionType': 'SPDDEFAULT',
                 'leverage':         1,
                 'params':           (0.0001, 1.0, 0.000000, 1.000000, 0.05000, 0.000000, 1.000000, 0.05000), #(1.0, 1.0, 0.000000, 0.057000, 0.123000, 0.074000, 0.000200, 10.0, 0.007200, 0.608000, 0.213700, 0.000000, 10.0),
                 'pslReentry':       False,
                }

#[2]: PPIPS to Process
PPIPSTOPROCESS = [{'ppips':            'USC12_ppips\\USC12_BTCUSDT',
                   'exitFunctionType': 'SPDDEFAULT',
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
                                   None],
                   'nSeekerPoints':          100,
                   'nRepetition':            10,
                   'learningRate':           0.001,
                   'deltaRatio':             0.10,
                   'beta_velocity':          0.999,
                   'beta_momentum':          0.900,
                   'repopulationRatio':      0.95,
                   'repopulationInterval':   10,
                   'repopulationGuideRatio': 0.5,
                   'repopulationDecayRate':  0.1,
                   'scoringSamples':         50,
                   'scoring':                ('SHARPERATIO', (1e-4, 0.2, None, 1.0)),
                   'terminationThreshold':   1e-6,
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
ex: _RCODETOREAD = 'rqpfpResult_1768722056' (Result Folder Name)
"""
RCODETOREAD = 'rqpfpResult_1768759273'

#[4]: Mode
"""
[1]: TEST
[2]: SEEK
[3]: READ 
"""
MODE = 'TEST'