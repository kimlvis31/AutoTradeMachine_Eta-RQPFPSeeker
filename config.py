#[1]: Parameer Test
PARAMETERTEST = {'ppips':            'USC12_ppips\\USC12_BTCUSDT',
                 'exitFunctionType': 'ROTATIONALGAUSSIAN1',
                 'leverage':         1,
                 'params':           (1.0, 1.0, 0.3019, 0.636598, 0.045564, 0.132109, 0.200677, 10.0, 0.193345, 0.573387, 0.163673, 0.842677, 10.0), #(1.0, 1.0, 0.000000, 0.057000, 0.123000, 0.074000, 0.000200, 10.0, 0.007200, 0.608000, 0.213700, 0.000000, 10.0),
                 'pslReentry':       True,
                }

#[2]: PPIPS to Process
PPIPSTOPROCESS = [{'ppips':            'USC12_ppips\\USC12_BTCUSDT',
                   'exitFunctionType': 'ROTATIONALGAUSSIAN1',
                   'leverage':   1,
                   'pslReentry': False,
                   'parameterBatchSize': 32*4096,
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
                   'nSeekerPoints':          100,
                   'nRepetition':            50,
                   'learningRate':           0.001,
                   'deltaRatio':             0.10,
                   'beta_velocity':          0.999,
                   'beta_momentum':          0.900,
                   'repopulationRatio':      0.95,
                   'repopulationInterval':   10,
                   'repopulationGuideRatio': 0.5,
                   'repopulationDecayRate':  0.1,
                   'scoringSamples':         50,
                   'scoring':                ('SHARPERATIO', (1e-4, 0.2, 0.1, 1.0)),
                   'terminationThreshold':   1e-6,
                  },
                 ]

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
MODE = 'SEEK'