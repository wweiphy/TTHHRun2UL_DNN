from tensorflow.keras import optimizers

config_dict = {}

        
config_dict["ge4j_ge4t_ttH"] = {
        "layers":                   [1024,800,600],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.5,
        "L2_Norm":                  5e-4,
        "L1_Norm":                  0,
        "batch_size":               1024,
        "optimizer":                optimizers.Adagrad(learning_rate = 0.01),
        "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 1.0,
        "earlystopping_epochs":     500,
        "activation_coefficient":   0.3,
        "saveEpoch":                False
        }
        
config_dict["ge4j_ge3t_ttH"] = {
        "layers":                   [512,256,128,64],
#        "layers":                   [256,128,64,32],
        "loss_function":            "categorical_crossentropy",
        "Dropout":                  0.2,
        "L2_Norm":                  0,
        "L1_Norm":                  0,
        "batch_size":               1024,
#        "optimizer":                optimizers.Adagrad(learning_rate = 0.05),
        "optimizer":                optimizers.Adam(learning_rate = 2e-5),
        "activation_function":      "relu",
        # "activation_function":      "leakyrelu",
        "output_activation":        "Softmax",
        "earlystopping_percentage": 1.0,
        "earlystopping_epochs":     500,
        "activation_coefficient":   0.3,
        "saveEpoch":                False
        }
        
        
