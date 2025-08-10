import evaluate

bleu_metric = evaluate.load("bleu")

import numpy as np
import pandas as pd

resluts = bleu_metric.compute(
    predictions=["the the the the the the"], references=[["the cat is on the mat"]], max_order=4
)
print(pd.DataFrame.from_dict(resluts, orient="index", columns=["value"]))
#                                                   value
# bleu                                                0.0
# precisions          [0.3333333333333333, 0.0, 0.0, 0.0]
# brevity_penalty                                     1.0
# length_ratio                                        1.0
# translation_length                                    6
# reference_length                                      6

resluts = bleu_metric.compute(
    predictions=["the cat is on mat"],
    references=[["the cat is on the mat"]],
    max_order=4
)
print(pd.DataFrame.from_dict(resluts, orient="index", columns=["value"]))
#                                                    value
# bleu                                             0.57893
# precisions          [1.0, 0.75, 0.6666666666666666, 0.5]
# brevity_penalty                                 0.818731
# length_ratio                                    0.833333
# translation_length                                     5
# reference_length                                       6


