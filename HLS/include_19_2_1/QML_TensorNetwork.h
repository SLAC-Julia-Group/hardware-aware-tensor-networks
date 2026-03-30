// include/QML_TensorNetwork.h
#ifndef QML_TENSORNETWORK_H
#define QML_TENSORNETWORK_H

#include "defines.h"

// Top-level HLS entry function
void tn4ad(
    const data_t encoded_input_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN],
    const data_t weights1_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    const data_t weights2_axi[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND],
    data_t* norm_score,
    bool* trigger_decision
);

#endif // QML_TENSORNETWORK_H