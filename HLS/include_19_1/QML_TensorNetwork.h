// include/QML_TensorNetwork.h
#ifndef QML_TENSORNETWORK_H
#define QML_TENSORNETWORK_H

#include "defines.h"

// Top-level HLS entry function
void tn4ad(
    const data_t encoded_input_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    const data_t weights_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND],
    data_t* norm_score,
    bool* trigger_decision
);

#endif // QML_TENSORNETWORK_H