// include/defines.h
#ifndef DEFINES_H
#define DEFINES_H

#include <ap_int.h>
#include <ap_fixed.h>

/// Type definitions
typedef float data_t;
#define INPUT_FEATURES 57

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const data_t OFFSET_5 = data_t(5.0);
const data_t OFFSET_PI = data_t(M_PI);
const data_t INV_1200 = data_t(0.000833333333);
const data_t INV_800 = data_t(0.00125);
const data_t INV_2500 = data_t(0.0004);
const data_t INV_10 = data_t(0.1);
const data_t INV_2PI = data_t(1.0) / data_t(2.0 * M_PI);
const data_t INV_38 = data_t(1.0) / data_t(38.0);

// 19→1 Single-layer SMPO configuration
#define BIT_WIDTH 16

// typedef float smpo_t;
typedef ap_fixed<BIT_WIDTH,6> smpo_t; //Range -32 to 32
#define SMPO_INPUT_SITES 19
#define SMPO_OUTPUT_SITE 9      // Center site (0-indexed)
#define SMPO_PHYS_IN 3
#define SMPO_PHYS_OUT 3
#define SMPO_BOND 4

// typedef float final_t;
typedef ap_fixed<BIT_WIDTH,8,AP_TRN,AP_SAT> final_t; //Range -128 to 128, with saturation on overflow

// ===== Testbench Constants =====
#define CHECKPOINT_INTERVAL 100000
#define MAX_EVENTS -1

#endif // DEFINES_H