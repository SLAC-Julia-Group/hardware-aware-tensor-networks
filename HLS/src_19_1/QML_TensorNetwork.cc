//================================
//Author(s): Abhilasha Dave, Sagar Addepalli
//Last Update: 12.03.2026
//================================
#include <hls_stream.h>
#include "QML_TensorNetwork.h"

/// ===== Helper Functions =====
/// Copy encoded input from AXI to local buffer with type conversion
void copy_inputs(
    const data_t encoded_input_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    smpo_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN]
) {
    #pragma HLS INLINE off
    for (int i = 0; i < SMPO_INPUT_SITES; i++)
        for (int f = 0; f < SMPO_PHYS_IN; f++) {
        	#pragma HLS PIPELINE II=1
            encoded_input[i][f] = smpo_t(encoded_input_axi[i][f]);
        }
}

/// Load weights from AXI into local buffer
void load_weights(
    const data_t weights_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND],
    smpo_t weights_buf[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND]
) {
	#pragma HLS INLINE off
    for (int i = 0; i < SMPO_INPUT_SITES; i++)
        for (int f = 0; f < SMPO_PHYS_IN; f++)
            for (int p = 0; p < SMPO_PHYS_OUT; p++)
                for (int l = 0; l < SMPO_BOND; l++)
                    for (int r = 0; r < SMPO_BOND; r++) {
                    	#pragma HLS PIPELINE II=1
                        weights_buf[i][f][p][l][r] = smpo_t(weights_axi[i][f][p][l][r]);
                    }
}

/// ===== Model Functions =====

/// Layer 1 SMPO application
void smpo_application(
    const smpo_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    const smpo_t weights[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND],
    smpo_t smpo_out[SMPO_INPUT_SITES][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=3

    for (int i = 0; i < SMPO_INPUT_SITES; i++) {
        #pragma HLS PIPELINE II=1
        for (int p = 0; p < SMPO_PHYS_OUT; p++) {
            // Skip non-output physical indices for sites without output
            if (i != SMPO_OUTPUT_SITE && p > 0) continue;
            for (int l = 0; l < SMPO_BOND; l++) {
                for (int r = 0; r < SMPO_BOND; r++) {
                    smpo_t acc = 0;
                    #pragma HLS BIND_OP variable=acc op=mul impl=fabric  // <-- Use LUTs
                    for (int f = 0; f < SMPO_PHYS_IN; f++) {
                        #pragma HLS UNROLL
                        acc += encoded_input[i][f] * weights[i][f][p][l][r];
                    }
                    smpo_out[i][p][l][r] = acc;
                }
            }
        }
    }
}

/// ===== Sideways Contraction =====
/// Contracts 19-site SMPO output down to a [PHYS_OUT] vector by sweeping
/// environment vectors inward from both boundaries toward the center site.
///
/// Critical design decisions:
///   1. Inner index (idx) is FULLY UNROLLED: all BOND outputs per wing are
///      computed as parallel combinational logic in a single pipeline stage.
///      This means each step of the s loop costs II cycles, not II × BOND.
///   2. Sweep uses FABRIC multipliers: combinational (no pipeline registers),
///      so the carried dependence left_env[s] → left_env[s+1] resolves
///      within the II window. DSP48 pipeline depth (~7 cycles) would force
///      II ≥ 4, defeating the purpose.
///   3. Merge uses unconstrained multipliers: no carried dependence across
///      iterations (each p is independent), so DSPs can pipeline freely.
///
/// Tensor layout: smpo_out[site][phys_out][left_bond][right_bond]
///   - Non-output sites: only phys_out=0 is populated
///   - Site 0:  trivial left boundary  (left bond index = 0)
///   - Site 18: trivial right boundary (right bond index = 0)

void sideways_contract(
    const smpo_t smpo_out[SMPO_INPUT_SITES][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND],
    smpo_t output[SMPO_PHYS_OUT]
) {
    #pragma HLS INLINE off

    smpo_t left_env[SMPO_BOND];
    smpo_t right_env[SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=left_env complete
    #pragma HLS ARRAY_PARTITION variable=right_env complete

    // ====================================================================
    // Initialization: extract boundary vectors (1 cycle, fully unrolled)
    // ====================================================================
    for (int idx = 0; idx < SMPO_BOND; idx++) {
        #pragma HLS UNROLL
        left_env[idx]  = smpo_out[0][0][0][idx];
        right_env[idx] = smpo_out[SMPO_INPUT_SITES - 1][0][idx][0];
    }

    // ====================================================================
    // Interleaved sweep: pipeline over s, all outputs unrolled per step
    // ====================================================================
    // Fabric multipliers ensure combinational evaluation — no DSP pipeline
    // registers — so the loop-carried dependence on left_env/right_env
    // resolves within a 2-cycle initiation interval.
    //
    // Resource cost: 2 × BOND² fabric multipliers (both wings, all outputs
    // in parallel). Reused across the 8 pipeline iterations.
    for (int s = 1; s < SMPO_OUTPUT_SITE; s++) {
        #pragma HLS PIPELINE II=2

        const int left_site  = s;
        const int right_site = SMPO_INPUT_SITES - 1 - s;

        smpo_t next_left[SMPO_BOND];
        smpo_t next_right[SMPO_BOND];
        #pragma HLS ARRAY_PARTITION variable=next_left complete
        #pragma HLS ARRAY_PARTITION variable=next_right complete

        for (int idx = 0; idx < SMPO_BOND; idx++) {
            #pragma HLS UNROLL

            // Left: next_left[r] = Σ_b left_env[b] · T_s[b][r]
            smpo_t acc_l = 0;
            #pragma HLS BIND_OP variable=acc_l op=mul impl=fabric
            for (int b = 0; b < SMPO_BOND; b++) {
                #pragma HLS UNROLL
                acc_l += left_env[b] * smpo_out[left_site][0][b][idx];
            }
            next_left[idx] = acc_l;

            // Right: next_right[l] = Σ_b T_s[l][b] · right_env[b]
            smpo_t acc_r = 0;
            #pragma HLS BIND_OP variable=acc_r op=mul impl=fabric
            for (int b = 0; b < SMPO_BOND; b++) {
                #pragma HLS UNROLL
                acc_r += smpo_out[right_site][0][idx][b] * right_env[b];
            }
            next_right[idx] = acc_r;
        }

        for (int idx = 0; idx < SMPO_BOND; idx++) {
            #pragma HLS UNROLL
            left_env[idx]  = next_left[idx];
            right_env[idx] = next_right[idx];
        }
    }

    // ====================================================================
    // Final merge: contract both environments through center output site
    // ====================================================================
    // output[p] = Σ_{l,r} left_env[l] · T_center[p][l][r] · right_env[r]
    for (int p = 0; p < SMPO_PHYS_OUT; p++) {
        #pragma HLS PIPELINE II=1
        smpo_t acc = 0;
        #pragma HLS BIND_OP variable=acc op=mul impl=fabric
        for (int l = 0; l < SMPO_BOND; l++) {
            for (int r = 0; r < SMPO_BOND; r++) {
                #pragma HLS UNROLL
                acc += left_env[l] *
                       smpo_out[SMPO_OUTPUT_SITE][p][l][r] *
                       right_env[r];
            }
        }
        output[p] = acc;
    }
}

/// Compute norm squared of output vector
void compute_norm_squared(
    const smpo_t output[SMPO_PHYS_OUT],
    final_t* norm_squared
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=output complete
    
    final_t norm_sq = 0;
    #pragma HLS BIND_OP variable=norm_sq op=mul impl=fabric
    for (int p = 0; p < SMPO_PHYS_OUT; p++) {
        #pragma HLS UNROLL
        norm_sq += (final_t)output[p] * (final_t)output[p];
    }
    
    *norm_squared = norm_sq;
}

// ===== Top-level HLS Entry Function =====
void tn4ad(
    const data_t encoded_input_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    const data_t weights_axi[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND],
    data_t* norm_score,
    bool* trigger_decision
) {
	#pragma HLS INTERFACE s_axilite port=return
	#pragma HLS INTERFACE m_axi port=encoded_input_axi offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=weights_axi offset=slave bundle=gmem
	#pragma HLS INTERFACE m_axi port=trigger_decision offset=slave bundle=gmem
	#pragma HLS INTERFACE s_axilite port=norm_score

    // Internal buffers
    smpo_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN];
    smpo_t smpo_out[SMPO_INPUT_SITES][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND];
    smpo_t out_vec[SMPO_PHYS_OUT];
    final_t final_norm_squared = 0.0;

    smpo_t weights[SMPO_INPUT_SITES][SMPO_PHYS_IN][SMPO_PHYS_OUT][SMPO_BOND][SMPO_BOND];

	#pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2
	#pragma HLS ARRAY_PARTITION variable=smpo_out complete dim=2
	#pragma HLS ARRAY_PARTITION variable=out_vec complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weights complete dim=3

    // Phase 0a: Copy and cast the input
    copy_inputs(encoded_input_axi, encoded_input);
    // Phase 0b: Load weights
    load_weights(weights_axi, weights);

    // Phase 1: Execute model logic
    smpo_application(encoded_input, weights, smpo_out);
    sideways_contract(smpo_out, out_vec);

    // Phase 2: Compute score
    compute_norm_squared(out_vec, &final_norm_squared);

    // Write outputs
    *norm_score = (data_t)final_norm_squared;
    *trigger_decision = (*norm_score < data_t(25.0)) || (*norm_score > data_t(75.0));
}
