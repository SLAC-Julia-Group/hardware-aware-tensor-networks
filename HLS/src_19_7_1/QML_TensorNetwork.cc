//===============================
//Author(s): Abhilasha Dave, Sagar Addepalli
//Last Update: 12.03.2026
//================================
//
// Cascaded SMPO: 19→7→1 Architecture
//   Layer 1: 19-site SMPO, bond b₁=2, spacing=3 → 7-site MPS (bond=2, phys=3)
//   Layer 2: 7-site SMPO, bond b₂=2, composite bond B=b₁b₂=4 → 1-site output
//
// Mathematically equivalent to a single 19→1 SMPO with bond=4.
//
// Expected defines in QML_TensorNetwork.h:
//   LAYER1_INPUT_SITES    = 19
//   LAYER1_OUTPUT_SITES   = 7
//   LAYER1_SPACING        = 3
//   LAYER1_PHYS_IN        = 3   (input features: pt, eta, phi)
//   LAYER1_PHYS_OUT       = 3
//   LAYER1_SMPO_BOND      = 2
//   LAYER2_OUTPUT_SITE    = 3   (center of 7 sites)
//   LAYER2_PHYS_OUT       = 3
//   LAYER2_SMPO_BOND      = 2
//   LAYER2_COMPOSITE_BOND = 4   (= LAYER1_SMPO_BOND × LAYER2_SMPO_BOND)
//
// Types: data_t (AXI), smpo_t, smpo_t (compute), final_t (accumulation)
//
#include <hls_stream.h>
#include "QML_TensorNetwork.h"

/// ===== Helper Functions =====
/// Copy encoded input from AXI to local buffer with type conversion
void copy_inputs(
    const data_t encoded_input_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN],
    smpo_t encoded_input[LAYER1_INPUT_SITES][LAYER1_PHYS_IN]
) {
    #pragma HLS INLINE off
    for (int i = 0; i < LAYER1_INPUT_SITES; i++)
        for (int f = 0; f < LAYER1_PHYS_IN; f++) {
        	#pragma HLS PIPELINE II=1
            encoded_input[i][f] = smpo_t(encoded_input_axi[i][f]);
        }
}

/// Load Layer 1 weights from AXI into local buffer with type conversion
void load_weights1(
    const data_t weights1_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t weights1_buf[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    for (int i = 0; i < LAYER1_INPUT_SITES; i++)
        for (int f = 0; f < LAYER1_PHYS_IN; f++)
            for (int p = 0; p < LAYER1_PHYS_OUT; p++)
                for (int l = 0; l < LAYER1_SMPO_BOND; l++)
                    for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                        #pragma HLS PIPELINE II=1
                        weights1_buf[i][f][p][l][r] = smpo_t(weights1_axi[i][f][p][l][r]);
                    }
}

/// Load Layer 2 weights from AXI into local buffer with type conversion
void load_weights2(
    const data_t weights2_axi[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND],
    smpo_t weights2_buf[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND]
) {
    #pragma HLS INLINE off
    for (int i = 0; i < LAYER1_OUTPUT_SITES; i++)
        for (int f = 0; f < LAYER1_PHYS_OUT; f++)
            for (int p = 0; p < LAYER2_PHYS_OUT; p++)
                for (int l = 0; l < LAYER2_SMPO_BOND; l++)
                    for (int r = 0; r < LAYER2_SMPO_BOND; r++) {
                        #pragma HLS PIPELINE II=1
                        weights2_buf[i][f][p][l][r] = smpo_t(weights2_axi[i][f][p][l][r]);
                    }
}

/// ===== Layer 1: 19→7 SMPO =====

/// Layer 1 vertical contraction
/// Contracts the physical input index (features) with Layer 1 SMPO weights
/// at each of the 19 input sites. Output sites (i % SPACING == 0) produce
/// tensors with shape [d'][b₁][b₁]. Non-output sites produce [1][b₁][b₁]
/// matrices (only p=0).
///
/// l1_out[i][p][l][r] = Σ_f encoded_input[i][f] × W1[i][f][p][l][r]
void layer1_smpo(
    const smpo_t encoded_input[LAYER1_INPUT_SITES][LAYER1_PHYS_IN],
    const smpo_t weights1[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1 complete dim=3

    for (int i = 0; i < LAYER1_INPUT_SITES; i++) {
        #pragma HLS PIPELINE II=1
        for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
            if (i % LAYER1_SPACING != 0 && p > 0) continue;
            for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
                for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                    smpo_t acc = 0;
                    #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                    for (int f = 0; f < LAYER1_PHYS_IN; f++) {
                        #pragma HLS UNROLL
                        acc += encoded_input[i][f] * weights1[i][f][p][l][r];
                    }
                    l1_out[i][p][l][r] = acc;
                }
            }
        }
    }
}

/// Layer 1 horizontal contraction — two-pass decomposition
/// Contracts groups of non-output sites between consecutive output sites,
/// reducing the 19-site L1 output to a 7-site MPS with bond b₁=2.
///
/// Group structure for spacing=3, output sites at {0, 3, 6, 9, 12, 15, 18}:
///   o=0: Copy site 0 directly (left boundary)
///   o=1: Chain sites {1, 2}, absorb into anchor site 3
///   o=2: Chain sites {4, 5}, absorb into anchor site 6
///   o=3: Chain sites {7, 8}, absorb into anchor site 9
///   o=4: Chain sites {10, 11}, absorb into anchor site 12
///   o=5: Chain sites {13, 14}, absorb into anchor site 15
///   o=6: Chain sites {16, 17}, absorb into anchor site 18
///
/// Two-pass decomposition avoids the three-way chained multiply present in
/// the single-loop formulation. Each pass contains only single-depth
/// multiplies, making BIND_OP impl=fabric effective (0 DSPs).
///
/// Pass 1 — Chain: contract each group's two non-output site matrices
///   chain[o][l][b1] = Σ_{b0} first[0][l][b0] × mid[0][b0][b1]
///
/// Pass 2 — Absorb: contract chain result with anchor output tensor
///   out[o][p][l][r] = Σ_{b1} chain[o][l][b1] × anchor[p][b1][r]
///
/// Both passes pipeline over o with all inner indices unrolled.
void layer1_contract_latency_aggressive(
    const smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=1
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=4

    #pragma HLS ARRAY_PARTITION variable=l1_contracted complete dim=1

    // ====================================================================
    // Pass 0: Copy boundary output site (o=0) directly
    // ====================================================================
    // Site 0 has no preceding non-output sites — its tensor passes through
    // unmodified as the first tensor of the 7-site output MPS.
    for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                l1_contracted[0][p][l][r] = l1_out[0][p][l][r];
            }
        }
    }

    // ====================================================================
    // Pass 1: Chain contraction — contract non-output site pairs per group
    // ====================================================================
    // For each group o=1..6, the two non-output sites (first, mid) between
    // the previous output site and the current anchor form a 2×2 matrix
    // product. Pipeline over o, unroll (l, b1, b0).
    //
    // Each iteration reads two sites from l1_out (first, mid) at p=0.
    // BRAM dual-port handles the two simultaneous reads.
    smpo_t chain_buf[LAYER1_OUTPUT_SITES][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=chain_buf complete

    for (int o = 1; o < LAYER1_OUTPUT_SITES; o++) {
        // #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL
        int first = (o - 1) * LAYER1_SPACING + 1;
        int mid   = first + 1;

        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int b1 = 0; b1 < LAYER1_SMPO_BOND; b1++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b0 = 0; b0 < LAYER1_SMPO_BOND; b0++) {
                    #pragma HLS UNROLL
                    acc += l1_out[first][0][l][b0] * l1_out[mid][0][b0][b1];
                }
                chain_buf[o][l][b1] = acc;
            }
        }
    }

    // ====================================================================
    // Pass 2: Absorb chain into anchor output tensor
    // ====================================================================
    // For each group o=1..6, contract the chain result with the anchor
    // site's tensor (which carries the physical output index p).
    // Pipeline over o, unroll (p, l, r, b1).
    //
    // Each iteration reads chain_buf[o] (fully partitioned, instant) and
    // l1_out[anchor] (one BRAM read, dims 2-4 partitioned → all 12 elements).
    for (int o = 1; o < LAYER1_OUTPUT_SITES; o++) {
        // #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL
        int anchor = o * LAYER1_SPACING;

        for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
            #pragma HLS UNROLL
            for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
                #pragma HLS UNROLL
                for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                    #pragma HLS UNROLL
                    smpo_t acc = 0;
                    #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                    for (int b1 = 0; b1 < LAYER1_SMPO_BOND; b1++) {
                        #pragma HLS UNROLL
                        acc += chain_buf[o][l][b1] * l1_out[anchor][p][b1][r];
                    }
                    l1_contracted[o][p][l][r] = acc;
                }
            }
        }
    }
}


void layer1_contract(
    const smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=4
 
    // ====================================================================
    // Pass 0: Copy boundary output site (o=0) directly
    // ====================================================================
    for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                l1_contracted[0][p][l][r] = l1_out[0][p][l][r];
            }
        }
    }
 
    // ====================================================================
    // Pass 1: Chain contraction — pipeline over o, unroll bond indices
    // ====================================================================
    smpo_t chain_buf[LAYER1_OUTPUT_SITES][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=chain_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=chain_buf complete dim=3
 
    for (int o = 1; o < LAYER1_OUTPUT_SITES; o++) {
        #pragma HLS PIPELINE II=1
        int first = (o - 1) * LAYER1_SPACING + 1;
        int mid   = first + 1;
 
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int b1 = 0; b1 < LAYER1_SMPO_BOND; b1++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b0 = 0; b0 < LAYER1_SMPO_BOND; b0++) {
                    #pragma HLS UNROLL
                    acc += l1_out[first][0][l][b0] * l1_out[mid][0][b0][b1];
                }
                chain_buf[o][l][b1] = acc;
            }
        }
    }
 
    // ====================================================================
    // Pass 2: Absorb chain into anchor — pipeline over o, unroll all inner
    // ====================================================================
    for (int o = 1; o < LAYER1_OUTPUT_SITES; o++) {
        #pragma HLS PIPELINE II=1
        int anchor = o * LAYER1_SPACING;
 
        for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
            #pragma HLS UNROLL
            for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
                #pragma HLS UNROLL
                for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                    #pragma HLS UNROLL
                    smpo_t acc = 0;
                    #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                    for (int b1 = 0; b1 < LAYER1_SMPO_BOND; b1++) {
                        #pragma HLS UNROLL
                        acc += chain_buf[o][l][b1] * l1_out[anchor][p][b1][r];
                    }
                    l1_contracted[o][p][l][r] = acc;
                }
            }
        }
    }
}

/// ===== Layer 2: 7→1 SMPO =====

/// Layer 2 vertical contraction with composite bond indexing
/// Contracts the physical index of the 7-site MPS (from Layer 1, bond b₁=2)
/// with Layer 2 SMPO weights (bond b₂=2). The output tensors carry composite
/// bond indices of dimension B = b₁ × b₂ = 4:
///
///   composite_left  = l_mps × b₂ + l_smpo
///   composite_right = r_mps × b₂ + r_smpo
///
/// l2_out[i][pout][lc][rc] = Σ_pin MPS[i][pin][l_mps][r_mps] × W2[i][pin][pout][l_smpo][r_smpo]
///
/// Only site LAYER2_OUTPUT_SITE (=3) produces all d' physical outputs;
/// all other sites produce only pout=0.
void layer2_smpo(
    const smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    const smpo_t weights2[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND],
    smpo_t l2_out[LAYER1_OUTPUT_SITES][LAYER2_PHYS_OUT][LAYER2_COMPOSITE_BOND][LAYER2_COMPOSITE_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l1_contracted complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2 complete dim=3

    for (int i = 0; i < LAYER1_OUTPUT_SITES; i++) {
        #pragma HLS PIPELINE II=1
        for (int pout = 0; pout < LAYER2_PHYS_OUT; pout++) {
            if (i != LAYER2_OUTPUT_SITE && pout > 0) continue;
            for (int lp = 0; lp < LAYER1_SMPO_BOND; lp++) {
                for (int rp = 0; rp < LAYER1_SMPO_BOND; rp++) {
                    for (int ls = 0; ls < LAYER2_SMPO_BOND; ls++) {
                        for (int rs = 0; rs < LAYER2_SMPO_BOND; rs++) {
                            int lc = lp * LAYER2_SMPO_BOND + ls;
                            int rc = rp * LAYER2_SMPO_BOND + rs;
                            smpo_t acc = 0;
                            #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                            for (int pin = 0; pin < LAYER1_PHYS_OUT; pin++) {
                                #pragma HLS UNROLL
                                acc += l1_contracted[i][pin][lp][rp] * weights2[i][pin][pout][ls][rs];
                            }
                            l2_out[i][pout][lc][rc] = acc;
                        }
                    }
                }
            }
        }
    }
}

/// Layer 2 sideways contraction
/// Interleaved bidirectional sweep over 7 sites at composite bond B=4,
/// producing a d'-vector from the center output site.
///
/// Left wing:  sites 0 (boundary init) → 1 → 2    (2 sweep steps)
/// Right wing: sites 6 (boundary init) → 5 → 4    (2 sweep steps)
/// Merge at center site 3.
///
/// Design follows the optimized 19→1 pattern:
///   - Inner bond index fully UNROLL'd (all B outputs per wing in parallel)
///   - Sweep at II=2 with FABRIC multipliers (combinational, no DSP pipeline
///     latency, so carried dependence resolves within 2-cycle window)
///   - Single-pass merge (consistent with 19→1 for one-to-one MAC comparison)
void layer2_contract(
    const smpo_t l2_out[LAYER1_OUTPUT_SITES][LAYER2_PHYS_OUT][LAYER2_COMPOSITE_BOND][LAYER2_COMPOSITE_BOND],
    smpo_t output[LAYER2_PHYS_OUT]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=4

    smpo_t left_env[LAYER2_COMPOSITE_BOND];
    smpo_t right_env[LAYER2_COMPOSITE_BOND];
    #pragma HLS ARRAY_PARTITION variable=left_env complete
    #pragma HLS ARRAY_PARTITION variable=right_env complete

    // ====================================================================
    // Initialization: extract boundary vectors (1 cycle, fully unrolled)
    // ====================================================================
    // Site 0: left boundary trivial (lc=0)
    // Site 6: right boundary trivial (rc=0)
    for (int idx = 0; idx < LAYER2_COMPOSITE_BOND; idx++) {
        #pragma HLS UNROLL
        left_env[idx]  = l2_out[0][0][0][idx];
        right_env[idx] = l2_out[LAYER1_OUTPUT_SITES - 1][0][idx][0];
    }

    // ====================================================================
    // Interleaved sweep: 2 steps, both wings per step
    // ====================================================================
    for (int s = 1; s < LAYER2_OUTPUT_SITE; s++) {
        #pragma HLS PIPELINE II=2

        const int left_site  = s;
        const int right_site = LAYER1_OUTPUT_SITES - 1 - s;

        smpo_t next_left[LAYER2_COMPOSITE_BOND];
        smpo_t next_right[LAYER2_COMPOSITE_BOND];
        #pragma HLS ARRAY_PARTITION variable=next_left complete
        #pragma HLS ARRAY_PARTITION variable=next_right complete

        for (int idx = 0; idx < LAYER2_COMPOSITE_BOND; idx++) {
            #pragma HLS UNROLL

            // Left: next_left[r] = Σ_b left_env[b] · T_s[b][r]
            smpo_t acc_l = 0;
            #pragma HLS BIND_OP variable=acc_l op=mul impl=fabric
            for (int b = 0; b < LAYER2_COMPOSITE_BOND; b++) {
                #pragma HLS UNROLL
                acc_l += left_env[b] * l2_out[left_site][0][b][idx];
            }
            next_left[idx] = acc_l;

            // Right: next_right[l] = Σ_b T_s[l][b] · right_env[b]
            smpo_t acc_r = 0;
            #pragma HLS BIND_OP variable=acc_r op=mul impl=fabric
            for (int b = 0; b < LAYER2_COMPOSITE_BOND; b++) {
                #pragma HLS UNROLL
                acc_r += l2_out[right_site][0][idx][b] * right_env[b];
            }
            next_right[idx] = acc_r;
        }

        for (int idx = 0; idx < LAYER2_COMPOSITE_BOND; idx++) {
            #pragma HLS UNROLL
            left_env[idx]  = next_left[idx];
            right_env[idx] = next_right[idx];
        }
    }

    // ====================================================================
    // Final merge: contract both environments through center output site
    // ====================================================================
    // output[p] = Σ_{l,r} left_env[l] · T_center[p][l][r] · right_env[r]
    for (int p = 0; p < LAYER2_PHYS_OUT; p++) {
        #pragma HLS PIPELINE II=1
        smpo_t acc = 0;
        #pragma HLS BIND_OP variable=acc op=mul impl=fabric
        for (int l = 0; l < LAYER2_COMPOSITE_BOND; l++) {
            for (int r = 0; r < LAYER2_COMPOSITE_BOND; r++) {
                #pragma HLS UNROLL
                acc += left_env[l] *
                       l2_out[LAYER2_OUTPUT_SITE][p][l][r] *
                       right_env[r];
            }
        }
        output[p] = acc;
    }
}

/// Compute norm squared of the output vector
void compute_norm_squared(
    const smpo_t output[LAYER2_PHYS_OUT],
    final_t* norm_squared
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1

    final_t norm_sq = 0;
    #pragma HLS BIND_OP variable=norm_sq op=mul impl=fabric
    for (int p = 0; p < LAYER2_PHYS_OUT; p++) {
        #pragma HLS UNROLL
        norm_sq += (final_t)output[p] * (final_t)output[p];
    }

    *norm_squared = norm_sq;
}

/// ===== Top-level HLS Entry Function =====
void tn4ad(
    const data_t encoded_input_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN],
    const data_t weights1_axi[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    const data_t weights2_axi[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND],
    data_t* norm_score,
    bool* trigger_decision
) {
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS INTERFACE m_axi port=encoded_input_axi offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=weights1_axi offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=weights2_axi offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=trigger_decision offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=norm_score

    // Internal buffers
    smpo_t encoded_input[LAYER1_INPUT_SITES][LAYER1_PHYS_IN];
    smpo_t weights1_local[LAYER1_INPUT_SITES][LAYER1_PHYS_IN][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    smpo_t weights2_local[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER2_PHYS_OUT][LAYER2_SMPO_BOND][LAYER2_SMPO_BOND];
    smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    smpo_t l2_out[LAYER1_OUTPUT_SITES][LAYER2_PHYS_OUT][LAYER2_COMPOSITE_BOND][LAYER2_COMPOSITE_BOND];
    smpo_t out_vec[LAYER2_PHYS_OUT];
    final_t final_norm_squared = 0.0;

    // Array partitioning for parallel access in compute functions
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1_local complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1_local complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights2_local complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2_local complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=2
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=4
    #pragma HLS ARRAY_PARTITION variable=l1_contracted complete dim=2
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=2
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=4
    #pragma HLS ARRAY_PARTITION variable=out_vec complete dim=1

    // Phase 0a: Load encoded input from AXI with type conversion
    copy_inputs(encoded_input_axi, encoded_input);
    // Phase 0b: Load weights from AXI
    load_weights1(weights1_axi, weights1_local);
    load_weights2(weights2_axi, weights2_local);

    // Phase 1: Layer 1 — apply SMPO and contract to 7-site MPS
    layer1_smpo(encoded_input, weights1_local, l1_out);
    layer1_contract_latency_aggressive(l1_out, l1_contracted);

    // Phase 2: Layer 2 — apply SMPO with composite indexing, then sideways contract
    layer2_smpo(l1_contracted, weights2_local, l2_out);
    layer2_contract(l2_out, out_vec);

    // Phase 3: Compute score and trigger decision
    compute_norm_squared(out_vec, &final_norm_squared);
    *norm_score = (data_t)final_norm_squared;
    *trigger_decision = (*norm_score < data_t(25.0)) || (*norm_score > data_t(75.0));
}