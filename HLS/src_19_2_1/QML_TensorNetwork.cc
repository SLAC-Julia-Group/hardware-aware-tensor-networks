//===============================
//Author(s): Abhilasha Dave, Sagar Addepalli
//Last Update: 12.03.2026
//================================
//
// Cascaded SMPO: 19→2→1 Architecture
//   Layer 1: 19-site SMPO, bond b₁=2 → 2-site MPS (bond=2, phys=3)
//            Output sites at {0, 18}.
//   Layer 2: 2-site SMPO, bond b₂=2, composite bond B=b₁b₂=4 → 1-site output
//            Output at site 1 (right boundary).
//
// Mathematically equivalent to a single 19→1 SMPO with bond=4.
//
// L1 horizontal uses a binary tree to contract 17 non-output site matrices
// in O(log₂ 17) = 5 levels, all products within each level fully parallel.
//
// Expected defines in QML_TensorNetwork.h:
//   LAYER1_INPUT_SITES    = 19
//   LAYER1_OUTPUT_SITES   = 2
//   LAYER1_PHYS_IN        = 3
//   LAYER1_PHYS_OUT       = 3
//   LAYER1_SMPO_BOND      = 2
//   LAYER2_OUTPUT_SITE    = 1
//   LAYER2_PHYS_OUT       = 3
//   LAYER2_SMPO_BOND      = 2
//   LAYER2_COMPOSITE_BOND = 4   (= LAYER1_SMPO_BOND × LAYER2_SMPO_BOND)
//
// Type: smpo_t = ap_fixed<16,6> throughout
//
#include <hls_stream.h>
#include "QML_TensorNetwork.h"

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

/// ===== Weight Loading (not part of core algorithm) =====

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

/// ===== Layer 1: 19→2 SMPO =====

/// Layer 1 vertical contraction
/// Contracts the physical input index with Layer 1 SMPO weights at each site.
/// Output sites (0 and 18) produce [d'][b₁][b₁] tensors; all others produce
/// [1][b₁][b₁] matrices (p=0 only).
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
            if (i != 0 && i != LAYER1_INPUT_SITES - 1 && p > 0) continue;
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

/// Layer 1 horizontal contraction — binary tree
/// Contracts the 17 non-output site matrices (sites 1–17) into a single
/// [b₁][b₁] chain matrix using a balanced binary tree, then absorbs the
/// result into the anchor output site (site 18).
///
/// Tree structure (17 input matrices):
///   Level 0: 8 pairwise products + 1 copy  → 9 matrices (all independent)
///   Level 1: 4 pairwise products + 1 copy  → 5 matrices
///   Level 2: 2 pairwise products + 1 copy  → 3 matrices
///   Level 3: 1 pairwise product  + 1 copy  → 2 matrices
///   Level 4: 1 pairwise product            → 1 matrix (chain result)
///
/// Each level completes in ~1 cycle: all products within a level are
/// independent and fully unrolled, with critical path = mul + add ≈ 4.1 ns
/// at b₁=2. Inter-level register boundaries are created naturally by the
/// data dependency between levels.
///
/// Final absorb:
///   l1_contracted[1][p][l][r] = Σ_b chain[l][b] × anchor[p][b][r]
void layer1_contract_latency_aggressive(
    const smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    // Full partition: tree reads all 17 non-output sites simultaneously
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=1
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=2
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=4

    // ====================================================================
    // Pass 0: Copy boundary output site 0
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
    // Tree contraction: sites 1–17 → single [b₁][b₁] chain matrix
    // ====================================================================
    // All intermediate buffers are fully partitioned registers.
    // Within each level, all products are independent and fully unrolled.
    // Between levels, data dependencies create natural register boundaries.

    // Level 0 output: 9 matrices
    smpo_t t0[9][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=t0 complete

    // Level 0: pairs (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), copy [17]
    for (int pair = 0; pair < 8; pair++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc += l1_out[1 + pair * 2][0][l][b] *
                           l1_out[2 + pair * 2][0][b][r];
                }
                t0[pair][l][r] = acc;
            }
        }
    }
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            t0[8][l][r] = l1_out[17][0][l][r];
        }
    }

    // Level 1 output: 5 matrices
    smpo_t t1[5][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=t1 complete

    for (int pair = 0; pair < 4; pair++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc += t0[pair * 2][l][b] * t0[pair * 2 + 1][b][r];
                }
                t1[pair][l][r] = acc;
            }
        }
    }
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            t1[4][l][r] = t0[8][l][r];
        }
    }

    // Level 2 output: 3 matrices
    smpo_t t2[3][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=t2 complete

    for (int pair = 0; pair < 2; pair++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc += t1[pair * 2][l][b] * t1[pair * 2 + 1][b][r];
                }
                t2[pair][l][r] = acc;
            }
        }
    }
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            t2[2][l][r] = t1[4][l][r];
        }
    }

    // Level 3 output: 2 matrices
    smpo_t t3[2][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=t3 complete

    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            smpo_t acc = 0;
            #pragma HLS BIND_OP variable=acc op=mul impl=fabric
            for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                #pragma HLS UNROLL
                acc += t2[0][l][b] * t2[1][b][r];
            }
            t3[0][l][r] = acc;
        }
    }
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            t3[1][l][r] = t2[2][l][r];
        }
    }

    // Level 4: final product → single chain matrix
    smpo_t chain[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=chain complete

    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            smpo_t acc = 0;
            #pragma HLS BIND_OP variable=acc op=mul impl=fabric
            for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                #pragma HLS UNROLL
                acc += t3[0][l][b] * t3[1][b][r];
            }
            chain[l][r] = acc;
        }
    }

    // ====================================================================
    // Absorb: contract chain into anchor site 18
    // ====================================================================
    // l1_contracted[1][p][l][r] = Σ_b chain[l][b] × anchor[p][b][r]
    // Anchor is site 18 (right boundary output). The right bond of site 18
    // is trivial (=1), so r=0 entries carry the signal; r>0 entries are zero
    // due to zero-padded boundary weights.
    for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
        #pragma HLS UNROLL
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc += chain[l][b] * l1_out[LAYER1_INPUT_SITES - 1][p][b][r];
                }
                l1_contracted[1][p][l][r] = acc;
            }
        }
    }
}

/// Layer 1 horizontal contraction — bidirectional matrix sweep
/// Mirrors the single 19→1 sideways_contract pattern exactly:
///   Init → Interleaved sweep (II=2, fabric) → Merge → Absorb
///
/// Key difference from single: environments are [b₁][b₁] matrices
/// instead of [b] vectors, because there are no trivial boundary bonds
/// in the interior chain (sites 1–17).
///
/// Per-cycle hardware: 2 × b₁³ = 16 fabric muls (vs 2 × b² = 32 in single)
///
/// Structure for 17 non-output sites (1–17):
///   Left init:  site 1              (left_env = T_1)
///   Left sweep: sites 2→8           (7 steps)
///   Right init: site 17             (right_env = T_17)
///   Right sweep: sites 16→10        (7 steps)
///   Merge at site 9:
///     chain[l][r] = Σ_{b1,b2} left_env[l][b1] · T_9[b1][b2] · right_env[b2][r]
///   Absorb into anchor (site 18):
///     l1_contracted[1][p][l][r] = Σ_b chain[l][b] · T_18[p][b][r]
void layer1_contract(
    const smpo_t l1_out[LAYER1_INPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND],
    smpo_t l1_contracted[LAYER1_OUTPUT_SITES][LAYER1_PHYS_OUT][LAYER1_SMPO_BOND][LAYER1_SMPO_BOND]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l1_out complete dim=4
 
    // ====================================================================
    // Pass 0: Copy boundary output site 0
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
    // Matrix environments (b₁ × b₁ = 2 × 2)
    // ====================================================================
    smpo_t left_env[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    smpo_t right_env[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=left_env complete
    #pragma HLS ARRAY_PARTITION variable=right_env complete
 
    // ====================================================================
    // Initialization: copy boundary matrices (1 cycle, fully unrolled)
    // ====================================================================
    // left_env = site 1 (leftmost non-output, p=0)
    // right_env = site 17 (rightmost non-output, p=0)
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            left_env[l][r]  = l1_out[1][0][l][r];
            right_env[l][r] = l1_out[LAYER1_INPUT_SITES - 2][0][l][r];
        }
    }
 
    // ====================================================================
    // Interleaved sweep: 7 steps, both wings per step
    // ====================================================================
    // Left:  sites 2→8,  left_env[l][r]  = Σ_b left_env[l][b]  · T_s[b][r]
    // Right: sites 16→10, right_env[l][r] = Σ_b T_s[l][b] · right_env[b][r]
    //
    // Fabric muls ensure combinational evaluation — carried dependence
    // on left_env/right_env resolves within II=2 window.
    for (int s = 1; s <= 7; s++) {
        #pragma HLS PIPELINE II=2
 
        const int left_site  = s + 1;                       // 2,3,4,5,6,7,8
        const int right_site = LAYER1_INPUT_SITES - 2 - s;  // 16,15,14,13,12,11,10
 
        smpo_t next_left[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
        smpo_t next_right[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
        #pragma HLS ARRAY_PARTITION variable=next_left complete
        #pragma HLS ARRAY_PARTITION variable=next_right complete
 
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
 
                // Left: matrix-matrix product
                smpo_t acc_l = 0;
                #pragma HLS BIND_OP variable=acc_l op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc_l += left_env[l][b] * l1_out[left_site][0][b][r];
                }
                next_left[l][r] = acc_l;
 
                // Right: matrix-matrix product
                smpo_t acc_r = 0;
                #pragma HLS BIND_OP variable=acc_r op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc_r += l1_out[right_site][0][l][b] * right_env[b][r];
                }
                next_right[l][r] = acc_r;
            }
        }
 
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                left_env[l][r]  = next_left[l][r];
                right_env[l][r] = next_right[l][r];
            }
        }
    }
 
    // ====================================================================
    // Merge at center site 9: matrix × matrix × matrix
    // ====================================================================
    // chain[l][r] = Σ_{b1,b2} left_env[l][b1] · T_9[b1][b2] · right_env[b2][r]
    // Site 9 is non-output (p=0 only), center of 17-site chain.
    smpo_t chain[LAYER1_SMPO_BOND][LAYER1_SMPO_BOND];
    #pragma HLS ARRAY_PARTITION variable=chain complete
 
    for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
        #pragma HLS UNROLL
        for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
            #pragma HLS UNROLL
            smpo_t acc = 0;
            #pragma HLS BIND_OP variable=acc op=mul impl=fabric
            for (int b1 = 0; b1 < LAYER1_SMPO_BOND; b1++) {
                #pragma HLS UNROLL
                for (int b2 = 0; b2 < LAYER1_SMPO_BOND; b2++) {
                    #pragma HLS UNROLL
                    acc += left_env[l][b1] *
                           l1_out[9][0][b1][b2] *
                           right_env[b2][r];
                }
            }
            chain[l][r] = acc;
        }
    }
 
    // ====================================================================
    // Absorb: contract chain into anchor site 18
    // ====================================================================
    // l1_contracted[1][p][l][r] = Σ_b chain[l][b] · T_18[p][b][r]
    for (int p = 0; p < LAYER1_PHYS_OUT; p++) {
        #pragma HLS PIPELINE II=1
        for (int l = 0; l < LAYER1_SMPO_BOND; l++) {
            #pragma HLS UNROLL
            for (int r = 0; r < LAYER1_SMPO_BOND; r++) {
                #pragma HLS UNROLL
                smpo_t acc = 0;
                #pragma HLS BIND_OP variable=acc op=mul impl=fabric
                for (int b = 0; b < LAYER1_SMPO_BOND; b++) {
                    #pragma HLS UNROLL
                    acc += chain[l][b] * l1_out[LAYER1_INPUT_SITES - 1][p][b][r];
                }
                l1_contracted[1][p][l][r] = acc;
            }
        }
    }
}

/// ===== Layer 2: 2→1 SMPO =====

/// Layer 2 vertical contraction with composite bond indexing
/// Contracts the physical index of the 2-site MPS (bond b₁=2) with Layer 2
/// SMPO weights (bond b₂=2). Output tensors carry composite bonds B=4.
///
/// l2_out[i][pout][lc][rc] = Σ_pin MPS[i][pin][lp][rp] × W2[i][pin][pout][ls][rs]
///   where lc = lp × b₂ + ls,  rc = rp × b₂ + rs
///
/// Site 0: non-output. Site 1: output (LAYER2_OUTPUT_SITE=1).
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
                                acc += l1_contracted[i][pin][lp][rp] *
                                       weights2[i][pin][pout][ls][rs];
                            }
                            l2_out[i][pout][lc][rc] = acc;
                        }
                    }
                }
            }
        }
    }
}

/// Layer 2 sideways contraction (specialized for x=2)
/// With only 2 sites and output at site 1 (right boundary), there is no
/// sweep phase. Site 0 provides the left environment; the output site's
/// right bond is trivial (boundary).
///
/// output[p] = Σ_l l2_out[0][0][0][l] × l2_out[1][p][l][0]
///
/// This is a simple dot product of length B=4 per physical output.
void layer2_contract(
    const smpo_t l2_out[LAYER1_OUTPUT_SITES][LAYER2_PHYS_OUT][LAYER2_COMPOSITE_BOND][LAYER2_COMPOSITE_BOND],
    smpo_t output[LAYER2_PHYS_OUT]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=3
    #pragma HLS ARRAY_PARTITION variable=l2_out complete dim=4

    // 2-site contraction with output at site 0 (left boundary):
    //   output[p] = Σ_r l2_out[0][p][0][r] × l2_out[1][0][r][0]
    //
    // Site 0 left boundary: left composite bond trivial (lc=0)
    // Site 1 right boundary: right composite bond trivial (rc=0)
    // Shared index r: right bond of site 0 = left bond of site 1
    for (int p = 0; p < LAYER2_PHYS_OUT; p++) {
        #pragma HLS PIPELINE II=1
        smpo_t acc = 0;
        #pragma HLS BIND_OP variable=acc op=mul impl=fabric
        for (int r = 0; r < LAYER2_COMPOSITE_BOND; r++) {
            #pragma HLS UNROLL
            acc += l2_out[0][p][0][r] * l2_out[1][0][r][0];
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

    // Array partitioning
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1_local complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights1_local complete dim=3
    #pragma HLS ARRAY_PARTITION variable=weights2_local complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2_local complete dim=3
    // l1_out fully partitioned: tree reads all sites simultaneously
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

    // Phase 1: Layer 1 — apply SMPO and tree-contract to 2-site MPS
    layer1_smpo(encoded_input, weights1_local, l1_out);
    layer1_contract_latency_aggressive(l1_out, l1_contracted);

    // Phase 2: Layer 2 — apply SMPO with composite indexing, contract to scalar
    layer2_smpo(l1_contracted, weights2_local, l2_out);
    layer2_contract(l2_out, out_vec);

    // Phase 3: Compute score and trigger decision
    compute_norm_squared(out_vec, &final_norm_squared);
    *norm_score = (data_t)final_norm_squared;
    *trigger_decision = (*norm_score < data_t(25.0)) || (*norm_score > data_t(75.0));
}