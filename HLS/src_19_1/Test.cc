// src/Test.cc
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>

#include "weights.h"
#include "QML_TensorNetwork.h"
#include <hls_math.h>

// Post spectral ordering, the sites are arranged as:
std::map<int, int> particle_map = {
    {0, 8},    // MET
    {1, 7},    // Electron 0
    {2, 4},    // Electron 1
    {3, 1},    // Electron 2
    {4, 0},    // Electron 3
    {5, 9},    // Muon 0
    {6, 16},   // Muon 1
    {7, 17},   // Muon 2
    {8, 18},   // Muon 3
    {9, 11},   // Jet 0
    {10, 13},  // Jet 1
    {11, 14},  // Jet 2
    {12, 12},  // Jet 3
    {13, 10},  // Jet 4
    {14, 6},   // Jet 5
    {15, 15},  // Jet 6
    {16, 5},   // Jet 7
    {17, 3},   // Jet 8
    {18, 2},   // Jet 9
};

/// Load and embed inputs
void load_input(
    const data_t input[INPUT_FEATURES],
    data_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN]
) {
    int feature_idx = 0;

    // Particle 0: MET
    encoded_input[particle_map[0]][0] =  input[feature_idx++] * INV_1200;  // pt
    encoded_input[particle_map[0]][1] = (input[feature_idx++] + OFFSET_5) * INV_10;  // eta
    encoded_input[particle_map[0]][2] = (input[feature_idx++] + OFFSET_PI) * INV_2PI;  // phi
    // Particles 1-4: Electrons
    for (int i = 0; i < 4; i++) {
        encoded_input[particle_map[1+i]][0] = input[feature_idx++] * INV_1200;  // pt
        encoded_input[particle_map[1+i]][1] = (input[feature_idx++] + OFFSET_5) * INV_10;  // eta
        encoded_input[particle_map[1+i]][2] = (input[feature_idx++] + OFFSET_PI) * INV_2PI;  // phi
    }
    // Particles 5-8: Muons
    for (int i = 0; i < 4; i++) {
        encoded_input[particle_map[5+i]][0] = input[feature_idx++] * INV_800;  // pt
        encoded_input[particle_map[5+i]][1] = (input[feature_idx++] + OFFSET_5) * INV_10;  // eta
        encoded_input[particle_map[5+i]][2] = (input[feature_idx++] + OFFSET_PI) * INV_2PI;  // phi
    }
    // Particles 9-18: Jets
    for (int i = 0; i < 10; i++) {
        encoded_input[particle_map[9+i]][0] = input[feature_idx++] * INV_2500;  // pt
        encoded_input[particle_map[9+i]][1] = (input[feature_idx++] + OFFSET_5) * INV_10;  // eta
        encoded_input[particle_map[9+i]][2] = (input[feature_idx++] + OFFSET_PI) * INV_2PI;  // phi
    }
}

/// Load and embed inputs
void scale_input(
    data_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    const data_t norm_factor
) {
    for (int i = 0; i < SMPO_INPUT_SITES; i++) {
        for (int j = 0; j < SMPO_PHYS_IN; j++) {
            encoded_input[i][j] *= norm_factor;
        }
    }
}

/// Compute initial norm squared of the input
void compute_initial_norm(
    const data_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN],
    data_t* initial_norm_squared
) {
    *initial_norm_squared = 1.0;

    for (int site = 0; site < SMPO_INPUT_SITES; site++) {
        data_t site_norm_sq = 0.0;
        for (int phys = 0; phys < SMPO_PHYS_IN; phys++) {
            data_t val = encoded_input[site][phys];
            site_norm_sq += val * val;
        }
        *initial_norm_squared *= site_norm_sq;
    }
}

int main(int argc, char **argv) {

   // Verify weights loaded
   std::cout << "=== WEIGHT VERIFICATION ===" << std::endl;
   std::cout << "Layer 1, Site 0, phys_in=1, phys_out=0:" << std::endl;
   std::cout << "  weights1_data[0][1][0][0][0] = " << weights1_data[0][1][0][0][0] << std::endl;
   std::cout << "Layer 1, Site 9, phys_in=0, phys_out=1:" << std::endl;
   std::cout << "  weights1_data[9][0][1][0][0] = " << weights1_data[9][0][1][0][0] << std::endl;
   std::cout << "============================" << std::endl;

   std::string inDir = "/home/sagar/QiML/Tensor_Network/firmware/shared/tn4ad/";
   std::vector<std::string> samples = {"background_full", "Ato4l", "hToTauTau", "hToTauNu", "LQ"};
   std::string quantization = "_16bit";

   for (const auto& sample : samples) {
      std::cout << "=== Processing sample: " << sample << " ===" << std::endl;
      std::ifstream fin(inDir+"data/inputs/"+sample+".dat");
      std::ifstream fref(inDir+"data/references/19_1/"+sample+".dat");
#ifdef RTL_SIM
      std::string RESULTS_LOG = inDir+"data/results_rtlsim/"+sample+".log";
#else
      std::string RESULTS_LOG = inDir+"data/results_csim/19_1/"+sample+quantization+".log";
#endif
      std::ofstream fout(RESULTS_LOG);

      std::string iline;
      std::string rline;
      int event_count = 0;
      bool has_reference = fref.is_open();

      if (!has_reference) {
         std::cout << "WARNING: Reference file not found, will skip comparison" << std::endl;
      }

      if (fin.is_open()) {
         std::cout << "INFO: Reading input data..." << std::endl;
         
         while (std::getline(fin, iline)) {
            if (event_count % CHECKPOINT_INTERVAL == 0) {
               std::cout << "Processing event " << event_count << std::endl;
            }

            // Parse input line (57 features)
            char* cstr = const_cast<char*>(iline.c_str());
            char* token;
            std::vector<float> input_vec;
            token = strtok(cstr, " ");
            while (token != NULL) {
               input_vec.push_back(atof(token));
               token = strtok(NULL, " ");
            }

            // Prepare inputs
            data_t input[INPUT_FEATURES];
            for (int i = 0; i < INPUT_FEATURES; i++) {
               input[i] = (i < input_vec.size()) ? input_vec[i] : 0.0;
            }
            // Load and normalize input
            data_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN];
            data_t initial_norm_squared;
            load_input(input, encoded_input);
            compute_initial_norm(encoded_input, &initial_norm_squared);
            // std::cout << "initial_norm = " << hls::sqrt(initial_norm_squared) << std::endl;
            data_t norm_factor = data_t(1.)/hls::pow(initial_norm_squared, INV_38);
            // std::cout << "norm_factor = " << norm_factor << std::endl;
            scale_input(encoded_input, norm_factor);
            /*
            if (event_count == 0) {
               std::cout << "=== FIRST EVENT INPUT ===" << std::endl;
               std::cout << "Raw inputs (first 10):" << std::endl;
               for (int i = 0; i < 10; i++) {
                  std::cout << "  input[" << i << "] = " << input[i] << std::endl;
               }
               std::cout << "=======================" << std::endl;
            }
            */
            // Run inference
            data_t norm_score;
            bool trigger_decision;
            tn4ad(encoded_input, weights1_data, &norm_score, &trigger_decision);
            // std::cout << "The norm squared is " << norm_score << std::endl;
            // Write outputs
            fout << norm_score << " " << trigger_decision << std::endl;

            // Checkpoint comparison (if reference available)
            if (has_reference && std::getline(fref, rline)) {
               if (event_count % CHECKPOINT_INTERVAL == 0) {
                  cstr = const_cast<char*>(rline.c_str());
                  std::vector<float> ref_vec;
                  token = strtok(cstr, " ");
                  while (token != NULL) {
                     ref_vec.push_back(atof(token));
                     token = strtok(NULL, " ");
                  }
                  
                  std::cout << "  Reference: norm=" << ref_vec[0] << std::endl;
                  std::cout << "  Computed:  norm=" << float(norm_score) << std::endl;
                  std::cout << "  Percent difference: " << 100.0 * std::abs(float(norm_score) - ref_vec[0]) / (std::abs(ref_vec[0]) + 1e-6) << "%" << std::endl;
               }
            }

            event_count++;
            if(MAX_EVENTS > 0 && event_count >= MAX_EVENTS) {
               break;
            }
         }
         
         fin.close();
         if (has_reference) fref.close();
         std::cout << "INFO: Processed " << event_count << " events" << std::endl;
         
      } else {
         std::cout << "INFO: Unable to open input file, using default zero input." << std::endl;

         // Default zero input test
         data_t input[INPUT_FEATURES] = {0};
         data_t encoded_input[SMPO_INPUT_SITES][SMPO_PHYS_IN] = {{0}};
         data_t norm_score;
         bool trigger_decision;

         tn4ad(encoded_input, weights1_data, &norm_score, &trigger_decision);
         
         std::cout << "Default test - norm_score: " << norm_score 
               << ", trigger: " << trigger_decision << std::endl;
         fout << norm_score << std::endl;
      }

      fout.close();
      std::cout << "INFO: Saved results to " << RESULTS_LOG << std::endl;
      std::cout << "=== Finishing running on sample: " << sample << " ===" << std::endl;
   }
   return 0;
}