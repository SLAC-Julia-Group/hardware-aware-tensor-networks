"""
HLS Generator - Skeleton for modular C++ generation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass


class Block(ABC):
    """Base class for HLS code generation blocks"""
    
    @abstractmethod
    def generate_declarations(self) -> str:
        """Generate variable declarations for this block"""
        pass
    
    @abstractmethod
    def generate_compute(self) -> str:
        """Generate computation code for this block"""
        pass
    
    @abstractmethod
    def get_output_info(self) -> Dict[str, Any]:
        """Return info about this block's output (dims, etc)"""
        pass


class DataEmbedding(Block):
    """Input data embedding block"""
    def __init__(self, input_sites: int, features: int, embedding_type: str):
        self.input_sites = input_sites
        self.features = features
        self.embedding_type = embedding_type
    
    def generate_declarations(self) -> str:
        if self.embedding_type in ["kinematic", "kinematicNorm"]:
            return f"""    // Raw input features (57 particle kinematics)
    data_t raw_input[57];
    data_t encoded_input[INPUT_SITES][INPUT_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2"""
        else:
            return f"""    // Raw input and embedded data
    data_t raw_input[INPUT_SITES];
    data_t encoded_input[INPUT_SITES][INPUT_FEATURES];
    #pragma HLS ARRAY_PARTITION variable=encoded_input complete dim=2"""
    
    def generate_compute(self) -> str:
        if self.embedding_type == "kinematic" or self.embedding_type == "kinematicNorm":
            return self._generate_kinematic_embedding()
        elif self.embedding_type == "trigonometric":
            return f"""    // Copy input data
    for (int i = 0; i < INPUT_SITES; i++) {{
        raw_input[i] = input[i];
    }}
    
    // Trigonometric embedding: [cos(πx/2), sin(πx/2)]
    for (int i = 0; i < INPUT_SITES; i++) {{
        #pragma HLS PIPELINE II=1
        data_t angle = M_PI * raw_input[i] / 2.0;
        encoded_input[i][0] = hls::cos(angle);
        encoded_input[i][1] = hls::sin(angle);
    }}"""
        elif self.embedding_type == "polynomial":
            return f"""    // Copy input data
    for (int i = 0; i < INPUT_SITES; i++) {{
        raw_input[i] = input[i];
    }}
    
    // Polynomial embedding: [1, x]
    for (int i = 0; i < INPUT_SITES; i++) {{
        #pragma HLS PIPELINE II=1
        encoded_input[i][0] = 1.0;
        encoded_input[i][1] = raw_input[i];
    }}"""
        else:
            return f"    // TODO: Unsupported embedding type: {self.embedding_type}"
    
    def get_output_info(self):
        return {"sites": self.input_sites, "features": self.features}

    def _generate_kinematic_embedding(self) -> str:
        """Generate kinematic embedding: 57→19 particles with optional normalization"""
        code = ["""    // Copy 57 raw input features
    for (int i = 0; i < 57; i++) {
        #pragma HLS PIPELINE II=1
        raw_input[i] = input[i];
    }
    
    // Reshape to 19 particles × 3 features (pt, eta, phi)
    int raw_idx = 0;
    
    // Particle 0: MET (eta=0 hardcoded)
    encoded_input[0][0] = raw_input[raw_idx++];  // pt
    encoded_input[0][1] = raw_input[raw_idx++];  // eta (dummy)
    encoded_input[0][2] = raw_input[raw_idx++];  // phi
    
    // Particles 1-4: Electrons
    for (int i = 1; i <= 4; i++) {
        #pragma HLS UNROLL
        encoded_input[i][0] = raw_input[raw_idx++];  // pt
        encoded_input[i][1] = raw_input[raw_idx++];  // eta
        encoded_input[i][2] = raw_input[raw_idx++];  // phi
    }
    
    // Particles 5-8: Muons
    for (int i = 5; i <= 8; i++) {
        #pragma HLS UNROLL
        encoded_input[i][0] = raw_input[raw_idx++];  // pt
        encoded_input[i][1] = raw_input[raw_idx++];  // eta
        encoded_input[i][2] = raw_input[raw_idx++];  // phi
    }
    
    // Particles 9-18: Jets
    for (int i = 9; i <= 18; i++) {
        #pragma HLS UNROLL
        encoded_input[i][0] = raw_input[raw_idx++];  // pt
        encoded_input[i][1] = raw_input[raw_idx++];  // eta
        encoded_input[i][2] = raw_input[raw_idx++];  // phi
    }"""]
        
        # Add normalization if requested
        if self.embedding_type == "kinematicNorm":
            code.append("""
    // Apply physics-aware normalization
    // Particle 0: MET
    encoded_input[0][0] = encoded_input[0][0] / 1200.0;
    encoded_input[0][1] = (encoded_input[0][1] + 5.0) / 10.0;
    encoded_input[0][2] = (encoded_input[0][2] + M_PI) / (2.0 * M_PI);
    
    // Particles 1-4: Electrons
    for (int i = 1; i <= 4; i++) {
        encoded_input[i][0] = encoded_input[i][0] / 1200.0;
        encoded_input[i][1] = (encoded_input[i][1] + 5.0) / 10.0;
        encoded_input[i][2] = (encoded_input[i][2] + M_PI) / (2.0 * M_PI);
    }
    
    // Particles 5-8: Muons
    for (int i = 5; i <= 8; i++) {
        encoded_input[i][0] = encoded_input[i][0] / 800.0;
        encoded_input[i][1] = (encoded_input[i][1] + 5.0) / 10.0;
        encoded_input[i][2] = (encoded_input[i][2] + M_PI) / (2.0 * M_PI);
    }
    
    // Particles 9-18: Jets
    for (int i = 9; i <= 18; i++) {
        encoded_input[i][0] = encoded_input[i][0] / 600.0;
        encoded_input[i][1] = (encoded_input[i][1] + 5.0) / 10.0;
        encoded_input[i][2] = (encoded_input[i][2] + M_PI) / (2.0 * M_PI);
    }""")
        
        return "\n".join(code)

class ApplySMPO(Block):
    """Apply SMPO to MPS - vertical contraction only"""
    def __init__(self, layer_name: str, input_sites: int, spacing: int, 
                 bond_dim: int, phys_in: int, phys_out: int, 
                 input_bond: int = 1, layer_idx: int = 0, **kwargs):
        self.layer_name = layer_name
        self.input_sites = input_sites
        self.spacing = spacing
        self.bond_dim = bond_dim
        self.phys_in = phys_in
        self.phys_out = phys_out
        self.input_bond = input_bond  # Bond from previous layer
        self.layer_idx = layer_idx
        self.composite_left = input_bond * bond_dim
        self.composite_right = input_bond * bond_dim
    
    def generate_declarations(self) -> str:
        layer_num = self.layer_idx + 1
        return f"""    // {self.layer_name}: SMPO output tensor
    data_t {self.layer_name}_out[LAYER{layer_num}_INPUT_SITES][LAYER{layer_num}_PHYS_OUT][LAYER{layer_num}_COMPOSITE_BOND][LAYER{layer_num}_COMPOSITE_BOND] = {{0}};
    #pragma HLS ARRAY_PARTITION variable={self.layer_name}_out complete dim=2"""
    
    def generate_compute(self) -> str:
        layer_num = self.layer_idx + 1
        
        if self.layer_idx == 0:
            # First layer - simple case, input MPS has bond dimension 1
            return f"""    // Apply {self.layer_name} SMPO (first layer, no composite indexing needed)
    for (int i = 0; i < LAYER{layer_num}_INPUT_SITES; i++) {{
        for (int p_out = 0; p_out < LAYER{layer_num}_PHYS_OUT; p_out++) {{
            // Skip computation for non-output sites when p_out > 0 (weights are zero)
            if (i % LAYER{layer_num}_SPACING != 0 && p_out > 0) continue;
            
            for (int l = 0; l < LAYER{layer_num}_SMPO_BOND; l++) {{
                for (int r = 0; r < LAYER{layer_num}_SMPO_BOND; r++) {{
                    #pragma HLS PIPELINE II=1
                    data_t acc = 0;
                    for (int f = 0; f < INPUT_FEATURES; f++) {{
                        acc += encoded_input[i][f] * weights{layer_num}[i][f][p_out][l][r];
                    }}
                    {self.layer_name}_out[i][p_out][l][r] = acc;
                }}
            }}
        }}
    }}"""
        else:
            # Later layers - need composite indexing
            prev_layer = f"layer{self.layer_idx}_contracted"
            return f"""    // Apply {self.layer_name} SMPO with composite indexing
    for (int site = 0; site < LAYER{layer_num}_INPUT_SITES; site++) {{
        // Skip p_out>0 for non-output sites
        bool is_output_site = (site % LAYER{layer_num}_SPACING == 0);
        
        for (int p_out = 0; p_out < LAYER{layer_num}_PHYS_OUT; p_out++) {{
            if (!is_output_site && p_out > 0) continue;
            
            for (int l_prev = 0; l_prev < LAYER{layer_num}_INPUT_BOND; l_prev++) {{
                for (int r_prev = 0; r_prev < LAYER{layer_num}_INPUT_BOND; r_prev++) {{
                    for (int l_smpo = 0; l_smpo < LAYER{layer_num}_SMPO_BOND; l_smpo++) {{
                        for (int r_smpo = 0; r_smpo < LAYER{layer_num}_SMPO_BOND; r_smpo++) {{
                            #pragma HLS PIPELINE II=1
                            // Composite indices
                            int left_composite = l_prev * LAYER{layer_num}_SMPO_BOND + l_smpo;
                            int right_composite = r_prev * LAYER{layer_num}_SMPO_BOND + r_smpo;
                            
                            data_t acc = 0;
                            for (int p_in = 0; p_in < LAYER{layer_num}_PHYS_IN; p_in++) {{
                                acc += {prev_layer}[site][p_in][l_prev][r_prev] * 
                                       weights{layer_num}[site][p_in][p_out][l_smpo][r_smpo];
                            }}
                            
                            {self.layer_name}_out[site][p_out][left_composite][right_composite] = acc;
                        }}
                    }}
                }}
            }}
        }}
    }}"""
    
    def get_output_info(self):
        return {
            "sites": self.input_sites,
            "spacing": self.spacing, 
            "bond_left": self.composite_left,
            "bond_right": self.composite_right,
            "phys_out": self.phys_out
        }


class ContractNeighbors(Block):
    """Contract sites based on spacing - handles variable spacing with proper edge cases"""
    def __init__(self, layer_name: str, layer_idx: int, input_sites: int, spacing: int, 
                 output_sites: int, phys_dim: int, bond_left: int, 
                 bond_right: int, **kwargs):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.input_sites = input_sites
        self.spacing = spacing
        self.output_sites = output_sites
        self.phys_dim = phys_dim
        self.bond_left = bond_left
        self.bond_right = bond_right
    
    def generate_declarations(self) -> str:
        layer_num = self.layer_idx + 1
        
        # Determine bond dimensions (account for truncation if needed)
        if hasattr(self, 'truncated_bond'):
            bond_str = f"LAYER{layer_num}_TRUNCATED_BOND"
        else:
            bond_str = f"LAYER{layer_num}_COMPOSITE_BOND"
            
        return f"""    // {self.layer_name}: Contracted output [{self.output_sites}]
    data_t {self.layer_name}_contracted[LAYER{layer_num}_OUTPUT_SITES][LAYER{layer_num}_PHYS_OUT][{bond_str}][{bond_str}] = {{0}};
    #pragma HLS ARRAY_PARTITION variable={self.layer_name}_contracted complete dim=2"""
    
    def generate_compute(self) -> str:
        layer_num = self.layer_idx + 1
        
        # For spacing=1, no contraction needed
        if self.spacing == 1:
            return f"""    // No contraction needed for spacing=1, just copy
    for (int i = 0; i < LAYER{layer_num}_OUTPUT_SITES; i++) {{
        for (int p = 0; p < LAYER{layer_num}_PHYS_OUT; p++) {{
            for (int l = 0; l < LAYER{layer_num}_COMPOSITE_BOND; l++) {{
                for (int r = 0; r < LAYER{layer_num}_COMPOSITE_BOND; r++) {{
                    {self.layer_name}_contracted[i][p][l][r] = {self.layer_name}_out[i][p][l][r];
                }}
            }}
        }}
    }}"""
        
        # Generate the contraction code
        code = [f"""    // Contract neighbors with spacing={self.spacing}
    for (int out_site = 0; out_site < LAYER{layer_num}_OUTPUT_SITES; out_site++) {{
        int first_site = out_site * LAYER{layer_num}_SPACING;
        
        // Determine how many sites in this group (handle edge case)
        int sites_in_group = LAYER{layer_num}_SPACING;
        if (first_site + LAYER{layer_num}_SPACING > LAYER{layer_num}_INPUT_SITES) {{
            sites_in_group = LAYER{layer_num}_INPUT_SITES - first_site;
        }}
        
        // Only first site's physical index matters (others are non-output sites with p=0)
        for (int p0 = 0; p0 < LAYER{layer_num}_PHYS_OUT; p0++) {{
            for (int l_first = 0; l_first < LAYER{layer_num}_COMPOSITE_BOND; l_first++) {{
                for (int r_last = 0; r_last < LAYER{layer_num}_COMPOSITE_BOND; r_last++) {{
                    #pragma HLS PIPELINE II=1
                    data_t acc = 0;
                    
                    if (sites_in_group == 1) {{
                        // Single site - just copy
                        acc = {self.layer_name}_out[first_site][p0][l_first][r_last];
                    }}"""]
        
        # Generate cases for different group sizes (2 to spacing)
        for group_size in range(2, self.spacing + 1):
            code.append(f"""                    else if (sites_in_group == {group_size}) {{""")
            
            # Generate bond loops
            bond_loops = []
            for b in range(group_size - 1):
                indent = "                        " + "    " * b
                bond_loops.append(f"{indent}for (int bond{b} = 0; bond{b} < LAYER{layer_num}_COMPOSITE_BOND; bond{b}++) {{")
            
            # Add the loops
            code.extend(bond_loops)
            
            # Generate the product calculation
            indent = "                        " + "    " * (group_size - 1)
            
            # First site
            code.append(f"{indent}data_t prod = {self.layer_name}_out[first_site][p0][l_first][bond0];")
            
            # Middle sites (all with p=0)
            for s in range(1, group_size - 1):
                prev_bond = f"bond{s-1}"
                next_bond = f"bond{s}"
                code.append(f"{indent}prod *= {self.layer_name}_out[first_site + {s}][0][{prev_bond}][{next_bond}];")
            
            # Last site (also with p=0)
            last_bond = f"bond{group_size-2}"
            code.append(f"{indent}prod *= {self.layer_name}_out[first_site + {group_size-1}][0][{last_bond}][r_last];")
            code.append(f"{indent}acc += prod;")
            
            # Close bond loops
            for b in range(group_size - 2, -1, -1):
                indent = "                        " + "    " * b
                code.append(f"{indent}}}")
            
            code.append("                    }")
        
        # Close the main loops and store result
        code.extend([
            "                    ",
            f"                    {self.layer_name}_contracted[out_site][p0][l_first][r_last] = acc;",
            "                }",
            "            }",
            "        }",
            "    }"
        ])
        
        return "\n".join(code)
    
    def get_output_info(self):
        return {
            "sites": self.output_sites,
            "phys_dim": self.phys_dim,
            "bond_left": self.bond_left,
            "bond_right": self.bond_right
        }


class ComputeNorm(Block):
    """Compute the norm squared of final MPS as score"""
    def __init__(self, num_sites: int, layer_name: str, layer_idx: int, 
                 is_truncated: bool = False, **kwargs):
        self.num_sites = num_sites
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.is_truncated = is_truncated
    
    def generate_declarations(self) -> str:
        if self.layer_idx == -1:
            return """    // Initial norm computation
    data_t initial_norm_squared = 0.0;"""
        else:
            return """    // Final norm computation
    data_t norm_squared = 0.0;"""
    
    def generate_compute(self) -> str:
        # Special case: input MPS with trivial bonds (bond_dim=1)
        if self.layer_idx == -1:
            return self._generate_input_norm_compute()

        layer_num = self.layer_idx + 1
        
        # Determine bond constant name
        if self.is_truncated:
            bond_const = f"LAYER{layer_num}_TRUNCATED_BOND"
        else:
            bond_const = f"LAYER{layer_num}_COMPOSITE_BOND"
        
        norm_var = "initial_norm_squared" if self.layer_idx == -1 else "norm_squared"
        
        code = [f"""    // Compute norm squared of {self.num_sites}-site MPS
    {norm_var} = 0.0;
    """]
        
        # Generate nested loops for physical indices
        for i in range(self.num_sites):
            indent = "    " * (i + 1)
            code.append(f"{indent}for (int p{i} = 0; p{i} < LAYER{layer_num}_PHYS_OUT; p{i}++) {{")
        
        # Generate nested loops for bond indices (num_sites - 1 bonds)
        for b in range(self.num_sites - 1):
            indent = "    " * (self.num_sites + b + 1)
            code.append(f"{indent}for (int bond{b}{b+1} = 0; bond{b}{b+1} < {bond_const}; bond{b}{b+1}++) {{")
        
        # Add pipeline pragma
        indent = "    " * (2 * self.num_sites)
        code.append(f"{indent}#pragma HLS PIPELINE II=1")
        code.append("")
        
        # Extract tensor values and compute product
        code.append(f"{indent}data_t product = 1.0;")
        for i in range(self.num_sites):
            left_bond = f"bond{i-1}{i}" if i > 0 else "0"
            right_bond = f"bond{i}{i+1}" if i < self.num_sites - 1 else "0"
            code.append(f"{indent}product *= {self.layer_name}[{i}][p{i}][{left_bond}][{right_bond}];")
        
        # Accumulate squared product
        code.append(f"{indent}{norm_var} += product * product;")
        
        # Close all loops
        for i in range(2 * self.num_sites - 1):
            indent = "    " * (2 * self.num_sites - 1 - i)
            code.append(f"{indent}}}")
        
        return "\n".join(code)

    def _generate_input_norm_compute(self) -> str:
        """Compute norm for input MPS with trivial bonds (bond_dim=1)"""
        code = [f"""    // Compute norm squared of input MPS (trivial bonds)
    initial_norm_squared = 0.0;
    
    for (int site = 0; site < {self.num_sites}; site++) {{
        for (int p = 0; p < INPUT_FEATURES; p++) {{
            #pragma HLS PIPELINE II=1
            data_t val = {self.layer_name}[site][p];
            initial_norm_squared += val * val;
        }}
    }}"""]
        
        return "\n".join(code)

    def get_output_info(self):
        return {"has_norm_score": True}

class ApplyReLU(Block):
    """Apply ReLU activation element-wise to MPS tensors"""
    
    def __init__(self, layer_name: str, layer_idx: int, num_sites: int, 
                 phys_dim: int, bond_dim: str):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.num_sites = num_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
    
    def generate_declarations(self) -> str:
        return f"    // ReLU applied to {self.layer_name}"
    
    def generate_compute(self) -> str:
        layer_num = self.layer_idx + 1
        
        code = [f"""    // Apply ReLU activation: max(0, x) element-wise
    for (int site = 0; site < {self.num_sites}; site++) {{
        for (int p = 0; p < LAYER{layer_num}_PHYS_OUT; p++) {{
            for (int l = 0; l < {self.bond_dim}; l++) {{
                for (int r = 0; r < {self.bond_dim}; r++) {{
                    #pragma HLS PIPELINE II=1
                    if ({self.layer_name}[site][p][l][r] < 0) {{
                        {self.layer_name}[site][p][l][r] = 0;
                    }}
                }}
            }}
        }}
    }}"""]
        
        return "\n".join(code)
    
    def get_output_info(self):
        return {"has_relu": True}

class TriggerDecision(Block):
    """Make trigger decision based on norm ratio window"""
    
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def generate_declarations(self) -> str:
        return """    // Norm ratio and trigger decision
    data_t norm_ratio_value;
    bool trigger_pass;"""
    
    def generate_compute(self) -> str:
        return f"""    // Compute norm ratio
    norm_ratio_value = norm_squared / initial_norm_squared;
    
    // Trigger decision: pass if norm_ratio in [{self.lower_bound}, {self.upper_bound}]
    trigger_pass = (norm_ratio_value < {self.lower_bound}) || (norm_ratio_value > {self.upper_bound});"""
    
    def get_output_info(self):
        return {"has_trigger": True}

class TruncateBonds(Block):
    """Optional bond truncation"""
    def __init__(self, layer_name: str, layer_idx: int, max_bond: int, method: str = "slice"):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.max_bond = max_bond
        self.method = method
    
    def generate_declarations(self) -> str:
        layer_num = self.layer_idx + 1
        return f"    // {self.layer_name}: Truncation to bond={self.max_bond}"
    
    def generate_compute(self) -> str:
        layer_num = self.layer_idx + 1
        return f"""
// Truncate bonds using {self.method} method
// Max bond: LAYER{layer_num}_TRUNCATED_BOND
// TODO: Implement truncation
"""
    
    def get_output_info(self):
        return {"truncated_bond": self.max_bond}


class HLSGenerator:
    """Main HLS code generator"""
    
    def __init__(self, model_config):
        self.model = model_config
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> List[Block]:
        """Build processing pipeline from model config"""
        pipeline = []
        
        # Input embedding
        pipeline.append(DataEmbedding(
            self.model.input_sites,
            self.model.input_features,
            self.model.embedding_type
        ))
        
        # Compute initial norm squared (before cascade)
        pipeline.append(ComputeNorm(
            num_sites=self.model.input_sites,
            layer_name="encoded_input",
            layer_idx=-1,  # Special index for input
            is_truncated=False
        ))

        # Track bond dimensions through layers
        current_bond = 1  # Initial MPS has bond dimension 1
        
        # Process each layer
        for i, layer in enumerate(self.model.layers):
            # Apply SMPO
            smpo_block = ApplySMPO(
                layer.name,
                layer.input_sites,
                layer.spacing,
                layer.bond_dim,
                layer.phys_in,
                layer.phys_out,
                input_bond=current_bond,
                layer_idx=i
            )
            pipeline.append(smpo_block)
            
            # Get output info from SMPO
            smpo_output = smpo_block.get_output_info()
            composite_bond = current_bond * layer.bond_dim
            
            # Contract
            contract_block = ContractNeighbors(
                layer.name,
                i,  # layer_idx
                layer.input_sites,
                layer.spacing,
                layer.output_sites,
                layer.phys_out,
                composite_bond,  # bond dimensions after SMPO
                composite_bond
            )
            pipeline.append(contract_block)

            # Apply ReLU if enabled for this layer
            if layer.enable_relu:
                bond_str = f"LAYER{i+1}_COMPOSITE_BOND"
                pipeline.append(ApplyReLU(
                    f"{layer.name}_contracted",
                    i,
                    layer.output_sites,
                    layer.phys_out,
                    bond_str
                ))

            # Optional truncation
            if layer.truncation.enabled:
                pipeline.append(TruncateBonds(
                    layer.name,
                    i,  # layer_idx
                    layer.truncation.max_bond,
                    layer.truncation.method
                ))
                current_bond = min(composite_bond, layer.truncation.max_bond)
                # Mark the contract block as having truncated bonds
                contract_block.truncated_bond = layer.truncation.max_bond
            else:
                current_bond = composite_bond
        
        # Add norm computation if enabled
        if self.model.output_config.compute_norm:
            last_layer = self.model.layers[-1]
            last_layer_idx = len(self.model.layers) - 1

            pipeline.append(ComputeNorm(
                num_sites=last_layer.output_sites,
                layer_name=f"layer{last_layer_idx + 1}_contracted",
                layer_idx=last_layer_idx,
                is_truncated=last_layer.truncation.enabled
            ))
            
            # Make trigger decision based on norm window
            pipeline.append(TriggerDecision(
                lower_bound=self.model.output_config.norm_window_lower,
                upper_bound=self.model.output_config.norm_window_upper
            ))

        return pipeline
    
    def generate(self) -> str:
        """Generate complete HLS C++ code"""
        code = []
        
        # Header
        code.append(self._generate_header())
        
        # Includes
        code.append(self._generate_includes())
        
        # Constants
        code.append(self._generate_constants())
        
        # Function signature
        code.append(self._generate_function_signature())
        
        # Read input
        code.append("\n    // ===== READ INPUT =====")
        code.append("    // Assume input is already in raw_input array")
        
        # Declarations
        code.append("\n    // ===== DECLARATIONS =====")
        for block in self.pipeline:
            decl = block.generate_declarations()
            if decl:  # Only add if not empty
                code.append(decl)
        
        # Computations
        code.append("\n    // ===== COMPUTATIONS =====")
        for block in self.pipeline:
            compute = block.generate_compute()
            if compute:  # Only add if not empty
                code.append(compute)
        
        # Footer
        code.append(self._generate_footer())
        
        return "\n".join(code)
    
    def _generate_header(self) -> str:
        header = f"""//
// Generated HLS Code for: {self.model.name}
// Architecture: {self._get_architecture_string()}"""
        
        if self.model.output_config.compute_norm:
            header += "\n// Output: MPS tensors + norm squared score"
        
        header += "\n//\n"
        return header
    
    def _generate_includes(self) -> str:
        includes = """#include <hls_stream.h>
#include <ap_fixed.h>
#include <hls_math.h>
#include <cmath>

typedef float data_t;  // Or ap_fixed<32,16> for fixed-point

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
"""
        return includes
    
    def _generate_constants(self) -> str:
        constants = [f"""// Model constants
#define INPUT_SITES {self.model.input_sites}
#define INPUT_FEATURES {self.model.input_features}"""]
        
        # Track bond dimensions through cascade
        current_bond = 1
        
        for i, layer in enumerate(self.model.layers):
            layer_num = i + 1
            int_bits = layer.precision_word - layer.precision_frac
            constants.append(f"typedef ap_fixed<{layer.precision_word},{int_bits}> layer{layer_num}_t;  // Q{int_bits}.{layer.precision_frac}")
            composite_bond = current_bond * layer.bond_dim
            
            constants.append(f"""
// Layer {layer_num} constants
#define LAYER{layer_num}_INPUT_SITES {layer.input_sites}
#define LAYER{layer_num}_OUTPUT_SITES {layer.output_sites}
#define LAYER{layer_num}_SPACING {layer.spacing}
#define LAYER{layer_num}_PHYS_IN {layer.phys_in}
#define LAYER{layer_num}_PHYS_OUT {layer.phys_out}
#define LAYER{layer_num}_SMPO_BOND {layer.bond_dim}
#define LAYER{layer_num}_INPUT_BOND {current_bond}
#define LAYER{layer_num}_COMPOSITE_BOND {composite_bond}""")
            
            if layer.truncation.enabled:
                constants.append(f"#define LAYER{layer_num}_TRUNCATED_BOND {layer.truncation.max_bond}")
                current_bond = min(composite_bond, layer.truncation.max_bond)
            else:
                current_bond = composite_bond
        
        # Final output dimension
        last_layer = self.model.layers[-1]
        constants.append(f"\n#define OUTPUT_DIM {last_layer.output_sites}")
        
        return "\n".join(constants)
    
    def _generate_function_signature(self) -> str:
        sig = ["void tn_model("]
        
        # Determine input size based on embedding type
        if self.model.embedding_type in ["kinematic", "kinematicNorm"]:
            sig.append(f"    data_t input[57],  // Raw particle features")
        else:
            sig.append(f"    data_t input[INPUT_SITES],")
        
        # Add weight parameters for each layer
        for i, layer in enumerate(self.model.layers):
            layer_num = i + 1
            if i == 0:
                # First layer: [sites][features][phys_out][bond][bond]
                sig.append(f"    const data_t weights{layer_num}[LAYER{layer_num}_INPUT_SITES][INPUT_FEATURES]"
                          f"[LAYER{layer_num}_PHYS_OUT][LAYER{layer_num}_SMPO_BOND][LAYER{layer_num}_SMPO_BOND],")
            else:
                # Later layers: [sites][phys_in][phys_out][bond][bond]
                sig.append(f"    const data_t weights{layer_num}[LAYER{layer_num}_INPUT_SITES]"
                          f"[LAYER{layer_num}_PHYS_IN][LAYER{layer_num}_PHYS_OUT]"
                          f"[LAYER{layer_num}_SMPO_BOND][LAYER{layer_num}_SMPO_BOND],")
        
        # Output parameters
        if self.model.output_config.compute_norm:
            sig.append(f"    data_t output[OUTPUT_DIM],")
            sig.append(f"    data_t* norm_ratio,  // Output: norm squared ratio")
            sig.append(f"    bool* trigger_decision  // Output: trigger pass/fail")
        else:
            sig.append(f"    data_t output[OUTPUT_DIM]")
            
        sig.append(""") {
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi port=input offset=slave
#pragma HLS INTERFACE m_axi port=output offset=slave""")
        
        if self.model.output_config.compute_norm:
            sig.append("#pragma HLS INTERFACE s_axilite port=norm_ratio")
            sig.append("#pragma HLS INTERFACE s_axilite port=trigger_decision")
        
        # Add interface pragmas for weights
        for i in range(len(self.model.layers)):
            sig.append(f"#pragma HLS INTERFACE m_axi port=weights{i+1} offset=slave bundle=gmem")
        
        return "\n".join(sig)
    
    def _generate_footer(self) -> str:
        num_layers = len(self.model.layers)
        last_layer = self.model.layers[-1]
        
        # Determine final bond dimension (after truncation if applicable)
        if last_layer.truncation.enabled:
            final_bond = f"LAYER{num_layers}_TRUNCATED_BOND"
        else:
            final_bond = f"LAYER{num_layers}_COMPOSITE_BOND"
        
        footer = ["""
    // ===== COPY FINAL OUTPUT =====
    // For now, just copy first physical component of each output site
    for (int i = 0; i < OUTPUT_DIM; i++) {
        // Sum over all bonds (full contraction)
        data_t sum = 0;"""]
        
        footer.append(f"""        for (int l = 0; l < {final_bond}; l++) {{
            for (int r = 0; r < {final_bond}; r++) {{
                sum += layer{num_layers}_contracted[i][0][l][r];  // Using p=0 for now
            }}
        }}
        output[i] = sum;
    }}""")
        
        if self.model.output_config.compute_norm:
            footer.append("""
    // Write norm score
    *norm_ratio = norm_ratio_value;
    *trigger_decision = trigger_pass;""")
        
        footer.append("""
} // end tn_model""")
        
        return "\n".join(footer)
    
    def _get_architecture_string(self) -> str:
        """Generate architecture string like '56 → 19 → 7 → 3'"""
        sites = [self.model.input_sites]
        sites.extend([layer.output_sites for layer in self.model.layers])
        return " → ".join(map(str, sites))