"""
Model configuration parser for TN HLS generator
Supports YAML format with automatic spacing inference
"""

import yaml
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# Hardware configuration
@dataclass
class HardwareConfig:
    """Hardware optimization hints"""
    pipeline_ii: int = 1
    array_partition_dims: List[int] = field(default_factory=lambda: [2])
    resource_sharing: bool = False
    parallel_contractions: bool = True


# Enhanced data structures
@dataclass
class BondTruncation:
    """Truncation settings for a layer"""
    enabled: bool = False
    max_bond: int = 0
    method: str = "slice"  # "slice" or "svd"


@dataclass 
class SMPOLayer:
    """Complete SMPO layer specification"""
    name: str
    input_sites: int
    output_sites: int
    phys_in: int
    phys_out: int
    bond_dim: int
    weight_name: str
    truncation: BondTruncation = field(default_factory=BondTruncation)
    optimize_non_output: bool = True
    # Spacing is calculated, not stored
    _spacing: Optional[int] = None
    
    @property
    def spacing(self) -> int:
        """Calculate spacing from input/output sites"""
        if self._spacing is None:
            self._spacing = self._calculate_spacing()
        return self._spacing
    
    def _calculate_spacing(self) -> int:
        """
        Calculate SMPO spacing - determines which sites have output indices.
        
        In SMPO, output indices appear at positions: 0, spacing, 2*spacing, ...
        First site (position 0) always has an output index.
        """
        if self.output_sites == 1:
            # Only site 0 has output - this means no other site has output
            # So we could use any large spacing, but use input_sites for clarity
            return self.input_sites
        
        if self.output_sites == self.input_sites:
            # Every site has output
            return 1
        
        # For M outputs across N sites:
        # Outputs at: 0, s, 2s, 3s, ..., (M-1)s
        # We need (M-1)*s < N to fit all outputs
        # So s <= (N-1)/(M-1)
        
        max_spacing = (self.input_sites - 1) // (self.output_sites - 1)
        
        # Try spacings from max down to find one that gives exactly M outputs
        for s in range(max_spacing, 0, -1):
            # Count outputs: positions 0, s, 2s, ... while < input_sites
            count = 0
            pos = 0
            while pos < self.input_sites:
                count += 1
                pos += s
            
            if count == self.output_sites:
                return s
        
        # If no exact match found, there's a configuration error
        raise ValueError(
            f"Cannot find valid spacing for {self.input_sites}→{self.output_sites}. "
            f"This input/output ratio may not be achievable with uniform spacing."
        )
    
    def validate_spacing(self) -> bool:
        """Validate that the calculated spacing produces correct output count"""
        # Count positions with outputs: 0, spacing, 2*spacing, ...
        count = 0
        pos = 0
        output_positions = []
        while pos < self.input_sites:
            count += 1
            output_positions.append(pos)
            pos += self.spacing
        
        if count != self.output_sites:
            raise ValueError(
                f"Layer {self.name}: Spacing {self.spacing} produces {count} outputs at positions {output_positions}, "
                f"expected {self.output_sites}. Input/output ratio may be invalid."
            )
        
        # Also validate edge case handling
        remainder = self.input_sites - output_positions[-1]
        if remainder > self.spacing:
            print(f"Warning: Layer {self.name} has {remainder} sites after last output position. "
                  f"These will be contracted with the last output group.")
        
        return True


@dataclass
class OutputConfig:
    """Output configuration"""
    format: str = "direct"  # "direct" or "streaming"
    compute_norm: bool = True  # Whether to compute MPS norm as score
    final_layer: Optional[str] = None
    
    
@dataclass
class ModelConfig:
    """Complete model configuration"""
    name: str
    input_sites: int
    input_features: int
    embedding_type: str
    data_type: str
    layers: List[SMPOLayer]
    output_config: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    def validate(self):
        """Validate configuration consistency"""
        current_sites = self.input_sites
        current_phys = self.input_features
        
        for i, layer in enumerate(self.layers):
            # Check input consistency
            if layer.input_sites != current_sites:
                raise ValueError(f"Layer {i} input mismatch: expected {current_sites}, got {layer.input_sites}")
            
            if i > 0 and layer.phys_in != self.layers[i-1].phys_out:
                raise ValueError(f"Layer {i} phys_in mismatch with previous layer phys_out")
            
            # Validate spacing calculation
            layer.validate_spacing()
            
            # Print calculated spacing for user verification
            print(f"Layer {layer.name}: Calculated spacing = {layer.spacing} "
                  f"({layer.input_sites} → {layer.output_sites})")
            
            current_sites = layer.output_sites
            current_phys = layer.phys_out
        
        return True
    
    def get_composite_bonds(self) -> List[Dict[str, int]]:
        """Calculate composite bond dimensions through the cascade"""
        bonds = []
        input_bond = 1  # First layer has trivial input bond
        
        for layer in self.layers:
            composite_bond = {
                'layer': layer.name,
                'input_bond': input_bond,
                'smpo_bond': layer.bond_dim,
                'composite_left': input_bond * layer.bond_dim,
                'composite_right': input_bond * layer.bond_dim
            }
            
            # Apply truncation if enabled
            if layer.truncation.enabled:
                composite_bond['truncated_to'] = layer.truncation.max_bond
                input_bond = min(layer.truncation.max_bond, composite_bond['composite_right'])
            else:
                input_bond = composite_bond['composite_right']
            
            bonds.append(composite_bond)
        
        return bonds


def load_model_config(config_file: str) -> ModelConfig:
    """Parse model configuration from YAML file"""
    
    # Determine file type
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            # Try to auto-detect
            content = f.read()
            f.seek(0)
            config_data = yaml.safe_load(content)
    
    # Parse hardware config if present
    hardware = HardwareConfig()
    if 'hardware' in config_data:
        hw_data = config_data['hardware']
        hardware = HardwareConfig(
            pipeline_ii=hw_data.get('pipeline_ii', 1),
            array_partition_dims=hw_data.get('array_partition_dims', [2]),
            resource_sharing=hw_data.get('resource_sharing', False),
            parallel_contractions=hw_data.get('parallel_contractions', True)
        )
    
    # Parse output config
    output_config = OutputConfig()
    if 'output' in config_data:
        out_data = config_data['output']
        output_config = OutputConfig(
            format=out_data.get('format', 'direct'),
            compute_norm=out_data.get('compute_norm', True),
            final_layer=out_data.get('final_layer', None)
        )
    
    # Parse layers
    layers = []
    for layer_data in config_data['layers']:
        # Parse truncation if present
        truncation = BondTruncation()
        if 'truncation' in layer_data:
            trunc_data = layer_data['truncation']
            truncation = BondTruncation(
                enabled=trunc_data.get('enabled', False),
                max_bond=trunc_data.get('max_bond', 0),
                method=trunc_data.get('method', 'slice')
            )
        
        layer = SMPOLayer(
            name=layer_data['name'],
            input_sites=layer_data['input_sites'],
            output_sites=layer_data['output_sites'],
            phys_in=layer_data['phys_in'],
            phys_out=layer_data['phys_out'],
            bond_dim=layer_data['bond_dim'],
            weight_name=layer_data['weight_name'],
            truncation=truncation,
            optimize_non_output=layer_data.get('optimize_non_output', True)
        )
        layers.append(layer)
    
    # Create model config
    model = ModelConfig(
        name=config_data['name'],
        input_sites=config_data['input']['sites'],
        input_features=config_data['input']['features'],
        embedding_type=config_data['input']['embedding'],
        data_type=config_data.get('data_type', 'float'),
        layers=layers,
        output_config=output_config,
        hardware=hardware
    )
    
    # Validate
    model.validate()
    
    return model


def create_example_config(output_file: str = "model_config.yml"):
    """Create an example configuration file matching the 56→19→7→3 architecture"""
    
    config = {
        "name": "cascaded_tn_autoencoder",
        "description": "56→19→7→3 autoencoder with automatic spacing",
        "data_type": "float",
        
        "input": {
            "sites": 56,
            "features": 2,
            "embedding": "trigonometric"
        },
        
        "hardware": {
            "pipeline_ii": 1,
            "array_partition_dims": [2],
            "resource_sharing": False,
            "parallel_contractions": True
        },
        
        "layers": [
            {
                "name": "layer1",
                "input_sites": 56,
                "output_sites": 19,  # spacing=3 will be calculated
                "phys_in": 2,
                "phys_out": 2,
                "bond_dim": 8,
                "weight_name": "weights1",
                "optimize_non_output": True,
                "truncation": {
                    "enabled": False
                }
            },
            {
                "name": "layer2", 
                "input_sites": 19,
                "output_sites": 7,   # spacing≈3 will be calculated
                "phys_in": 2,
                "phys_out": 2,
                "bond_dim": 6,
                "weight_name": "weights2",
                "optimize_non_output": True,
                "truncation": {
                    "enabled": False
                }
            },
            {
                "name": "layer3",
                "input_sites": 7,
                "output_sites": 3,   # spacing≈2 will be calculated
                "phys_in": 2,
                "phys_out": 2,
                "bond_dim": 4,
                "weight_name": "weights3",
                "optimize_non_output": True,
                "truncation": {
                    "enabled": False
                }
            }
        ],
        
        "output": {
            "format": "direct",
            "compute_norm": True,  # Compute MPS norm as score
            "final_layer": "layer3"
        }
    }

    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created example config file: {output_file}")


def print_model_summary(config: ModelConfig):
    """Print a summary of the model configuration"""
    print(f"\nModel: {config.name}")
    print(f"Architecture: {config.input_sites}", end="")
    
    for layer in config.layers:
        print(f" → {layer.output_sites}", end="")
        if layer.truncation.enabled:
            print(f"(↓{layer.truncation.max_bond})", end="")
    
    print(f"\nEmbedding: {config.embedding_type}")
    print(f"Data type: {config.data_type}")
    
    print("\nLayer details:")
    for i, layer in enumerate(config.layers):
        print(f"  {layer.name}: "
              f"sites {layer.input_sites}→{layer.output_sites}, "
              f"spacing={layer.spacing} (calculated), "
              f"bond={layer.bond_dim}")
        if layer.truncation.enabled:
            print(f"    └─ Truncate to bond={layer.truncation.max_bond} ({layer.truncation.method})")
    
    print("\nComposite bond dimensions:")
    bonds = config.get_composite_bonds()
    for bond_info in bonds:
        print(f"  {bond_info['layer']}: "
              f"input_bond={bond_info['input_bond']} × "
              f"smpo_bond={bond_info['smpo_bond']} = "
              f"{bond_info['composite_left']}")
        if 'truncated_to' in bond_info:
            print(f"    └─ Truncated to {bond_info['truncated_to']}")
    
    print(f"\nHardware hints:")
    print(f"  Pipeline II: {config.hardware.pipeline_ii}")
    print(f"  Array partitioning: dims={config.hardware.array_partition_dims}")
    print(f"  Resource sharing: {config.hardware.resource_sharing}")
    print(f"  Parallel contractions: {config.hardware.parallel_contractions}")


# Example usage
if __name__ == "__main__":
    # Create example config
    create_example_config("model_config.yml")
    
    # Parse and display
    config = load_model_config("model_config.yml")
    print_model_summary(config)