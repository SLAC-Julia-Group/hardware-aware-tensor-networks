"""
Smart dimension calculator for cascaded tensor networks.

This module automatically calculates optimal spacing and bond dimensions
for tensor network layers to achieve desired compression ratios.
"""

from typing import List, Union, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from ..core.base import LayerConfig


@dataclass
class SpacingResult:
    """Result of spacing calculation for a layer."""
    spacing: Union[int, List[int]]
    actual_outputs: int
    target_outputs: int
    efficiency: float  # How close we got to target
    method: str  # Method used for calculation
    
    def __repr__(self):
        eff_str = f"{self.efficiency*100:.1f}%"
        if isinstance(self.spacing, int):
            return f"SpacingResult(spacing={self.spacing}, outputs={self.actual_outputs}/{self.target_outputs}, eff={eff_str}, method='{self.method}')"
        else:
            return f"SpacingResult(spacing=[custom], outputs={self.actual_outputs}/{self.target_outputs}, eff={eff_str}, method='{self.method}')"


class DimensionCalculator:
    """
    Calculates optimal dimensions for cascaded tensor networks.
    
    Handles:
    - Automatic spacing calculation (uniform and non-uniform)
    - Bond dimension optimization
    - Cyclic boundary condition adjustments
    - Validation of achievable configurations
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def calculate_cascade_spacings(self, layer_dims: List[int], 
                                 cyclic: bool = False,
                                 allow_approximate: bool = True) -> List[SpacingResult]:
        """
        Calculate spacings for an entire cascade.
        
        Args:
            layer_dims: List of dimensions [input, hidden1, hidden2, ..., output]
            cyclic: Whether to use cyclic boundary conditions
            allow_approximate: Whether to allow approximate output counts
            
        Returns:
            List of SpacingResult objects for each layer
        """
        if len(layer_dims) < 2:
            raise ValueError("Need at least 2 dimensions (input and output)")
        
        # Check cyclic compatibility first
        if cyclic:
            incompatible = self.check_cyclic_compatibility(layer_dims)
            if incompatible:
                print("\n" + "="*60)
                print("⚠️  CYCLIC COMPATIBILITY WARNING")
                print("="*60)
                for issue in incompatible:
                    print(f"  ❌ {issue}")
                
                # Suggest compatible dimensions
                suggested = self.suggest_cyclic_compatible_dims(
                    layer_dims[0], len(layer_dims)-1, layer_dims[-1]
                )
                print(f"\n  💡 Suggested cyclic-compatible dimensions:")
                print(f"     {' → '.join(map(str, suggested))}")
                print("="*60 + "\n")
        
        results = []
        
        for i in range(len(layer_dims) - 1):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1]
            
            if self.debug:
                print(f"\n[DIM_CALC] Layer {i}: {input_dim} → {output_dim}")
            
            result = self.calculate_optimal_spacing(
                input_dim, output_dim, cyclic, allow_approximate
            )
            results.append(result)
            
            if self.debug:
                print(f"[DIM_CALC] {result}")
        
        return results
    
    def calculate_optimal_spacing(self, input_dim: int, output_dim: int,
                                cyclic: bool = False, 
                                allow_approximate: bool = True) -> SpacingResult:
        """
        Calculate optimal spacing for a single layer.
        
        Tries multiple strategies:
        1. Uniform spacing (if it divides evenly)
        2. Near-uniform spacing (if approximate allowed)
        3. Custom non-uniform spacing (for difficult cases)
        """
        # Special cases
        if output_dim >= input_dim:
            # Expansion or identity - all tensors have outputs
            return SpacingResult(
                spacing=1,
                actual_outputs=input_dim,
                target_outputs=output_dim,
                efficiency=min(1.0, output_dim / input_dim),
                method="expansion"
            )
        
        if output_dim == 1:
            # Extreme compression - only first tensor has output
            spacing = input_dim if not cyclic else input_dim
            return SpacingResult(
                spacing=spacing,
                actual_outputs=1,
                target_outputs=1,
                efficiency=1.0,
                method="single_output"
            )
        
        # Try uniform spacing first
        uniform_result = self._try_uniform_spacing(input_dim, output_dim, cyclic)
        if uniform_result.efficiency == 1.0:
            return uniform_result
        
        # If exact uniform doesn't work and we allow approximation
        if allow_approximate:
            # Try near-uniform spacing
            near_uniform_result = self._try_near_uniform_spacing(
                input_dim, output_dim, cyclic
            )
            if near_uniform_result.efficiency > 0.8:  # Good enough
                return near_uniform_result
            
            # For difficult cases, use custom spacing
            custom_result = self._calculate_custom_spacing(
                input_dim, output_dim, cyclic
            )
            
            # Return best result
            results = [uniform_result, near_uniform_result, custom_result]
            return max(results, key=lambda r: r.efficiency)
        
        return uniform_result

    def calculate_centered_positions(self, input_dim: int, output_dim: int) -> List[int]:
        """
        Calculate symmetrically centered output positions.
        
        Distributes outputs evenly across [0, input_dim-1], always including
        both endpoints (except for single output, which goes to center).
        
        Args:
            input_dim: Number of input sites (L)
            output_dim: Number of desired outputs (M)
            
        Returns:
            List of output positions, sorted ascending
            
        Examples:
            >>> calc.calculate_centered_positions(19, 1)
            [9]
            >>> calc.calculate_centered_positions(19, 2)
            [0, 18]
            >>> calc.calculate_centered_positions(19, 3)
            [0, 9, 18]
            >>> calc.calculate_centered_positions(19, 5)
            [0, 4, 9, 14, 18]
        """
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        if output_dim > input_dim:
            raise ValueError(
                f"Cannot have more outputs ({output_dim}) than inputs ({input_dim})"
            )
        
        if output_dim == input_dim:
            # Every site has an output
            return list(range(input_dim))
        
        if output_dim == 1:
            # Single output at center
            return [(input_dim - 1) // 2]
        
        # Distribute evenly from 0 to input_dim-1
        positions = []
        for i in range(output_dim):
            # Linspace-style: i * (L-1) / (M-1)
            pos = round(i * (input_dim - 1) / (output_dim - 1))
            positions.append(pos)
        
        # Ensure uniqueness (rounding could theoretically cause duplicates for edge cases)
        positions = sorted(set(positions))
        
        # Validate we got the right count (should always pass, but safety check)
        if len(positions) != output_dim:
            if self.debug:
                print(f"[WARNING] Centered positions: expected {output_dim}, got {len(positions)}")
                print(f"          Positions: {positions}")
            # Fall back to uniform spacing if centering failed
            return None
        
        return positions

    def _try_uniform_spacing(self, input_dim: int, output_dim: int, 
                           cyclic: bool) -> SpacingResult:
        """Try to find uniform spacing that gives exact output count."""
        # For uniform spacing: outputs = ceil(input_dim / spacing)
        # We want: ceil(input_dim / spacing) = output_dim
        
        # Try different spacings
        for spacing in range(input_dim, 0, -1):
            if cyclic:
                # In cyclic case, we get exactly input_dim/spacing outputs
                if input_dim % spacing == 0:
                    actual_outputs = input_dim // spacing
                else:
                    continue  # Skip non-divisors for cyclic
            else:
                # In open case, we get ceil(input_dim/spacing) outputs
                actual_outputs = len(range(0, input_dim, spacing))
            
            if actual_outputs == output_dim:
                return SpacingResult(
                    spacing=spacing,
                    actual_outputs=actual_outputs,
                    target_outputs=output_dim,
                    efficiency=1.0,
                    method="uniform"
                )
        
        # If no exact match found
        if cyclic:
            # For cyclic, find best approximation among divisors
            divisors = self._get_divisors(input_dim)
            possible_outputs = [(input_dim // d, d) for d in divisors]
            
            # Find closest output count
            best_outputs, best_spacing = min(
                possible_outputs,
                key=lambda x: abs(x[0] - output_dim)
            )
            
            # If we're far off, warn the user
            if abs(best_outputs - output_dim) > output_dim * 0.2:
                print(f"[WARNING] Cyclic: {input_dim}→{output_dim} not exactly achievable.")
                print(f"          Possible outputs: {[p[0] for p in sorted(possible_outputs, reverse=True)[:8]]}")
                print(f"          Using {best_outputs} (spacing={best_spacing})")
            
            return SpacingResult(
                spacing=best_spacing,
                actual_outputs=best_outputs,
                target_outputs=output_dim,
                efficiency=1.0 - abs(best_outputs - output_dim) / output_dim,
                method="uniform_approx"
            )
        else:
            # Non-cyclic: find closest spacing
            best_spacing = max(1, (input_dim - 1) // (output_dim - 1)) if output_dim > 1 else input_dim
            actual_outputs = len(range(0, input_dim, best_spacing))
            
            return SpacingResult(
                spacing=best_spacing,
                actual_outputs=actual_outputs,
                target_outputs=output_dim,
                efficiency=1.0 - abs(actual_outputs - output_dim) / output_dim,
                method="uniform_approx"
            )
    
    def _try_near_uniform_spacing(self, input_dim: int, output_dim: int,
                                 cyclic: bool) -> SpacingResult:
        """Try near-uniform spacing with small variations."""
        base_spacing = max(1, input_dim // output_dim)
        
        # Try base_spacing and base_spacing+1 mixed
        # This handles cases like 56→32 where uniform doesn't work perfectly
        
        if cyclic:
            # For cyclic, we need more careful handling
            # TODO: Implement mixed spacing for cyclic case
            return self._try_uniform_spacing(input_dim, output_dim, cyclic)
        
        # Calculate how many of each spacing we need
        # n1 * base_spacing + n2 * (base_spacing + 1) ≈ input_dim
        # n1 + n2 = output_dim
        
        n2 = input_dim - output_dim * base_spacing
        n1 = output_dim - n2
        
        if n1 >= 0 and n2 >= 0:
            # Create alternating pattern
            spacings = [base_spacing] * n1 + [base_spacing + 1] * n2
            # Shuffle for better distribution
            import random
            random.seed(42)  # Deterministic shuffle
            random.shuffle(spacings)
            
            # Verify actual output count
            position = 0
            outputs = 0
            for s in spacings:
                if position < input_dim:
                    outputs += 1
                    position += s
            
            return SpacingResult(
                spacing=spacings,
                actual_outputs=outputs,
                target_outputs=output_dim,
                efficiency=1.0 - abs(outputs - output_dim) / output_dim,
                method="near_uniform"
            )
        
        return self._try_uniform_spacing(input_dim, output_dim, cyclic)
    
    def _calculate_custom_spacing(self, input_dim: int, output_dim: int,
                                cyclic: bool) -> SpacingResult:
        """Calculate custom non-uniform spacing for difficult cases."""
        # Use logarithmic spacing for better information distribution
        if cyclic:
            # For cyclic, use uniform as fallback for now
            return self._try_uniform_spacing(input_dim, output_dim, cyclic)
        
        # Generate positions using inverse transform sampling
        # This gives us more outputs at the beginning and end
        positions = []
        for i in range(output_dim):
            # Map to [0, 1], apply transform, map back
            normalized = i / (output_dim - 1) if output_dim > 1 else 0.5
            # Use sqrt for moderate compression, could be tuned
            transformed = np.sqrt(normalized)
            position = int(transformed * (input_dim - 1))
            positions.append(position)
        
        # Convert positions to spacings
        spacings = []
        for i in range(1, len(positions)):
            spacings.append(positions[i] - positions[i-1])
        
        # Handle last spacing to ensure we stay within bounds
        if positions[-1] < input_dim - 1:
            spacings.append(input_dim - 1 - positions[-1])
        
        return SpacingResult(
            spacing=spacings,
            actual_outputs=len(positions),
            target_outputs=output_dim,
            efficiency=1.0,  # We always hit the target with custom
            method="custom_logarithmic"
        )
    
    def suggest_bond_dimensions(self, layer_dims: List[int],
                              base_bond_dim: Optional[int] = None,
                              min_bond_dim: int = 4,
                              max_bond_dim: int = 64) -> List[int]:
        """
        Suggest bond dimensions for each layer.
        
        Strategy:
        - Start with higher bond dim for early layers (more capacity)
        - Gradually decrease for later layers
        - Respect min/max constraints
        """
        num_layers = len(layer_dims) - 1
        
        if base_bond_dim is None:
            # Auto-calculate base from compression ratio
            total_compression = layer_dims[0] / layer_dims[-1]
            base_bond_dim = min(max_bond_dim, max(min_bond_dim, 
                                                 int(np.sqrt(layer_dims[0]))))
        
        bond_dims = []
        
        for i in range(num_layers):
            # Calculate compression at this layer
            local_compression = layer_dims[i] / layer_dims[i + 1]
            
            # Exponential decay with layer depth
            decay_factor = np.exp(-i / num_layers)
            
            # Adjust by local compression needs
            compression_factor = np.sqrt(local_compression)
            
            suggested = int(base_bond_dim * decay_factor * compression_factor)
            suggested = max(min_bond_dim, min(max_bond_dim, suggested))
            
            bond_dims.append(suggested)
            
            if self.debug:
                print(f"[DIM_CALC] Layer {i} bond dim: {suggested} "
                      f"(compression: {local_compression:.2f}x)")
        
        return bond_dims
    
    def validate_cascade_dimensions(self, configs: List[LayerConfig]) -> Tuple[bool, List[str]]:
        """
        Validate that a cascade configuration is achievable.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        for i in range(len(configs) - 1):
            curr = configs[i]
            next_cfg = configs[i + 1]
            
            # Check basic connectivity
            if curr.output_dim != next_cfg.input_dim:
                issues.append(f"Layer {i}→{i+1}: dimension mismatch "
                            f"({curr.output_dim} != {next_cfg.input_dim})")
            
            # Check cyclic compatibility
            if curr.cyclic != next_cfg.cyclic:
                issues.append(f"Layer {i}→{i+1}: cyclic mismatch")
            
            # Check bond dimension reasonableness
            if curr.bond_dim < 2:
                issues.append(f"Layer {i}: bond dim too small ({curr.bond_dim})")
        
        # Check for extreme compressions
        for i, cfg in enumerate(configs):
            compression = cfg.input_dim / cfg.output_dim if cfg.output_dim > 0 else float('inf')
            if compression > 10:
                issues.append(f"Layer {i}: extreme compression ({compression:.1f}x)")
        
        return len(issues) == 0, issues
    
    def check_cyclic_compatibility(self, layer_dims: List[int]) -> List[str]:
        """
        Check if dimensions are compatible with cyclic boundaries.
        
        For cyclic networks, each layer's output dim must be achievable
        by uniformly sampling the input dim (i.e., input_dim % spacing == 0).
        
        Returns:
            List of compatibility issues (empty if compatible)
        """
        issues = []
        
        for i in range(len(layer_dims) - 1):
            input_dim = layer_dims[i]
            output_dim = layer_dims[i + 1]
            
            if output_dim >= input_dim:
                continue  # Expansion is always possible
            
            # Find divisors of input_dim
            divisors = self._get_divisors(input_dim)
            possible_outputs = [input_dim // d for d in divisors]
            
            if output_dim not in possible_outputs:
                issues.append(
                    f"Layer {i}: {input_dim}→{output_dim} impossible with cyclic. "
                    f"Valid outputs: {sorted(possible_outputs, reverse=True)[:5]}..."
                )
        
        return issues
    
    def suggest_cyclic_compatible_dims(self, input_dim: int, num_layers: int,
                                     target_output: Optional[int] = None) -> List[int]:
        """
        Suggest dimensions that work well with cyclic boundaries.
        
        Args:
            input_dim: Starting dimension
            num_layers: Number of layers (excluding input)
            target_output: Desired final dimension (optional)
            
        Returns:
            List of dimensions [input, hidden1, ..., output]
        """
        divisors = self._get_divisors(input_dim)
        
        if target_output is not None:
            # Check if target is achievable
            if target_output > input_dim:
                print(f"[WARNING] Target {target_output} > input {input_dim}, using {input_dim}")
                target_output = input_dim
            elif input_dim % target_output != 0 and target_output not in divisors:
                # Find closest divisor
                closest = min(divisors, key=lambda d: abs(input_dim//d - target_output))
                actual_output = input_dim // closest
                print(f"[WARNING] Target {target_output} not achievable, using {actual_output}")
                target_output = actual_output
        
        # Strategy: Use geometric progression through divisors
        if num_layers == 1:
            return [input_dim, target_output or 1]
        
        # Find a nice geometric sequence
        dims = [input_dim]
        
        if target_output:
            # Work backwards from target
            ratio = (target_output / input_dim) ** (1 / num_layers)
            
            for i in range(1, num_layers):
                ideal = input_dim * (ratio ** i)
                # Find closest achievable dimension
                best_dim = min(
                    [input_dim // d for d in divisors],
                    key=lambda x: abs(x - ideal) if x > 0 else float('inf')
                )
                dims.append(best_dim)
            
            dims.append(target_output)
        else:
            # No target - use nice progression
            # Try to reduce by factors of 2, 3, 4, etc.
            current = input_dim
            for i in range(num_layers):
                # Find a good divisor
                for factor in [2, 3, 4, 5, 7, 8]:
                    if current % factor == 0 and current // factor > 0:
                        current = current // factor
                        break
                else:
                    # No good factor found, use largest divisor > 1
                    valid_divisors = [d for d in self._get_divisors(current) if d > 1]
                    if valid_divisors:
                        current = current // min(valid_divisors)
                    else:
                        current = 1
                dims.append(current)
        
        # Ensure strictly decreasing (for compression)
        for i in range(1, len(dims)):
            if dims[i] >= dims[i-1]:
                dims[i] = dims[i-1] // 2 if dims[i-1] > 1 else 1
        
        return dims
    
    def _get_divisors(self, n: int) -> List[int]:
        """Get all divisors of n in ascending order."""
        divisors = []
        for i in range(1, int(np.sqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)