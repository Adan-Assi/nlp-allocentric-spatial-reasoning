"""
Task 3: Instruction Under-specification Generator (BASELINE)
Systematic degradation of RVS navigation instructions

This is a BASELINE REGEX-BASED implementation for Week 1 feasibility.
Known limitations: may miss complex entities, produce ungrammatical outputs.

Status: Week 1 - Proof of concept for presentation
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Variant:
    """One underspecified variant of an instruction."""
    instruction_id: str
    region: str
    original_instruction: str
    strategy_applied: str
    underspecified_instruction: str
    removed_information: str
    expected_impact: str
    ground_truth_lat: float
    ground_truth_lon: float


class UnderspecificationGenerator:
    """
    BASELINE generator for underspecified RVS instruction variants.
    
    Implements three strategies targeting RVS linguistic phenomena:
    1. Spatial Relation Removal (targets 96% with cardinals)
    2. Landmark Omission (targets multi-relation instructions)
    3. Distance/Precision Degradation (targets 28% with counts)
    
    LIMITATIONS (Week 1 baseline):
    - Regex-based: may miss multi-word landmarks
    - May produce ungrammatical outputs
    - Simple pattern matching only
    
    For production: would need NLP parser (spaCy/stanza)
    """
    
    def __init__(self, rvs_data_path: str = None):
        """
        Initialize with RVS dataset.
        
        Args:
            rvs_data_path: Path to RVS JSON/CSV, or None for sample data
        """
        self.data = self.load_rvs_data(rvs_data_path)
        
        # Regex patterns for identifying spatial elements
        self.cardinal_directions = r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b'
        self.ordinal_directions = r'\b(northern|southern|eastern|western)\b'
        self.distance_patterns = r'\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(\d+)?\s*(blocks?|meters?|feet|km|miles?|streets?)\b'
        self.relative_terms = r'\b(closer to|near|next to|between|past|across from|diagonal|around)\b'
        
    def load_rvs_data(self, path: str = None) -> pd.DataFrame:
        """
        Load RVS dataset.
        
        Args:
            path: Path to data file, or None for sample data
            
        Returns:
            DataFrame with: id, region, instruction, target_lat, target_lon
        """
        if path is None:
            print("📝 Using sample RVS-like data for testing")
            return self._create_sample_data()
        
        # Load from file
        if path.endswith('.json'):
            return pd.read_json(path)
        elif path.endswith('.csv'):
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path}")
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample RVS-like data (based on paper examples)."""
        samples = [
            {
                'id': 'manhattan_001',
                'region': 'Manhattan',
                'instruction': '2 blocks north of Central Park and 1 block east of the museum',
                'target_lat': 40.7678,
                'target_lon': -73.9712
            },
            {
                'id': 'manhattan_002',
                'region': 'Manhattan',
                'instruction': 'East of the library, past the fountain, near the Y-shaped intersection',
                'target_lat': 40.7489,
                'target_lon': -73.9680
            },
            {
                'id': 'brooklyn_001',
                'region': 'Brooklyn',
                'instruction': 'Exactly 3 blocks south on Atlantic Avenue',
                'target_lat': 40.6782,
                'target_lon': -73.9442
            },
            {
                'id': 'manhattan_003',
                'region': 'Manhattan',
                'instruction': 'Between the post office and the park, closer to the park',
                'target_lat': 40.7580,
                'target_lon': -73.9855
            },
            {
                'id': 'queens_001',
                'region': 'Queens',
                'instruction': 'At the corner of Queens Blvd and 42nd St, north of the subway station',
                'target_lat': 40.7433,
                'target_lon': -73.9196
            },
            {
                'id': 'manhattan_004',
                'region': 'Manhattan',
                'instruction': '3-4 blocks from Columbus Circle, north of a police station',
                'target_lat': 40.7712,
                'target_lon': -73.9820
            },
            {
                'id': 'manhattan_005',
                'region': 'Manhattan',
                'instruction': 'Walk north on 8th Ave, at a parking entrance a block north of a police station',
                'target_lat': 40.7645,
                'target_lon': -73.9890
            },
        ]
        return pd.DataFrame(samples)
    
    # ========== STRATEGY 1: SPATIAL RELATION REMOVAL ==========
    
    def strategy_1_remove_spatial_relations(self, instruction: str) -> Dict[str, str]:
        """
        Remove cardinal directions, distances, and relative terms.
        
        Targets: 96% of RVS instructions (cardinal directions)
                 88% of RVS instructions (allocentric relations)
        
        Args:
            instruction: Original instruction text
            
        Returns:
            Dict with 'underspecified' and 'removed' keys
        """
        removed = []
        modified = instruction
        
        # Remove cardinal directions (north, south, east, west, etc.)
        cardinal_matches = re.findall(self.cardinal_directions, modified, re.IGNORECASE)
        if cardinal_matches:
            removed.append(f"Cardinal directions: {', '.join(set(cardinal_matches))}")
            # Replace with generic 'from' or remove entirely
            modified = re.sub(self.cardinal_directions + r'\s+of\b', 'from', modified, flags=re.IGNORECASE)
            modified = re.sub(r'\b' + self.cardinal_directions, '', modified, flags=re.IGNORECASE)
        
        # Remove ordinal directions (northern, southern, etc.)
        ordinal_matches = re.findall(self.ordinal_directions, modified, re.IGNORECASE)
        if ordinal_matches:
            removed.append(f"Ordinal directions: {', '.join(set(ordinal_matches))}")
            modified = re.sub(self.ordinal_directions, '', modified, flags=re.IGNORECASE)
        
        # Remove distance measurements
        distance_matches = re.findall(self.distance_patterns, modified, re.IGNORECASE)
        if distance_matches:
            distance_strs = []
            for a, b, unit in distance_matches:
                if b:
                    distance_strs.append(f"{a}-{b} {unit}")
                else:
                    distance_strs.append(f"{a} {unit}")
            removed.append(f"Distances: {', '.join(distance_strs)}")
            # Replace with vague term
            modified = re.sub(self.distance_patterns, 'some distance', modified, flags=re.IGNORECASE)
        
        # Simplify relative position terms (keep minimal info)
        relative_matches = re.findall(self.relative_terms, modified, re.IGNORECASE)
        if relative_matches:
            removed.append(f"Relative terms: {', '.join(set(relative_matches))}")
            # Replace with 'near'
            modified = re.sub(self.relative_terms, 'near', modified, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        modified = re.sub(r'\s+', ' ', modified).strip()
        modified = re.sub(r'\s+,', ',', modified)
        modified = re.sub(r',\s*,', ',', modified)
        modified = re.sub(r'^\s*,\s*', '', modified)
        
        return {
            'underspecified': modified,
            'removed': '; '.join(removed) if removed else 'None'
        }
    
    # ========== STRATEGY 2: LANDMARK OMISSION ==========
    
    def strategy_2_remove_landmarks(self, instruction: str) -> Dict[str, str]:
        """
        Remove intermediate or secondary landmarks.
        
        Targets: Instructions with multiple spatial relations (avg 3-4 per instruction)
        
        LIMITATION: Simple regex - may miss complex multi-word landmarks.
        
        Args:
            instruction: Original instruction text
            
        Returns:
            Dict with 'underspecified' and 'removed' keys
        """
        removed = []
        modified = instruction
        
        # Remove "past the X" constructs (intermediate landmarks)
        past_pattern = r',?\s*past\s+the\s+[\w\s-]+?(?=,|and|$)'
        past_matches = re.findall(past_pattern, modified, re.IGNORECASE)
        if past_matches:
            removed.append(f"Intermediate 'past' landmarks: {len(past_matches)} removed")
            modified = re.sub(past_pattern, '', modified, flags=re.IGNORECASE)
        
        # Remove "near the X" as secondary constraint (keep if it's primary)
        # Only remove if there's another landmark reference
        if modified.count(',') > 0 or ' and ' in modified.lower():
            near_pattern = r',\s*near\s+the\s+[\w\s-]+?(?=,|and|$)'
            near_matches = re.findall(near_pattern, modified, re.IGNORECASE)
            if near_matches:
                removed.append(f"Secondary 'near' landmarks: {len(near_matches)} removed")
                modified = re.sub(near_pattern, '', modified, flags=re.IGNORECASE)
        
        # Remove street intersections (e.g., "at the corner of X and Y")
        intersection_pattern = r',?\s*at\s+the\s+corner\s+of\s+[^,]+?(?=,|and\s+(?:north|south|east|west)|$)'
        intersection_matches = re.findall(intersection_pattern, modified, re.IGNORECASE)
        if intersection_matches:
            removed.append(f"Street intersections: {len(intersection_matches)} removed")
            modified = re.sub(intersection_pattern, '', modified, flags=re.IGNORECASE)
        
        # Clean up
        modified = re.sub(r'\s+', ' ', modified).strip()
        modified = re.sub(r'\s+,', ',', modified)
        modified = re.sub(r',\s*,', ',', modified)
        modified = re.sub(r'^\s*,\s*', '', modified)
        modified = re.sub(r',\s*$', '', modified)
        
        return {
            'underspecified': modified,
            'removed': '; '.join(removed) if removed else 'None'
        }
    
    # ========== STRATEGY 3: PRECISION DEGRADATION ==========
    
    def strategy_3_degrade_precision(self, instruction: str) -> Dict[str, str]:
        """
        Replace exact measurements with vague quantifiers.
        
        Targets: 28% of RVS instructions (contain counts)
        
        Args:
            instruction: Original instruction text
            
        Returns:
            Dict with 'underspecified' and 'removed' keys
        """
        removed = []
        modified = instruction
        
        # Replace exact block/street counts
        block_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(\d+)?\s*(blocks?|streets?)\b'
        block_matches = re.findall(block_pattern, modified, re.IGNORECASE)
        if block_matches:
            exact_counts = [m[0] + (f'-{m[1]}' if m[1] else '') for m in block_matches]
            removed.append(f"Exact block counts: {', '.join(exact_counts)}")
            modified = re.sub(block_pattern, r'some \3', modified, flags=re.IGNORECASE)
        
        # Replace exact distances (meters, feet, km)
        metric_pattern = r'\b\d+(?:\.\d+)?\s*(meters?|feet|km|miles?)\b'
        metric_matches = re.findall(metric_pattern, modified, re.IGNORECASE)
        if metric_matches:
            removed.append(f"Exact distances: {len(metric_matches)} removed")
            modified = re.sub(metric_pattern, 'some distance', modified, flags=re.IGNORECASE)
        
        # Replace "exactly" precision marker
        if re.search(r'\bexactly\b', modified, re.IGNORECASE):
            removed.append("Precision marker: 'exactly'")
            modified = re.sub(r'\bexactly\s+', '', modified, flags=re.IGNORECASE)
        
        # Replace specific ordinals (first, second, third)
        ordinal_pattern = r'\b(first|second|third|fourth|fifth)\b'
        ordinal_matches = re.findall(ordinal_pattern, modified, re.IGNORECASE)
        if ordinal_matches:
            removed.append(f"Ordinal positions: {', '.join(ordinal_matches)}")
            modified = re.sub(ordinal_pattern, 'a', modified, flags=re.IGNORECASE)
        
        # Clean up
        modified = re.sub(r'\s+', ' ', modified).strip()
        
        return {
            'underspecified': modified,
            'removed': '; '.join(removed) if removed else 'None'
        }
    
    # ========== MAIN GENERATION ==========
    
    def apply_all_strategies(self, row: pd.Series) -> List[Variant]:
        """
        Apply all applicable strategies to one instruction.
        
        Args:
            row: DataFrame row with instruction data
            
        Returns:
            List of Variant objects (one per applicable strategy)
        """
        variants = []
        instruction = row['instruction']
        
        # Strategy 1: Spatial Relation Removal
        s1_result = self.strategy_1_remove_spatial_relations(instruction)
        if s1_result['removed'] != 'None':
            variants.append(Variant(
                instruction_id=row['id'],
                region=row['region'],
                original_instruction=instruction,
                strategy_applied='Strategy 1: Spatial Relation Removal',
                underspecified_instruction=s1_result['underspecified'],
                removed_information=s1_result['removed'],
                expected_impact='May fail to infer direction/distance; could guess randomly',
                ground_truth_lat=row['target_lat'],
                ground_truth_lon=row['target_lon']
            ))
        
        # Strategy 2: Landmark Omission
        s2_result = self.strategy_2_remove_landmarks(instruction)
        if s2_result['removed'] != 'None':
            variants.append(Variant(
                instruction_id=row['id'],
                region=row['region'],
                original_instruction=instruction,
                strategy_applied='Strategy 2: Landmark Omission',
                underspecified_instruction=s2_result['underspecified'],
                removed_information=s2_result['removed'],
                expected_impact='May lose precision; hallucinate missing landmarks',
                ground_truth_lat=row['target_lat'],
                ground_truth_lon=row['target_lon']
            ))
        
        # Strategy 3: Distance/Precision Degradation
        s3_result = self.strategy_3_degrade_precision(instruction)
        if s3_result['removed'] != 'None':
            variants.append(Variant(
                instruction_id=row['id'],
                region=row['region'],
                original_instruction=instruction,
                strategy_applied='Strategy 3: Distance/Precision Degradation',
                underspecified_instruction=s3_result['underspecified'],
                removed_information=s3_result['removed'],
                expected_impact='May over/underestimate distance; overconfident guessing',
                ground_truth_lat=row['target_lat'],
                ground_truth_lon=row['target_lon']
            ))
        
        return variants
    
    def generate_dataset(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate underspecified dataset.
        
        Args:
            n_samples: Number of original instructions to process
            
        Returns:
            DataFrame with all variants
        """
        # Sample instructions
        if len(self.data) > n_samples:
            sampled = self.data.sample(n=n_samples, random_state=42)
        else:
            sampled = self.data
            print(f"ℹ️  Using all {len(sampled)} available instructions")
        
        # Generate variants
        all_variants = []
        for idx, row in sampled.iterrows():
            variants = self.apply_all_strategies(row)
            all_variants.extend(variants)
        
        # Convert to DataFrame
        variant_dicts = [asdict(v) for v in all_variants]
        df = pd.DataFrame(variant_dicts)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save dataset to CSV."""
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} variants to {output_path}")
    
    def print_quality_report(self, df: pd.DataFrame):
        """Print quality report for manual review."""
        print("\n" + "="*70)
        print("QUALITY REPORT - BASELINE UNDERSPECIFICATION GENERATOR")
        print("="*70)
        
        print(f"\n📊 Generation Statistics:")
        print(f"   Total variants: {len(df)}")
        print(f"   Unique instructions: {df['instruction_id'].nunique()}")
        print(f"\n   Strategy breakdown:")
        for strategy, count in df['strategy_applied'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"   - {strategy}: {count} ({pct:.1f}%)")
        
        print(f"\n⚠️  KNOWN LIMITATIONS (Regex-based baseline):")
        print(f"   - May miss multi-word landmarks (e.g., 'St. Vincent de Paul Church')")
        print(f"   - May produce ungrammatical outputs in complex cases")
        print(f"   - Simple pattern matching only (no semantic understanding)")
        
        print(f"\n✅ RECOMMENDED NEXT STEPS:")
        print(f"   1. Manually review 10-20 random samples")
        print(f"   2. Check for grammatical issues")
        print(f"   3. Verify removed information is accurate")
        print(f"   4. For production: upgrade to spaCy/stanza-based parser")


# ========== EXAMPLE USAGE ==========

if __name__ == '__main__':
    print("="*70)
    print("TASK 3: BASELINE UNDERSPECIFICATION GENERATOR")
    print("="*70)
    
    # Initialize
    generator = UnderspecificationGenerator()
    
    # Generate dataset
    print("\n🔄 Generating underspecified variants...")
    variants_df = generator.generate_dataset(n_samples=100)
    
    # Quality report
    generator.print_quality_report(variants_df)
    
    # Show examples
    print("\n" + "="*70)
    print("SAMPLE VARIANTS (First 3)")
    print("="*70)
    for idx, row in variants_df.head(3).iterrows():
        print(f"\n{idx+1}. {row['instruction_id']} - {row['strategy_applied']}")
        print(f"   Original: {row['original_instruction']}")
        print(f"   Modified: {row['underspecified_instruction']}")
        print(f"   Removed:  {row['removed_information']}")
    
    # Save
    output_file = 'underspecified_instructions.csv'
    generator.save_dataset(variants_df, output_file)
    
    print("\n" + "="*70)
    print(f"✅ TASK 3 COMPLETE - Baseline generator ready for Week 1")
    print(f"   Output: {output_file}")
    print(f"   Status: Proof-of-concept for feasibility demonstration")
    print("="*70)
