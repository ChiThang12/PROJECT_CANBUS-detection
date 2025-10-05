"""
Decision Tree Simulator
Simulates hardware decision tree inference using tree.mem and features.mem
"""

class DecisionTreeSimulator:
    def __init__(self, tree_file='tree.mem', features_file='features.mem'):
        self.tree_file = tree_file
        self.features_file = features_file
        self.tree = []
        self.features = []
        
    def load_tree(self):
        """Load tree.mem (95-bit binary format)"""
        print("="*80)
        print("Loading Decision Tree from:", self.tree_file)
        print("="*80)
        
        with open(self.tree_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            binary = line.strip()
            if len(binary) != 95:
                print(f"WARNING: Invalid line length {len(binary)}, expected 95 bits")
                continue
            
            # Parse 95-bit format
            node = {
                'node_id':     int(binary[0:9], 2),      # [94:86]
                'feature_idx': int(binary[9:11], 2),     # [85:84]
                'threshold':   int(binary[11:75], 2),    # [83:20]
                'left_child':  int(binary[75:84], 2),    # [19:11]
                'right_child': int(binary[84:93], 2),    # [10:2]
                'prediction':  int(binary[93:95], 2),    # [1:0]
            }
            
            # Check if leaf node
            node['is_leaf'] = (node['left_child'] == 0 and node['right_child'] == 0)
            
            self.tree.append(node)
        
        print(f"✓ Loaded {len(self.tree)} nodes")
        
        # Display first few nodes
        print("\nFirst 5 nodes:")
        for i in range(min(5, len(self.tree))):
            node = self.tree[i]
            print(f"  Node {node['node_id']:3d}: ", end="")
            if node['is_leaf']:
                print(f"LEAF prediction={node['prediction']}")
            else:
                feat_map = {0: '00(Timestamp)', 1: '01(ArbID)', 2: '10(Data)', 3: 'LEAF'}
                print(f"Feature[{feat_map.get(node['feature_idx'], '??')}] <= {node['threshold']} "
                      f"? L={node['left_child']} : R={node['right_child']}")
        
        return len(self.tree)
    
    def load_features(self):
        """Load features.mem (64-bit Q32.32 format)"""
        print("\n" + "="*80)
        print("Loading Features from:", self.features_file)
        print("="*80)
        
        with open(self.features_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise ValueError(f"Expected 3 features, got {len(lines)}")
        
        for i, line in enumerate(lines[:3]):
            binary = line.strip()
            if len(binary) != 64:
                print(f"WARNING: Feature {i} has length {len(binary)}, expected 64 bits")
                continue
            
            # Parse Q32.32 format
            value_int = int(binary, 2)
            int_part = (value_int >> 32) & 0xFFFFFFFF
            frac_part = value_int & 0xFFFFFFFF
            float_value = int_part + (frac_part / (2**32))
            
            self.features.append({
                'binary': binary,
                'int_value': value_int,
                'int_part': int_part,
                'frac_part': frac_part,
                'float_value': float_value
            })
        
        print(f"✓ Loaded {len(self.features)} features")
        print("\nFeature values:")
        feature_names = ['00 (Timestamp)', '01 (Arb ID)', '10 (Data)']
        for i, feat in enumerate(self.features):
            print(f"  Feature {feature_names[i]}:")
            print(f"    Binary:  {feat['binary']}")
            print(f"    Int:     {feat['int_part']} (0x{feat['int_part']:08X})")
            print(f"    Frac:    {feat['frac_part']} (0x{feat['frac_part']:08X})")
            print(f"    Float:   {feat['float_value']:.6f}")
            print(f"    Q32.32:  0x{feat['int_value']:016X}")
        
        return len(self.features)
    
    def compare_q32(self, feature_value, threshold_value):
        """
        Compare Q32.32 values: feature <= threshold
        Simulates q32_comparator.v logic
        """
        # Extract integer and fractional parts
        feat_int = (feature_value >> 32) & 0xFFFFFFFF
        feat_frac = feature_value & 0xFFFFFFFF
        thresh_int = (threshold_value >> 32) & 0xFFFFFFFF
        thresh_frac = threshold_value & 0xFFFFFFFF
        
        # Compare (signed integer, unsigned fractional)
        # Convert to signed 32-bit
        if feat_int & 0x80000000:
            feat_int_signed = feat_int - 0x100000000
        else:
            feat_int_signed = feat_int
            
        if thresh_int & 0x80000000:
            thresh_int_signed = thresh_int - 0x100000000
        else:
            thresh_int_signed = thresh_int
        
        # Compare logic
        if feat_int_signed < thresh_int_signed:
            return True  # Go left
        elif feat_int_signed == thresh_int_signed:
            return feat_frac <= thresh_frac  # Compare fractional
        else:
            return False  # Go right
    
    def inference(self, verbose=True):
        """
        Run decision tree inference
        Simulates decision_tree_engine.v
        """
        print("\n" + "="*80)
        print("STARTING INFERENCE")
        print("="*80)
        
        current_node_id = 0
        depth = 0
        max_depth = 20
        
        while depth < max_depth:
            # Get current node
            if current_node_id >= len(self.tree):
                print(f"\nERROR: Node {current_node_id} out of range!")
                return None
            
            node = self.tree[current_node_id]
            
            if verbose:
                print(f"\nDepth {depth}: Node {node['node_id']}")
            
            # Check if leaf
            if node['is_leaf']:
                prediction = node['prediction']
                is_attack = (prediction == 0b01)
                
                print("\n" + "="*80)
                print("REACHED LEAF NODE")
                print("="*80)
                print(f"  Final Node:  {node['node_id']}")
                print(f"  Prediction:  {prediction:02b} ({'ATTACK' if is_attack else 'NORMAL'})")
                print(f"  Tree Depth:  {depth}")
                print("="*80)
                
                return {
                    'prediction': prediction,
                    'is_attack': is_attack,
                    'final_node': node['node_id'],
                    'depth': depth
                }
            
            # Internal node - compare feature
            feature_idx = node['feature_idx']
            threshold = node['threshold']
            
            if feature_idx >= len(self.features):
                print(f"ERROR: Invalid feature index {feature_idx}")
                return None
            
            feature_value = self.features[feature_idx]['int_value']
            
            # Compare
            go_left = self.compare_q32(feature_value, threshold)
            
            if verbose:
                feat_names = ['Timestamp', 'Arb ID', 'Data']
                feat_float = self.features[feature_idx]['float_value']
                thresh_int = (threshold >> 32) & 0xFFFFFFFF
                thresh_frac = threshold & 0xFFFFFFFF
                thresh_float = thresh_int + (thresh_frac / (2**32))
                
                print(f"  Feature[{feature_idx}] ({feat_names[feature_idx]})")
                print(f"    Value:     {feat_float:.6f} (0x{feature_value:016X})")
                print(f"    Threshold: {thresh_float:.6f} (0x{threshold:016X})")
                print(f"    Compare:   {feat_float:.6f} <= {thresh_float:.6f} ? {go_left}")
                print(f"  Decision:  Go {'LEFT' if go_left else 'RIGHT'} to node {node['left_child'] if go_left else node['right_child']}")
            
            # Move to next node
            if go_left:
                current_node_id = node['left_child']
            else:
                current_node_id = node['right_child']
            
            depth += 1
        
        print(f"\nERROR: Max depth {max_depth} reached!")
        return None


def main():
    """Main function to run simulation"""
    print("\n" + "="*80)
    print("DECISION TREE HARDWARE SIMULATOR")
    print("Simulates RTL behavior using tree.mem and features.mem")
    print("="*80)
    
    # Create simulator
    sim = DecisionTreeSimulator('tree.mem', 'features.mem')
    
    # Load tree and features
    try:
        sim.load_tree()
        sim.load_features()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Make sure tree.mem and features.mem exist in current directory")
        return
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    
    # Run inference
    result = sim.inference(verbose=True)
    
    if result:
        print("\n" + "="*80)
        print("SIMULATION COMPLETE")
        print("="*80)
        print(f"Result: {'🚨 ATTACK DETECTED' if result['is_attack'] else '✅ NORMAL TRAFFIC'}")
        print(f"Class: {result['prediction']:02b}")
        print(f"Final Node: {result['final_node']}")
        print(f"Tree Depth: {result['depth']}")
        print("="*80)
    else:
        print("\n❌ SIMULATION FAILED")


if __name__ == "__main__":
    main()