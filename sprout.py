import argparse
import yaml
from make_seeds import run_make_seeds
from make_grow import run_make_grow
from make_adaptive_seed import run_make_adaptive_seed
from sam_predict import run_sam_yaml
from batch_sam import run_batch_sam

from batch_grow import run_batch_grow
from batch_seeds import run_batch_adaptive_seed, run_batch_seeds

import os

def load_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="SPROUT CLI: generate seeds with a YAML config",
        usage=
        """
        For seeds generation:
            python sprout.py --seeds --config path/to/config.yaml
        
        For adaptive seed generation
            python sprout.py --adaptive_seed --config path/to/config.yaml
        
        For growing
            python sprout.py --grow --config path/to/config.yaml
            
        For Foundation model prediction:
            python sprout.py --sam --config path/to/config.yaml  
            
        To run in batch mode (pipeline version), add --batch:
            python sprout.py --seeds --batch --config path/to/config.yaml

        For more info, see README.md and ./template examples.
        """
    )
    

    # Mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seeds', action='store_true', help="Run the seed generation function")
    group.add_argument('--adaptive_seed', action='store_true', help="Run the adaptive seed generation function")
    group.add_argument('--grow', action='store_true', help="Run the grow function")
    group.add_argument('--sam', action='store_true', help="Run the SAM prediction function")
    
    parser.add_argument('--batch', action='store_true', help="Run in batch mode using pipeline functions")
    parser.add_argument('--config', type=str, help="Path to the YAML config file")

    args = parser.parse_args()

    assert args.config.endswith(".yaml"), f"Error: {args.config} does not end with '.yaml'"
    assert os.path.isfile(args.config), f"Error: File not found - {args.config}"
    
    if args.seeds:
        if not args.config:
            print("[ERROR] --config is required when using --seeds")
            parser.print_help()
            exit(1)
        # config = load_config(args.config)
        if args.batch:
            run_batch_seeds(args.config)
        else:
            run_make_seeds(args.config)
    elif args.grow:
        if not args.config:
            print("[ERROR] --config is required when using --grow")
            parser.print_help()
            exit(1)
        # config = load_config(args.config)
        if args.batch:
            run_batch_grow(args.config)
        else:
            run_make_grow(args.config)
    elif args.adaptive_seed:
        
        if not args.config:
            print("[ERROR] --config is required when using --adaptive_seed")
            parser.print_help()
            exit(1)
        if args.batch:
            run_batch_adaptive_seed(args.config)
        else:
            run_make_adaptive_seed(args.config)
    elif args.sam:
        if not args.config:
            print("[ERROR] --config is required when using --sam")
            parser.print_help()
            exit(1)
        # config = load_config(args.config)
        if args.batch:
            run_batch_sam(args.config)
        else:
            run_sam_yaml(args.config)

    else:
        print("[ERROR] No valid action specified.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
