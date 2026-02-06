import argparse
import yaml
from make_seeds import run_make_seeds
from make_grow import run_make_grow
from make_adaptive_seed import run_make_adaptive_seed


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
            
        For Predictions using PROMPT extracted from seeds for SAM
            python sprout.py --prompt --sam --config path/to/config.yaml  
        
        For Predictions using PROMPT extracted from seeds for nnInteractive
            python sprout.py --prompt --nninteractive --config path/to/config.yaml
            
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
    group.add_argument('--prompt', action='store_true', help="Run the PROMPT extraction function")
    
    # a group args only when --prompt is specified
    # sam or nninteractive
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--sam', action='store_true', help="Use SAM for prompt extraction")
    prompt_group.add_argument('--nninteractive', action='store_true', help="Use nnInteractive for prompt extraction")
    
    parser.add_argument('--batch', action='store_true', help="Run in batch mode using pipeline functions")
    parser.add_argument('--config', type=str, help="Path to the YAML config file")

    args = parser.parse_args()

    assert args.config.endswith(".yaml"), f"Error: {args.config} does not end with '.yaml'"
    assert os.path.isfile(args.config), f"Error: File not found - {args.config}"
    
    # check sam or interactive only when --prompt is specified
    if not args.prompt:
        if args.sam or args.nninteractive:
            print("[WARNING] --sam or --nninteractive is only valid when using --prompt. Ignoring these flags.")
        
    
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
    elif args.prompt:
        # either sam or nninteractive must be specified
        if not (args.sam or args.nninteractive):
            print("[ERROR] Either --sam or --nninteractive must be specified when using --prompt")
            parser.print_help()
            exit(1)
        if args.sam:
            from sprout_core.sam_predict import run_sam_yaml
            from sprout_core.batch_sam import run_batch_sam
            if not args.config:
                print("[ERROR] --config is required when using --prompt --sam")
                parser.print_help()
                exit(1)
            # config = load_config(args.config)
            if args.batch:
                run_batch_sam(args.config)
            else:
                run_sam_yaml(args.config)
        elif args.nninteractive:
            import sprout_core.nninteractive_predict as nninteractive_predict 
            import sprout_core.batch_nninteractive as batch_nninteractive
            if not args.config:
                print("[ERROR] --config is required when using --prompt --nninteractive")
                parser.print_help()
                exit(1)
            if args.batch:
                batch_nninteractive.run_batch_nninteractive(args.config)
            else:
                
                nninteractive_predict.run_nninteractive_yaml(args.config)

    else:
        print("[ERROR] No valid action specified.")
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
