import argparse
import yaml
from sprout_core.make_seeds import run_make_seeds
from sprout_core.make_grow import run_make_grow
from sprout_core.make_adaptive_seed import run_make_adaptive_seed
from sprout_core.make_sweep_adaptive_seed import run_sweep_adaptive_seed

from sprout_core.batch_grow import run_batch_grow
from sprout_core.batch_seeds import run_batch_adaptive_seed, run_batch_seeds, run_batch_sweep_adaptive_seed

import os
from pathlib import Path

from sprout_core import __version__

HINT = "Templates: sprout --template <step>. Docs: https://github.com/EchanHe/SPROUT"

# step: (command, template, batch template, its CSV)
TEMPLATES = {
    "seeds": ("--seeds", "make_seeds.yaml", "batch_seeds.yaml", "seeds_input.csv"),
    "adaptive_seed": ("--adaptive_seed", "make_adaptive_seed.yaml", "batch_adaptive_seed.yaml", "seeds_input.csv"),
    "sweep_adaptive": ("--sweep_adaptive", "sweep_adaptive_seed.yaml", "batch_sweep_adaptive_seed.yaml",
                       "sweep_adaptive_seeds_input.csv"),
    "grow": ("--grow", "make_grow.yaml", "batch_grow.yaml", "grow_input.csv"),
    "nninteractive": ("--prompt --nninteractive", "nninteractive_predict.yaml", "batch_nninteractive.yaml",
                      "nninteractive_input.csv"),
    "sam": ("--prompt --sam", "sam_predict.yaml", "batch_sam.yaml", "sam_input.csv"),
}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"{self.prog}: error: {message}\n{HINT}\n")


def template_dir():
    """The templates: in the package when installed, else template/ of the clone."""
    here = Path(__file__).resolve().parent
    return next(d for d in (here / "templates", here.parent / "template") if d.is_dir())


def write_template(step, batch):
    command, single, batch_yaml, csv = TEMPLATES[step]
    names = [batch_yaml, csv] if batch else [single]
    here = [name for name in names if Path(name).exists()]
    if here:
        print(f"[ERROR] {', '.join(here)} already here, not overwritten.")
        return 1
    for name in names:
        # the batch YAML points to its CSV in template/; here they sit side by side
        Path(name).write_bytes((template_dir() / name).read_bytes().replace(b'"./template/', b'"./'))
    print(f"Wrote {', '.join(names)}. Edit the paths, then: sprout {command}{' --batch' if batch else ''} "
          f"--config {names[0]}")
    return 0


def load_config(path):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        exit(1)

def main():
    parser = Parser(
        description="SPROUT CLI: generate seeds with a YAML config",
        usage=
        """
        For seeds generation:
            sprout --seeds --config path/to/config.yaml
        
        For adaptive seed generation (classic)
            sprout --adaptive_seed --config path/to/config.yaml

        For fast sweep + adaptive combine (recommended)
            sprout --sweep_adaptive --config path/to/config.yaml
        
        For growing
            sprout --grow --config path/to/config.yaml
            
        For Predictions using PROMPT extracted from seeds for SAM
            sprout --prompt --sam --config path/to/config.yaml  
        
        For Predictions using PROMPT extracted from seeds for nnInteractive
            sprout --prompt --nninteractive --config path/to/config.yaml
            
        To run in batch mode (pipeline version), add --batch:
            sprout --seeds --batch --config path/to/config.yaml

        To write the YAML template of a step here (add --batch for the batch template):
            sprout --template seeds

        In a clone of the repository, "python sprout.py" works the same as "sprout".
        """ + HINT
    )
    

    # Mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seeds', action='store_true', help="Run the seed generation function")
    group.add_argument('--adaptive_seed', action='store_true', help="Run the adaptive seed generation function (classic)")
    group.add_argument('--sweep_adaptive', action='store_true', help="Run fast sweep + adaptive combine (recommended)")
    group.add_argument('--grow', action='store_true', help="Run the grow function")
    group.add_argument('--prompt', action='store_true', help="Run the PROMPT extraction function")
    group.add_argument('--template', choices=list(TEMPLATES), metavar="STEP",
                       help=f"Write the YAML template of a step here: {', '.join(TEMPLATES)}")

    # a group args only when --prompt is specified
    # sam or nninteractive
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--sam', action='store_true', help="Use SAM for prompt extraction")
    prompt_group.add_argument('--nninteractive', '--nn', action='store_true', help="Use nnInteractive for prompt extraction")
    
    parser.add_argument('--batch', action='store_true', help="Run in batch mode using pipeline functions")
    parser.add_argument('--config', type=str, help="Path to the YAML config file")
    parser.add_argument('--version', action='version', version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    if args.template:
        exit(write_template(args.template, args.batch))
    if args.config is None:
        parser.error("the following arguments are required: --config")
    step = next((s for s in TEMPLATES if getattr(args, s)), "<step>")
    if not args.config.endswith(".yaml"):
        print(f"[ERROR] The config must be a .yaml file: {args.config}")
        exit(1)
    if not os.path.isfile(args.config):
        print(f"[ERROR] Config not found: {args.config}. Get a template: sprout --template {step}")
        exit(1)

    # check sam or interactive only when --prompt is specified
    if not args.prompt:
        if args.sam or args.nninteractive:
            print("[WARNING] --sam or --nninteractive is only valid when using --prompt. Ignoring these flags.")
        
    
    if args.seeds:
        # config = load_config(args.config)
        if args.batch:
            run_batch_seeds(args.config)
        else:
            run_make_seeds(args.config)
    elif args.grow:
        # config = load_config(args.config)
        if args.batch:
            run_batch_grow(args.config)
        else:
            run_make_grow(args.config)
    elif args.adaptive_seed:
        
        if args.batch:
            run_batch_adaptive_seed(args.config)
        else:
            run_make_adaptive_seed(args.config)
    elif args.sweep_adaptive:
        if not args.config:
            print("[ERROR] --config is required when using --sweep_adaptive")
            parser.print_help()
            exit(1)
        if args.batch:
            run_batch_sweep_adaptive_seed(args.config)
        else:
            run_sweep_adaptive_seed(args.config)
    elif args.prompt:
        # either sam or nninteractive must be specified
        if not (args.sam or args.nninteractive):
            print("[ERROR] Either --sam or --nninteractive must be specified when using --prompt")
            parser.print_help()
            exit(1)
        if args.sam:
            from sprout_core.sam_predict import run_sam_yaml
            from sprout_core.batch_sam import run_batch_sam
            # config = load_config(args.config)
            if args.batch:
                run_batch_sam(args.config)
            else:
                run_sam_yaml(args.config)
        elif args.nninteractive:
            try:
                import sprout_core.nninteractive_predict as nninteractive_predict
                import sprout_core.batch_nninteractive as batch_nninteractive
            except ImportError as e:
                print(f"[ERROR] nnInteractive needs extra packages ({e}). Install them with: "
                      'pip install "sprout-seg[ai]"')
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
