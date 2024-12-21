# How to Use

## Demo

Once the package is installed, you can run an SDS experiment using the following command:

```bash
python main.py --config-name sds
```

To enable logging with **Weights & Biases (wandb)**, specify your wandb API key like this:

```bash
python main.py --config-name sds +wandb_key=your_wandb_key
```

If you prefer not to use wandb, you can view logs locally with **TensorBoard**. Logs are saved in the `PROJECT_ROOT/output` directory:

```bash
tensorboard --logdir output
```

## Additional Reference

Example configuration files are available in the `PROJECT_ROOT/conf` directory. These configurations follow a **hierarchical structure**.

### Configuration Example

When you run the command:

```bash
python main.py --config-name sds
```

The configuration file `PROJECT_ROOT/conf/sds.yaml` is loaded. Here’s an example content of the file:

```yaml
name: sds  # Name for instantiation from the registry
group: ${name}
note: ""

defaults:
  - base  # Inherit from PROJECT_ROOT/conf/base.yaml
  - rasterizer: pixels  # Use pixels as the 2D representation, inherit from PROJECT_ROOT/conf/rasterizer/pixels.yaml
  - wt_schedule: dreamfusion  # Use DreamFusion schedule, inherit from PROJECT_ROOT/conf/wt_schedule/dreamfusion.yaml
  - algorithm: sds_algorithm  # Use the SDS optimization algorithm, inherit from PROJECT_ROOT/conf/algorithm/sds_algorithm.yaml
  - _self_  # Override default values with those defined below

prompt: "a photograph of an astronaut riding a horse."
neg_prompt: ""

iterations: 2000
log_interval: 200

algorithm:
  guidance_scale: 7.5

rasterizer:
  batch_size: 4
  optimizer:
    opt_args:
      lr: 0.1
```

### Notes:
- **Hierarchical structure**: You can refer more about the hierarchical config structure in the [Hydra documentation](https://hydra.cc/docs/configure_hydra/intro/).
- **Custom settings**: You can override values as needed by modifying the YAML files or passing arguments directly via the command line. Also refer to the above links for more details.

This structure makes it easy to experiment with different combinations of settings, algorithms, or representations.

> [!WARNING]  
> The framework automatically scans and loads modules from `PROJECT_ROOT/sds_2d/algorithm/`, `PROJECT_ROOT/sds_2d/rasterizer/`, and `PROJECT_ROOT/sds_2d/wt_schedule/`.
> This offers convenience in extending the framework with new algorithms, representations, or weight and timestep schedules.
> However, **ensure that server files are synchronized with local files, and no redundant Python files exist in these directories on the server**, 
> as redundant files may be unintentionally loaded and cause errors.
> This is particularly critical when using IDEs like PyCharm.