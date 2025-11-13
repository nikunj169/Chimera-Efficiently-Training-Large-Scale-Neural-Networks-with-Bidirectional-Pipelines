"""
CHIMERA INTERACTIVE PLAYGROUND
==============================

Interactive tool to explore Chimera pipeline parallelism with your own inputs.
Visualize schedules, analyze performance, and experiment with configurations.
"""
import torch
from chimera.engine import BidirectionalSchedule, ScheduleType, StagePartitioner
from chimera.models import BertConfig, BertForPipelineParallelism, GPT2Config, GPT2ForPipelineParallelism
from chimera.config import PerformanceModel, AutoTuner


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")


def print_subheader(text):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.END}")


def get_int_input(prompt, min_val=None, max_val=None, default=None):
    """Get validated integer input from user"""
    while True:
        if default is not None:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        try:
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"{Colors.RED}Value must be >= {min_val}{Colors.END}")
                continue
            if max_val is not None and value > max_val:
                print(f"{Colors.RED}Value must be <= {max_val}{Colors.END}")
                continue
            return value
        except ValueError:
            print(f"{Colors.RED}Please enter a valid integer{Colors.END}")


def get_float_input(prompt, min_val=None, max_val=None, default=None):
    """Get validated float input from user"""
    while True:
        if default is not None:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        try:
            value = float(user_input)
            if min_val is not None and value < min_val:
                print(f"{Colors.RED}Value must be >= {min_val}{Colors.END}")
                continue
            if max_val is not None and value > max_val:
                print(f"{Colors.RED}Value must be <= {max_val}{Colors.END}")
                continue
            return value
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number{Colors.END}")


def get_choice(prompt, options):
    """Get choice from list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            choice = int(input(f"Enter choice (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return choice - 1
            print(f"{Colors.RED}Invalid choice. Try again.{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Please enter a number.{Colors.END}")


def mode_1_schedule_explorer():
    """Interactive schedule generation and visualization"""
    print_header("SCHEDULE EXPLORER")
    
    print(f"\n{Colors.YELLOW}Generate and visualize Chimera pipeline schedules{Colors.END}")
    
    # Get parameters
    print_subheader("Pipeline Configuration")
    D = get_int_input("Number of pipeline stages (D, must be even)", min_val=2, max_val=32, default=4)
    
    # Validate D is even
    if D % 2 != 0:
        print(f"{Colors.RED}D must be even! Adjusting to {D+1}...{Colors.END}")
        D = D + 1
    
    N = get_int_input("Number of micro-batches (N)", min_val=1, max_val=64, default=8)
    W = get_int_input("Number of replicas (W)", min_val=1, max_val=16, default=1)
    
    # Choose schedule strategy
    if N > D:
        print_subheader("Schedule Strategy (N > D)")
        strategies = ["BASE", "DIRECT_CONCAT", "FORWARD_DOUBLING", "BACKWARD_HALVING"]
        strategy_idx = get_choice("Choose scheduling strategy:", strategies)
        strategy = ScheduleType[strategies[strategy_idx]]
    else:
        strategy = ScheduleType.BASE
        print(f"\n{Colors.CYAN}Using BASE strategy (N <= D){Colors.END}")
    
    # Generate schedule
    print(f"\n{Colors.GREEN}Generating schedule...{Colors.END}")
    scheduler = BidirectionalSchedule(D=D, N=N, W=W)
    schedule = scheduler.build_schedule(strategy)
    
    # Display visualization
    print_subheader("Schedule Timeline")
    print(scheduler.visualize_schedule())
    
    # Display statistics
    print_subheader("Schedule Statistics")
    stats = scheduler.compute_bubble_stats()
    
    print(f"  Pipeline Depth (D):        {Colors.BOLD}{D}{Colors.END}")
    print(f"  Micro-batches (N):         {Colors.BOLD}{N}{Colors.END}")
    print(f"  Replicas (W):              {Colors.BOLD}{W}{Colors.END}")
    print(f"  Total bubbles:             {Colors.BOLD}{stats.total_bubbles}{Colors.END}")
    print(f"  Bubble ratio:              {Colors.BOLD}{stats.bubble_ratio:.3f}{Colors.END}")
    print(f"  Critical path (Cf):        {Colors.BOLD}{stats.critical_path_forward}{Colors.END}")
    print(f"  Critical path (Cb):        {Colors.BOLD}{stats.critical_path_backward}{Colors.END}")
    
    # Compare with baseline
    gpipe_bubbles = 2 * (D - 1)
    improvement = (gpipe_bubbles - stats.total_bubbles) / gpipe_bubbles * 100
    
    print_subheader("Comparison with GPipe")
    print(f"  GPipe bubbles:             {Colors.RED}{gpipe_bubbles}{Colors.END}")
    print(f"  Chimera bubbles:           {Colors.GREEN}{stats.total_bubbles}{Colors.END}")
    print(f"  Improvement:               {Colors.GREEN}{Colors.BOLD}{improvement:.1f}%{Colors.END}")
    
    # Show eager sync stages
    eager_stages = scheduler.get_eager_sync_stages()
    print(f"\n  Eager sync stages:         {Colors.YELLOW}{eager_stages}{Colors.END}")


def mode_2_performance_estimator():
    """Interactive performance estimation"""
    print_header("PERFORMANCE ESTIMATOR")
    
    print(f"\n{Colors.YELLOW}Estimate iteration time and throughput{Colors.END}")
    
    # Get network parameters
    print_subheader("Network Characteristics")
    print("(Leave default for typical values)")
    
    alpha = get_float_input("Network latency α (seconds)", min_val=0, default=1e-5)
    beta = get_float_input("Inverse bandwidth β (s/byte)", min_val=0, default=1e-9)
    
    bandwidth_gbps = 1 / (beta * 1e9)
    print(f"{Colors.CYAN}→ Bandwidth: {bandwidth_gbps:.2f} GB/s{Colors.END}")
    
    # Get computation parameters
    print_subheader("Computation Parameters")
    F_t = get_float_input("Forward time per micro-batch (seconds)", min_val=0, default=0.1)
    
    use_recompute = input("Enable recomputation? (y/n, default: n): ").strip().lower() == 'y'
    
    # Get pipeline config
    print_subheader("Pipeline Configuration")
    D = get_int_input("Pipeline stages (D, even)", min_val=2, max_val=32, default=4)
    if D % 2 != 0:
        D = D + 1
        print(f"{Colors.CYAN}Adjusted to D={D} (even){Colors.END}")
    
    N = get_int_input("Micro-batches (N)", min_val=1, max_val=64, default=8)
    W = get_int_input("Replicas (W)", min_val=1, max_val=16, default=2)
    
    # Message size
    print_subheader("Message Size")
    print("Typical values: BERT (4-8 MB), GPT-2 (8-16 MB)")
    message_mb = get_float_input("Message size (MB)", min_val=0.1, default=4.0)
    message_bytes = int(message_mb * 1024 * 1024)
    
    # Create performance model
    print(f"\n{Colors.GREEN}Computing performance estimate...{Colors.END}")
    
    perf_model = PerformanceModel(
        alpha=alpha,
        beta=beta,
        F_t=F_t,
        recompute_enabled=use_recompute
    )
    
    # Generate schedule
    scheduler = BidirectionalSchedule(D=D, N=N, W=W)
    schedule = scheduler.build_schedule()
    stats = scheduler.compute_bubble_stats()
    
    # Estimate performance
    perf = perf_model.estimate_iteration_time(
        D=D, N=N, W=W,
        C_f=stats.critical_path_forward,
        C_b=stats.critical_path_backward,
        message_size_bytes=message_bytes,
        eager_sync_stages=scheduler.get_eager_sync_stages()
    )
    
    # Display results
    print_subheader("Performance Estimate")
    print(f"  Forward time (F_t):        {Colors.BOLD}{perf_model.F_t*1000:.2f} ms{Colors.END}")
    print(f"  Backward time (B_t):       {Colors.BOLD}{perf_model.B_t*1000:.2f} ms{Colors.END}")
    if use_recompute:
        print(f"  {Colors.YELLOW}(Recompute adds 33% overhead){Colors.END}")
    
    print(f"\n  Total iteration time:      {Colors.BOLD}{Colors.GREEN}{perf['total_time']:.3f} s{Colors.END}")
    print(f"  Throughput:                {Colors.BOLD}{Colors.GREEN}{perf['throughput']:.2f} micro-batches/s{Colors.END}")
    
    print(f"\n  Forward time:              {perf['forward_time']:.3f} s")
    print(f"  Backward time:             {perf['backward_time']:.3f} s")
    print(f"  Allreduce time:            {perf['allreduce_time']:.3f} s")
    print(f"  Unoverlapped allreduce:    {perf['unoverlapped_allreduce']:.3f} s")
    print(f"  Overlap fraction:          {Colors.CYAN}{perf['overlap_fraction']*100:.1f}%{Colors.END}")


def mode_3_model_configurator():
    """Interactive model configuration and partitioning"""
    print_header("MODEL CONFIGURATOR")
    
    print(f"\n{Colors.YELLOW}Configure and partition large models{Colors.END}")
    
    # Choose model type
    print_subheader("Model Selection")
    models = ["BERT-48 (669M params)", "GPT-2 64L (1.3B params)", "Custom BERT", "Custom GPT-2"]
    model_idx = get_choice("Choose model:", models)
    
    if model_idx == 0:  # BERT-48
        config = BertConfig(
            num_hidden_layers=48,
            hidden_size=1024,
            vocab_size=30522,
            num_attention_heads=16,
            intermediate_size=4096
        )
        model_type = "bert"
        print(f"{Colors.GREEN}Using BERT-48 preset configuration{Colors.END}")
        
    elif model_idx == 1:  # GPT-2 64L
        config = GPT2Config(
            n_layer=64,
            n_embd=1280,
            n_head=20,
            n_inner=5120
        )
        model_type = "gpt2"
        print(f"{Colors.GREEN}Using GPT-2 64L preset configuration{Colors.END}")
        
    elif model_idx == 2:  # Custom BERT
        print_subheader("Custom BERT Configuration")
        num_layers = get_int_input("Number of layers", min_val=1, max_val=96, default=24)
        hidden_size = get_int_input("Hidden size", min_val=128, max_val=4096, default=768)
        num_heads = get_int_input("Attention heads", min_val=1, max_val=64, default=12)
        intermediate_size = get_int_input("Intermediate size", min_val=256, max_val=16384, default=3072)
        
        config = BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size
        )
        model_type = "bert"
        
    else:  # Custom GPT-2
        print_subheader("Custom GPT-2 Configuration")
        num_layers = get_int_input("Number of layers", min_val=1, max_val=96, default=32)
        hidden_size = get_int_input("Hidden size", min_val=128, max_val=4096, default=1024)
        num_heads = get_int_input("Attention heads", min_val=1, max_val=64, default=16)
        intermediate_size = get_int_input("Intermediate size", min_val=256, max_val=16384, default=4096)
        
        config = GPT2Config(
            n_layer=num_layers,
            n_embd=hidden_size,
            n_head=num_heads,
            n_inner=intermediate_size
        )
        model_type = "gpt2"
    
    # Get pipeline configuration
    print_subheader("Pipeline Configuration")
    D = get_int_input("Number of pipeline stages (D, even)", min_val=2, max_val=32, default=4)
    if D % 2 != 0:
        D = D + 1
    
    # Create model
    print(f"\n{Colors.GREEN}Creating model...{Colors.END}")
    
    if model_type == "bert":
        model = BertForPipelineParallelism(config, num_stages=D)
        num_layers = config.num_hidden_layers
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        intermediate_size = config.intermediate_size
    else:
        model = GPT2ForPipelineParallelism(config, num_stages=D)
        num_layers = config.n_layer
        hidden_size = config.n_embd
        vocab_size = config.vocab_size
        intermediate_size = config.n_inner
    
    # Display model info
    print_subheader("Model Information")
    print(f"  Model type:                {Colors.BOLD}{model_type.upper()}{Colors.END}")
    print(f"  Total layers:              {Colors.BOLD}{num_layers}{Colors.END}")
    print(f"  Hidden size:               {Colors.BOLD}{hidden_size}{Colors.END}")
    print(f"  Vocabulary size:           {Colors.BOLD}{vocab_size}{Colors.END}")
    print(f"  Intermediate size:         {Colors.BOLD}{intermediate_size}{Colors.END}")
    
    # Display partitioning
    print_subheader("Pipeline Partitioning")
    for i, stage in enumerate(model.stages):
        layer_range = stage.layer_range
        num_blocks = layer_range[1] - layer_range[0]
        print(f"  Stage {i}:  Layers {layer_range[0]:2d}-{layer_range[1]:2d}  "
              f"({Colors.CYAN}{num_blocks:2d} blocks{Colors.END})")
    
    # Memory estimation
    B = get_int_input("\nMicro-batch size (B) for memory estimate", min_val=1, max_val=128, default=4)
    
    print(f"\n{Colors.GREEN}Computing memory requirements...{Colors.END}")
    
    partitioner = StagePartitioner(
        num_stages=D,
        model_config={
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size,
            'num_attention_heads': config.num_attention_heads if model_type == 'bert' else config.n_head,
            'intermediate_size': intermediate_size,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
    )
    
    partitions = partitioner.partition_even_blocks()
    memory_profile = partitioner.get_memory_profile(partitions, micro_batch_size=B)
    
    print_subheader(f"Memory Profile (B={B})")
    for stage_id, mem in memory_profile.items():
        total_mb = mem.weight_memory_mb + mem.peak_activation_mb
        print(f"  Stage {stage_id}:  {total_mb:7.2f} MB  "
              f"(weights: {mem.weight_memory_mb:6.2f} MB, "
              f"activations: {mem.peak_activation_mb:6.2f} MB)")
    
    # Test forward pass
    print(f"\n{Colors.CYAN}Testing data flow...{Colors.END}")
    stage_0 = model.get_stage(0)
    
    input_ids = torch.randint(0, vocab_size, (2, 64))
    output = stage_0(input_ids)
    
    print(f"  Input shape:               {Colors.GREEN}{tuple(input_ids.shape)}{Colors.END}")
    print(f"  Stage 0 output shape:      {Colors.GREEN}{tuple(output.shape)}{Colors.END}")
    print(f"  {Colors.GREEN}✓ Model ready for training!{Colors.END}")


def mode_4_autotune():
    """Interactive autotuning for optimal configuration"""
    print_header("AUTO-CONFIGURATION")
    
    print(f"\n{Colors.YELLOW}Automatically select optimal (W, D, B) configuration{Colors.END}")
    
    # Get constraints
    print_subheader("System Constraints")
    P = get_int_input("Total number of processes (GPUs)", min_val=2, max_val=512, default=16)
    memory_gb = get_float_input("Memory per device (GB)", min_val=1.0, max_val=80.0, default=16.0)
    
    # Get workload
    print_subheader("Workload Configuration")
    target_batch_size = get_int_input("Target global batch size", min_val=1, max_val=1024, default=128)
    
    # Choose model
    models = ["BERT-48", "GPT-2 64L", "Custom"]
    model_idx = get_choice("Choose model:", models)
    
    if model_idx == 0:
        model_config = {
            'num_layers': 48,
            'hidden_size': 1024,
            'vocab_size': 30522,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
    elif model_idx == 1:
        model_config = {
            'num_layers': 64,
            'hidden_size': 1280,
            'vocab_size': 50257,
            'num_attention_heads': 20,
            'intermediate_size': 5120,
            'max_sequence_length': 1024,
            'dtype_bytes': 2
        }
    else:
        print_subheader("Custom Model")
        model_config = {
            'num_layers': get_int_input("Number of layers", min_val=1, default=24),
            'hidden_size': get_int_input("Hidden size", min_val=128, default=768),
            'vocab_size': get_int_input("Vocab size", min_val=1000, default=30522),
            'num_attention_heads': get_int_input("Attention heads", min_val=1, default=12),
            'intermediate_size': get_int_input("Intermediate size", min_val=256, default=3072),
            'max_sequence_length': get_int_input("Max sequence length", min_val=64, default=512),
            'dtype_bytes': 2
        }
    
    # Get performance params
    print_subheader("Performance Parameters")
    F_t = get_float_input("Forward time estimate (s)", min_val=0.001, default=0.1)
    
    # Create autotuner
    print(f"\n{Colors.GREEN}Running AutoTune...{Colors.END}")
    
    perf_model = PerformanceModel(alpha=1e-5, beta=1e-9, F_t=F_t)
    
    autotuner = AutoTuner(
        perf_model=perf_model,
        total_processes=P,
        memory_budget_gb=memory_gb,
        model_config=model_config
    )
    
    config = autotuner.select_configuration(target_batch_size=target_batch_size)
    
    # Display results
    print_subheader("Optimal Configuration")
    print(f"  Replicas (W):              {Colors.BOLD}{Colors.GREEN}{config['W']}{Colors.END}")
    print(f"  Pipeline stages (D):       {Colors.BOLD}{Colors.GREEN}{config['D']}{Colors.END}")
    print(f"  Micro-batch size (B):      {Colors.BOLD}{Colors.GREEN}{config['B']}{Colors.END}")
    print(f"  Micro-batches/replica (N): {Colors.BOLD}{Colors.GREEN}{config['N']}{Colors.END}")
    print(f"  Global batch size:         {Colors.BOLD}{config['global_batch_size']}{Colors.END}")
    print(f"  Schedule strategy:         {Colors.BOLD}{config['schedule_strategy']}{Colors.END}")
    
    print_subheader("Performance Prediction")
    perf = config['performance']
    print(f"  Total iteration time:      {Colors.BOLD}{perf['total_time']:.3f} s{Colors.END}")
    print(f"  Throughput:                {Colors.BOLD}{perf['throughput']:.2f} micro-batches/s{Colors.END}")
    print(f"  Samples per second:        {Colors.BOLD}{perf['throughput'] * config['B']:.2f}{Colors.END}")
    
    # Show command to use
    print_subheader("Usage Command")
    print(f"{Colors.CYAN}torchrun --nproc_per_node={config['W']*config['D']} \\")
    print(f"    chimera/runners/train.py \\")
    print(f"    --W {config['W']} --D {config['D']} --N {config['N']} --B {config['B']}{Colors.END}")


def main():
    """Main interactive menu"""
    while True:
        print_header("CHIMERA INTERACTIVE PLAYGROUND")
        
        print(f"\n{Colors.YELLOW}Explore Chimera pipeline parallelism with your inputs!{Colors.END}\n")
        
        modes = [
            "Schedule Explorer - Visualize pipeline schedules",
            "Performance Estimator - Estimate iteration time",
            "Model Configurator - Configure and partition models",
            "Auto-Configuration - Find optimal (W, D, B)",
            "Exit"
        ]
        
        choice = get_choice("Choose a mode:", modes)
        
        if choice == 0:
            mode_1_schedule_explorer()
        elif choice == 1:
            mode_2_performance_estimator()
        elif choice == 2:
            mode_3_model_configurator()
        elif choice == 3:
            mode_4_autotune()
        elif choice == 4:
            print(f"\n{Colors.GREEN}Thanks for using Chimera Playground!{Colors.END}\n")
            break
        
        # Ask to continue
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        cont = input(f"\nPress Enter to return to main menu (or 'q' to quit): ")
        if cont.lower() == 'q':
            print(f"\n{Colors.GREEN}Thanks for using Chimera Playground!{Colors.END}\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Goodbye!{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()
