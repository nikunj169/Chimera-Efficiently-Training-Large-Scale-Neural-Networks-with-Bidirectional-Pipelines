# Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines

## Overview

This project is a PyTorch-based library that implements **Chimera**, a novel pipeline parallelism scheme designed for the efficient training of large-scale neural networks. It addresses the significant challenges of training models with billions of parameters, particularly the issue of GPU underutilization (known as "bubbles") in traditional pipeline parallelism.

## Motivation: Why Chimera?

Training state-of-the-art deep learning models, such as large language models (LLMs) like BERT and GPT variants, requires immense computational resources. Distributing these models across multiple GPUs is essential, but existing pipeline parallelism methods often suffer from inefficiencies due to idle GPU time.

**Chimera** aims to solve this by introducing a **bidirectional pipeline scheduling algorithm**. This approach significantly reduces idle periods, leading to more balanced memory consumption and higher training throughput. Our implementation is based on the research detailed in the paper:

**[2107.06925v5.pdf: Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](2107.06925v5.pdf)**

## Key Features & Impact

By implementing the Chimera approach, this project demonstrates:

*   **Significant Throughput Improvement**: Achieves **1.16x-2.34x higher training throughput** compared to traditional methods, as shown in the original research. This translates directly to faster model development and reduced computational costs.
*   **Enhanced Resource Utilization**: Reduces GPU idle time ("bubbles") by up to 50%, ensuring that expensive hardware resources are used more effectively.
*   **Balanced Memory Consumption**: Optimizes memory usage across GPUs, enabling the training of larger models or larger batch sizes that would otherwise be infeasible due to memory constraints.
*   **Scalability**: Designed for efficient training on large-scale distributed environments, including supercomputers with thousands of GPUs, showcasing expertise in high-performance computing for AI.

## Project Structure and Functionality (Detailed Component Breakdown)

This section provides a detailed look into each significant Python file, explaining its role, inputs, outputs, and how it contributes to the overall Chimera system.

### Top-Level Files

*   **`chimera_playground.py`**
    *   **Purpose:** An interactive script designed to showcase the core functionalities of the Chimera library. It serves as an excellent entry point for understanding the system's capabilities without diving deep into the distributed training setup.
    *   **Working:** Guides the user through different modes, demonstrating scheduling, partitioning, performance estimation, and autotuning. It orchestrates calls to various `chimera` modules.
    *   **Inputs:** User selections for demonstration modes and parameters.
    *   **Outputs:** Visualizations of schedules, performance estimates, and model configurations.
    *   **What it shows:** How Chimera's components work together, the impact of different configurations, and the benefits of autotuning.
    *   **How we did it:** By integrating calls to the `chimera.engine`, `chimera.config`, and `chimera.models` modules in an interactive command-line interface.

*   **`demo_model.py`**
    *   **Purpose:** Demonstrates how a complex neural network model is adapted and prepared for Chimera's pipeline parallelism.
    *   **Working:** Loads a pre-defined model (e.g., BERT or GPT-2), applies the `StagePartitioner` to break it into pipeline stages, and shows the resulting model structure.
    *   **Inputs:** A base model definition (e.g., from `chimera/models`).
    *   **Outputs:** A staged model ready for pipeline execution, along with memory estimates for each stage.
    *   **What it shows:** The process of transforming a monolithic model into a pipeline-compatible architecture.
    *   **How we did it:** By utilizing the `chimera.engine.partition.StagePartitioner` and `chimera.models` module.

*   **`demo_performance.py`**
    *   **Purpose:** Illustrates the performance benefits of Chimera by running simplified benchmarks or simulations.
    *   **Working:** Uses the `PerformanceModel` to estimate throughput for various configurations (Chimera vs. baselines) or runs small-scale training loops to measure actual performance.
    *   **Inputs:** Configuration parameters (e.g., number of GPUs, model size, micro-batch size).
    *   **Outputs:** Performance metrics like throughput (sequences/second) and bubble ratios.
    *   **What it shows:** The quantifiable advantages of Chimera's scheduling over other pipeline parallelism methods.
    *   **How we did it:** By leveraging `chimera.config.perf_model` and potentially `chimera.runners.benchmarks`.

*   **`demo_schedule.py`**
    *   **Purpose:** Provides a visual representation of Chimera's bidirectional pipeline schedule.
    *   **Working:** Generates a `BidirectionalSchedule` and visualizes the temporal execution of forward and backward passes across different GPUs and pipeline stages.
    *   **Inputs:** Pipeline depth, micro-batch size, number of workers.
    *   **Outputs:** A graphical or textual representation of the schedule, highlighting overlapping computations and reduced bubbles.
    *   **What it shows:** The core mechanism of Chimera's efficiency and how it minimizes idle time.
    *   **How we did it:** By instantiating and visualizing the output of `chimera.engine.schedule.BidirectionalSchedule`.

*   **`requirements.txt`**
    *   **Purpose:** Lists all Python package dependencies required to run the project.
    *   **Working:** Used by `pip` to install necessary libraries.
    *   **Inputs:** None (static file).
    *   **Outputs:** None (static file).
    *   **What it shows:** The project's software environment.

*   **`2107.06925v5.pdf`**
    *   **Purpose:** The original research paper detailing the Chimera method.
    *   **Working:** Provides the theoretical foundation, algorithms, and experimental results that this project implements.
    *   **Inputs:** None (static file).
    *   **Outputs:** None (static file).
    *   **What it shows:** The scientific basis of the project.

### `chimera/` directory

#### `chimera/config/`

*   **`autotune.py`**
    *   **Purpose:** Automates the complex process of finding the optimal parallelization strategy for a given model and hardware setup.
    *   **Working:** The `AutoTuner` class systematically explores different configurations (e.g., pipeline depth, data parallelism width, micro-batch sizes). For each configuration, it uses the `StagePartitioner` to check memory feasibility and the `PerformanceModel` to predict training throughput. It then selects the configuration that yields the best predicted performance.
    *   **Inputs:** Model definition, hardware constraints (e.g., GPU memory), search space for parallelization parameters.
    *   **Outputs:** An optimized configuration (pipeline depth, data parallelism, micro-batch size) that maximizes throughput while respecting memory limits.
    *   **What it shows:** How to abstract away the complexity of distributed training setup, making the system user-friendly and efficient.
    *   **How we did it:** By integrating `StagePartitioner` and `PerformanceModel` within a search algorithm to find optimal parameters.

*   **`perf_model.py`**
    *   **Purpose:** Provides an analytical model to predict the training throughput of different Chimera configurations.
    *   **Working:** The `PerformanceModel` takes various configuration parameters (e.g., number of stages, micro-batch size, communication costs) and calculates an estimated training time per iteration, considering factors like computation, communication overhead, and pipeline bubbles.
    *   **Inputs:** Configuration parameters (e.g., `D` for pipeline depth, `W` for data parallelism, `B` for micro-batch size), estimated computation and communication costs.
    *   **Outputs:** Predicted training throughput (sequences/second) for a given configuration.
    *   **What it shows:** The ability to model complex system performance and use it for optimization.
    *   **How we did it:** By implementing the analytical equations derived in the Chimera paper, which account for various overheads in pipeline parallelism.

#### `chimera/engine/`

*   **`partition.py`**
    *   **Purpose:** Handles the spatial distribution of a neural network's layers across multiple devices and estimates memory usage.
    *   **Working:** The `StagePartitioner` takes a PyTorch `nn.Module` and a target number of pipeline stages. It analyzes the model's layers, groups them into stages, and assigns them to virtual devices. Crucially, it estimates the memory footprint (weights, activations, optimizer states) for each stage, which is vital for ensuring memory feasibility.
    *   **Inputs:** A PyTorch `nn.Module` (the model to be partitioned), desired number of pipeline stages, device memory limits.
    *   **Outputs:** A list of `nn.Module` instances, each representing a pipeline stage, along with detailed memory estimates for each stage.
    *   **What it shows:** How to break down a large model for distributed execution while being memory-aware.
    *   **How we did it:** By implementing logic to traverse the model graph, group layers, and calculate memory requirements based on layer types and input shapes.

*   **`recompute.py`**
    *   **Purpose:** Implements activation recomputation (also known as "gradient checkpointing") to reduce memory consumption during training.
    *   **Working:** Instead of storing all intermediate activations from the forward pass (which can be memory-intensive for deep networks), recomputation re-runs parts of the forward pass during the backward pass to regenerate necessary activations on-the-fly.
    *   **Inputs:** A PyTorch module or a sequence of operations.
    *   **Outputs:** Reduced peak memory usage, especially for activations, at the cost of some additional computation during the backward pass.
    *   **What it shows:** Advanced memory optimization techniques crucial for training extremely large models.
    *   **How we did it:** By integrating PyTorch's `torch.utils.checkpoint` or similar custom logic to selectively re-execute forward pass operations.

*   **`runtime.py`**
    *   **Purpose:** Manages the execution environment and communication primitives for distributed training.
    *   **Working:** Provides abstractions for inter-device and inter-node communication (e.g., sending/receiving tensors between pipeline stages), device management, and synchronization. It acts as the backbone for executing the generated schedules.
    *   **Inputs:** Communication backend (e.g., `gloo`, `nccl`), device IDs, distributed group information.
    *   **Outputs:** Facilitates seamless data transfer and synchronization between distributed processes.
    *   **What it shows:** Expertise in building robust distributed systems for machine learning.
    *   **How we did it:** By wrapping and extending PyTorch's distributed communication primitives (`torch.distributed`).

*   **`schedule.py`**
    *   **Purpose:** Implements the core Chimera bidirectional pipeline scheduling algorithm.
    *   **Working:** The `BidirectionalSchedule` class generates a precise temporal execution plan for each GPU. It determines when each micro-batch's forward and backward passes should run on each pipeline stage, optimizing for minimal "bubbles" (idle time) by overlapping operations across stages and directions.
    *   **Inputs:** A list of partitioned model stages, micro-batch size, pipeline depth.
    *   **Outputs:** A detailed sequence of `ScheduleSlot` objects for each GPU, indicating the operation (forward/backward), micro-batch ID, and stage ID to execute at each time step.
    *   **What it shows:** The central innovation of Chimera, demonstrating advanced algorithm design for distributed optimization.
    *   **How we did it:** By implementing the complex logic described in the paper to generate the bidirectional execution pattern, carefully managing dependencies between micro-batches and stages.

#### `chimera/models/`

*   **`bert48.py`**
    *   **Purpose:** Provides an example of how to adapt a BERT model (specifically, a 48-layer variant) for Chimera's pipeline parallelism.
    *   **Working:** Defines `BertStage` as a modular unit of the BERT model, suitable for partitioning. `BertForPipelineParallelism` then assembles these stages, handling inter-stage data transfer.
    *   **Inputs:** BERT model configuration (e.g., number of layers, hidden size).
    *   **Outputs:** A PyTorch `nn.Module` structured into pipeline stages, ready for distributed execution.
    *   **What it shows:** Practical application of model partitioning to a real-world, complex architecture.
    *   **How we did it:** By re-implementing or wrapping parts of a standard BERT model into `nn.Module` sub-classes that represent pipeline stages.

*   **`gpt2_64.py`**
    *   **Purpose:** Similar to `bert48.py`, this file demonstrates the adaptation of a GPT-2 model (specifically, a 64-layer variant) for Chimera's pipeline parallelism.
    *   **Working:** Defines `GPT2Stage` and `GPT2ForPipelineParallelism` to enable the GPT-2 model to be broken down and executed across pipeline stages.
    *   **Inputs:** GPT-2 model configuration.
    *   **Outputs:** A pipeline-ready GPT-2 model.
    *   **What it shows:** Versatility of the Chimera framework across different large language model architectures.
    *   **How we did it:** By applying the same principles of model decomposition as in `bert48.py` to the GPT-2 architecture.

#### `chimera/runners/`

*   **`benchmarks.py`**
    *   **Purpose:** Contains scripts and utilities for systematically evaluating the performance of Chimera under various conditions.
    *   **Working:** Sets up different distributed training scenarios, runs training loops, and collects performance metrics (e.g., throughput, memory usage, GPU utilization).
    *   **Inputs:** Model configurations, hardware setups, number of GPUs.
    *   **Outputs:** Detailed performance logs and aggregated metrics.
    *   **What it shows:** Rigorous evaluation and validation of the Chimera system's efficiency.
    *   **How we did it:** By orchestrating distributed training runs and integrating profiling tools.

*   **`train.py`**
    *   **Purpose:** The main entry point for executing distributed training using the Chimera framework.
    *   **Working:** The `ChimeraTrainer` class initializes the distributed environment, loads the partitioned model stage assigned to the current GPU, sets up the communication, and executes the training loop according to the generated schedule. Each GPU runs an instance of this script.
    *   **Inputs:** Command-line arguments for model configuration, parallelization strategy, device ID, and distributed environment parameters.
    *   **Outputs:** A trained model, performance logs, and potentially checkpoints.
    *   **What it shows:** The end-to-end execution of a distributed training job using Chimera.
    *   **How we did it:** By integrating all other Chimera components (model stages, schedule, runtime communication) into a cohesive distributed training loop.

### `tests/` directory

*   **Purpose:** Contains a suite of tests to ensure the correctness, reliability, and performance of the Chimera library.
*   **Working:** Includes unit tests for individual components (e.g., `partition.py`, `schedule.py`), integration tests to verify the interaction between modules, and potentially validation tests against expected outputs or performance benchmarks.
*   **Inputs:** Test cases, mock data.
*   **Outputs:** Test results (pass/fail), coverage reports.
*   **What it shows:** Commitment to code quality, robustness, and adherence to expected behavior.
*   **How we did it:** By writing comprehensive test cases using a testing framework (e.g., `pytest`, `unittest`).

## Getting Started: How to Run the Project

To set up and run the Chimera project locally:

### Prerequisites

*   Python 3.8+
*   PyTorch (with CUDA support for GPU acceleration)
*   CUDA Toolkit (compatible with your PyTorch installation)
*   `pyyaml`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Chimera-Efficiently-Training-Large-Scale-Neural-Networks-with-Bidirectional-Pipelines.git
    cd Chimera-Efficiently-Training-Large-Scale-Neural-Networks-with-Bidirectional-Pipelines
    ```
    *(Note: Replace `https://github.com/your-username/Chimera-Efficiently-Training-Large-Scale-Neural-Networks-with-Bidirectional-Pipelines.git` with the actual URL of your repository once it's hosted.)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running Demos

Explore the project's capabilities with the provided demo scripts:

*   **Interactive Playground:**
    ```bash
    python chimera_playground.py
    ```
    This script offers an interactive tour of Chimera's scheduling, partitioning, performance estimation, and autotuning features.

*   **Performance Demonstration:**
    ```bash
    python demo_performance.py
    ```
    Run this to see performance benchmarks.

*   **Schedule Visualization:**
    ```bash
    python demo_schedule.py
    ```
    Visualize how the bidirectional pipeline schedule works.

*(Note: For actual distributed training, you would typically use `torch.distributed.launch` or `torchrun` with `chimera/runners/train.py`, configured for your specific hardware setup.)*

## Contribution

We welcome contributions to enhance Chimera's capabilities, extend model support, or improve its efficiency. Please refer to the `CONTRIBUTING.md` (if available) for guidelines.

## License

[License Information - e.g., MIT License]
