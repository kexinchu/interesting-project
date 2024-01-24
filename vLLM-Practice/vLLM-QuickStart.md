# vLLM Source Code
- Link: https://github.com/vllm-project/vllm
- Edition: v0.2.3
  - The hardware environment is A10, only support cuda 11.2
  - based Python 3.10

### Sample
```python
# 可参考: examples/offline_inference.py
llm = LLM(model='/model_path')
sampling_params = SamplingParams(temperature=0.95, top_p=0.7, top_k=40, max_tokens=2048)
outputs = llm.generate(prompts, sampling_params)
```

### LLM initiation(模型加载)
- Inferface: vllm/entrypoints/llm.py
```python
class LLM:
    # 初始化, 传入model, tokenizer, quantization等入参 声明engine对象, 用于实际计算
    def __init__(model, ...):
        # engine_args 入参信息
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        # engine: vllm/engine/llm_engine.py
    # 入参信息
    """
    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
    """
```

- LLMEngine: vllm/engine/llm_engine.py
```python
# An LLM engine that receives requests and generates texts.
class LLMEngine:
    # 初始化
    def __init__(self, model_config, ... ...):
        """实例化"""
        self.tokenizer = get_tokenizer(...)  # from config
        # Create the parallel GPU workers
        # Ray - 分布式计算框架
        if self.parallel_config.worker_use_ray:
            # allowing Worker to be lazliy initialized after Ray sets CUDA_VISIBLE_DEVICES
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
        # Profile the memory usage and initialize the cache.
        # 计算可以分配的GPU、CPU块数，初始化分配kv_cache的cache_engine
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config)

    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """
        Creates an LLM engine from the engine arguments."
        Note:
            在Python中，cls代表的事类本身，self对应的则是类的一个实例对象；
            =>
                因为cls等同于类本身，类方法中可以通过使用cls来实例化一个对象
        """
        ... ...
        # 返回一个engine对象
        return cls(*engine_configs, ...)
    
    # 使用 1 个GPU - 1 workers
    def _init_workers():
        # Lazy import the Worker to avoid importing torch.cuda/xformers before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker
        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,                      # 固定GPU 0
            distributed_init_method,
        )
        self.workers.append(worker)
        # 初始化model
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
        # 加载 weights 参数
        self._run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

    def _init_cache():
        # Profiles the memory usage and initializes the KV cache.
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        # 运行一次输入负载最大（输入长度为模型最大上下文长度）的模型的前向传播（开启topk、topp），计算剩余内存、显存（可存放kvcache）可划分块数
        num_blocks = Worker.profile_num_available_blocks(
            block_size,
            gpu_memory_utilization,
            cpu_swap_space
        )
        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        self.cache_config.num_gpu_blocks = min(b[0] for b in num_blocks)
        self.cache_config.num_cpu_blocks = min(b[1] for b in num_blocks)
       
        # Initialize the cache.
        Worker.init_cache_engine(cache_config=self.cache_config)
```

- Worker: vllm/worker/worker.py
```python
# A worker class that executes (a partition of) the model on a GPU.
class Worker:
    def __init__(self, ...):
        # vllm/worker/model_runner.py
        # 实际计算模块
        self.model_runner = ModelRunner(model_config, parallel_config, scheduler_config)
    
    def profile_num_available_blocks():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 计算一次前向传播: execute model with max num sequences and the total number of tokens equal to max_num_batched_tokens dummy input data 0
        self.model_runner.profile_run()

        # GPU 和 CPU上的block块数计算
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        
        # release the GPU memory
        torch.cuda.empty_cache()
    
    def init_model(self):
        # set device
        torch.cuda.set_device(self.device)

        # check dtype
        # 当前支持 fp32, fp16 和 bfp16
        _check_if_gpu_supports_dtype(self.model_config.dtype)

        # 初始化distributed environment
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)
        # 调用 torch.distributed.init_process_group(), 借助nccl通信

    def load_model(self):
        """
        model_runner.py中调用的 from vllm.model_executor import get_model
        
        self.model = get_model(self.model.config)

        生效的是：vllm/model_executor/model_loader.py
        这里有当前支持的所有model的list： _MODEL_REGISTRY

        # 实际调用pytorch中的 model.load_weights()方法
        """
        self.model_runner.load_model()
```

### LLM Inference(模型推理)
- LLM: vllm/entrypoints/llm.py
```python
class LLM:
    # 对外开放的调用接口，输入prompt，返回LLM执行结果
    def generate(self, prompts, sampling_params, prompt_token_ids, use_tqdm):
        """
        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        # 核心逻辑如下： _add_request 想queue中添加prompts
        #              _run_engine 执行llm_engine返回结果
        # 输入是多个Test Prompt
        num_requests = len(prompts) if prompts is not None else len(
            prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            token_ids = None if prompt_token_ids is None else prompt_token_ids[
                i]
            self._add_request(prompt, sampling_params, token_ids)
        # 执行 engine
        return self._run_engine(use_tqdm)

    def _add_requests():
        request_id = str(next(self.request_counter))
        # 增加requests
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)
    
    def _run_engine():
        # 核心逻辑
        outputs: List[RequestOutput] = []
        # 调用engine，查看当时是否还有未完成的requests
        # len(self.waiting) + len(self.running) + len(self.swapped)
        while self.llm_engine.has_unfinished_requests():
            # step 执行一个decoder iteration
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                # 判断哪些结束了
                if output.finished:
                    outputs.append(output)

```

- LLMEngine: vllm/engine/llm_engine.py
```python
class LLMEngine:
    # Add a request to the engine's request pool.
    def add_requests(self, ...):
        # tokenizition
        prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    # 执行一个forward iteration, 得到一个token
    # prefilling or token-generation
    def step(self) -> List[RequestOutput]:
        """
        Performs one decoding iteration and returns newly generated results.
        """
        # get SequenceGroups to be executed in the next iteration and the token blocks to be swapped in/out/copy.
        seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        if scheduler_outputs.is_empty():
            return ignored

        # Execute the model. - 执行一次forward, 获取结果
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # 更新Sequence信息
        return self._process_model_outputs(output, scheduler_outputs)
    
    def _run_workers(self, method: str, *args, ...):
        # Runs the given method on all workers. method: "execute_model"
        work_groups = [self.workers]

        # run all workers
        for workers in work_groups:
            all_outputs.extend(
                # 注意，这里的workers还是一个List
                self._run_workers_in_batch(workers, method, *args, **kwargs))
                # 此方法 为每个worker执行method方法
                """
                # getattr 从worker中获取 method 方法：
                #   load_model
                #   execute_model  # 执行inference
                executor = getattr(worker, method)

                # execute - 实际调用的是 Worker.execute_model 方法
                output = executor(*args, **kwargs)
                """
```

- Worker: vllm/worker/worker.py
```python
class Worker:
    @torch.inference_mode()
    def execute_model(
        self, 
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> SamplerOutput:
        # Issue cache operations： Swap_in, Swap_out, Copy

        # 执行execute
        output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache, cache_events)
        return output
```

- ModelRunner: vllm/worker/model_runner.py
```python
class ModelRunner:
    @torch.inference_mode()
    def execute_model(self, seq_group_metadata_list, kv_cache, cache_events):
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Prepare input tensors.
        if is_prompt:
            inputs = self._prepare_prompt(seq_group_metadata_list)
        else:
            inputs = self._prepare_decode(seq_group_metadata_list)
        input_tokens, input_positions, input_metadata = inputs

        # Execute the model forward.
        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )

        # Sample the next token. 使用抽样逻辑
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
```