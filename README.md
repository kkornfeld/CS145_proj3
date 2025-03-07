# Retrieval-Augmented Generation Model

Kyan Kornfeld and Quinn Hilger

## Pipeline Setup

To begin, create and activate the Python environment:

```
conda create -n rag python=3.10
conda activate rag
```

Next, install the required packages:
```
pip install -r requirements.txt
pip install --upgrade openai
```

Then, gain access to the meta-llama/Llama-3.2-3B-Instruct model on Hugging Face. Log in to Hugging Face using the token from the account that has access:
```
huggingface-cli login--token "your_access_token"
```

Next, download the dataset from [Meta KDD Cup 2024 Dataset](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files). Place the downloaded files in the data directory.

## Execution

Before running the RAG model, you first need to set up the VLLM server:

```
vllm serve meta-llama/Llama-3.2-3B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager --max_model_len=16384
```

Then, generate the predictions:
```
python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --split 1 --model_name "rag_baseline" --llm_name "meta-llama/Llama-3.2-3B-Instruct" --is_server --vllm_server "http://localhost:8088/v1"
```

Once the predictions are generated, evaluate them:
```
python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "rag_baseline" --llm_name "meta-llama/Llama-3.2-3B-Instruct" --is_server --vllm_server "http://localhost:8088/v1" --max_retries 10
```

The output of the evaluation script will be the final score and statistics.