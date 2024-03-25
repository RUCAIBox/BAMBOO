# BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities

This repository contains the evaluation code, prompt, and datasets for the paper [BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities](https://arxiv.org/abs/2309.13345).

BAMBOO benchmark is  a comprehensive benchmark to analyze LLMs’ long text modeling. In BAMBOO benchmark, there are 10 datasets from 5 tasks, i.e., question answering, hallucination detection, langauge modeling, code completion, and text sorting. Our benchmark is constructed with the following principles:

* Comprehensive Capacity Evaluation
* Avoidance of Data Contamination
* Accurate Automatic Evaluation
* Different Length Levels


## Repository Structure

- `datasets`: This directory contains the data files in the benchmark. There are 10 datasets, and each dataset contains 2 files of different lengths(4k, 16k).

- `evaluate.py`: This Python script is used to evaluate the outputs of your long text models.

- `prompt.json`: This json file contains prompts used to evaluating your long context model. 

- `requirements.txt`: Python packages to be installed for evaluating your outputs.

- `private_eval`: The directory contains evaluate code of `private_eval` datasets, which refers to [PyCodeGPT](https://github.com/microsoft/PyCodeGPT)


## Evaluation
If you obtain outputs of each datasets, you can create a python environment ``BAMBOO`` and evaluate your outputs.

1. Create and activate the conda environment named BAMBOO:
    
    ```bash
    conda create -n BAMBOO
    conda activate BAMBOO
    ```

2. Install the required Python packages by running:
    
    ```bash
    pip install -r requirements.txt
    ```
3. Run the generation script with your model. For example:

    ```bash
    python code/run_for_all.py --output_path output_path --input_path input_path --prompt_name prompt_name --prompt_path prompt_path --model_path model_path
    ```

4. Run the evaluation script with your model's output. For example:
    
    ```bash
    python evaluate.py --input_path your_file.jsonl --task task
    ```
### Evaluating a model from huggingface

Here, we provide instructions on utilizing a model from Hugging Face to generate responses to prompts in BAMBOO. We employ the model ``lmsys/vicuna-7b-v1.5-16k``, along with its template in ``fastchat``, as an illustrative example. You can execute the following script to evaluate ``altqa_4K`` with the aforementioned model:

```bash
python run_for_all.py --output_path pred_altqa_long_longchat.jsonl --input_path datasets/altqa_middle.jsonl --prompt_name altqa --prompt_path prompt.json --model_path lmsys/vicuna-7b-v1.5-16k
```

### Output format

Each data point in your jsonl file should at least contains two keys:

* **pred**: prediction of the model.
* **answer**: the right answer.

### Task Selection
Task should chosen from the list ``['meetingqa','paperqa','altqa','senhallu','abshallu','meetingpred','showspred','reportsumsort','showssort','private_eval']``


## License

This repository is released under the [MIT License](LICENSE).

## Citation

If you use this benchmark or code in your research, please consider citing the original paper:

```
@article{dong2023bamboo,
  title={BAMBOO: A Comprehensive Benchmark for Evaluating Long Text Modeling Capacities of Large Language Models},
  author={Dong, Zican and Tang, Tianyi and Li, Junyi and Zhao, Wayne Xin and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2309.13345},
  year={2023}
}
```