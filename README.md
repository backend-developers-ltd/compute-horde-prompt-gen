# Compute Horde Prompt Gen
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Script to generate batches of random unique prompts to be used in the Compute Horde project synthetic jobs.

The prompt that generates prompts is inspired from [Bittensor Subnet 18 (Cortex. t)] (https://github.com/Datura-ai/cortex.t/blob/276cfcf742e8b442500435a1c1862ac4dffa9e20/cortext/utils.py#L193) (licensed under the MIT License.)

The generated prompts will be saved in `<output_folder_path>/prompts_<uuid>.txt`, each line of the text file containing a prompt.

supports llama3 (`meta-llama/Meta-Llama-3.1-8B-Instruct`) and phi3 (`microsoft/Phi-3.5-mini-instruct`) models

### build image 


```bash
cd src/compute_horde_prompt_gen

# download model data
python3 download_model.py --model_name phi3 --huggingface_token <API_KEY>

# build the image
docker build -t compute-horde-prompt-gen .
```


### run image
```bash
docker run -v ./output/:/app/output/ compute-horde-prompt-gen --model_name phi3 --number_of_prompts_per_batch 4 --uuids uuid1,uuid2,uuid3
```

### testint
```bash
cd src/compute_horde_prompt_gen
python3 run.py --model_name mock --number_of_prompts_per_batch 4 --uuids uuid1,uuid2,uuid3
```

---

## License
This repository is licensed under the MIT License.
