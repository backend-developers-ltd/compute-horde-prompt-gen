# Compute Horde Prompt Gen
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Script to generate batches of prompts for the Compute Horde project synthetic jobs.
The prompt that generates prompts is inspired from [Bittensor Subnet 18 (Cortex. t)] (https://github.com/Datura-ai/cortex.t/blob/main/cortext/utils.py#L139)


### build image 


```bash
# download the model data from huggingface
python3 download_model.py --hugging face_api_key <API_KEY>

cd src/compute_horde_prompt_gen
docker build -t compute-horde-prompt-gen .
```


### run image
```bash
docker run -v ./output/:/app/output/ compute-horde-prompt-gen --dynamic_number_of_batches_in_a_single_go 3 --dynamic_number_of_prompts_in_a_batch 4 --uui uuid1,uuid2,uuid3
```

### testint
```bash
python3 run.py --mock_model --dynamic_number_of_batches_in_a_single_go 3 --dynamic_number_of_prompts_in_a_batch 4 --uui uuid1,uuid2,uuid3
```

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

