## Alignment Score with VQA

This folder contains the codes for computing the alignment score of a 3D asset
This example uses GPT-4o-mini as the VQA model. You can easily replace it to other open-source models (e.g., LLaVA)


### Quick Start
Install OpenAI
```bash
pip install openai

export OPENAI_API_KEY=[YOUR OPENAI KEY]
```

See example of how to compute the score for the sampe `a_corgi_taking_a_selfie.mp4`
in `example.ipynb`

If you want to other VLMs, replace the `get_vqa_anwer` function in the notebook by your own model.