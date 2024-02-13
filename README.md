## Jupyter notebooks

- gpt2_base.ipynb :  Contains code for gpt2 finetuning and evaluation
- gpt2distill_base.ipynb : Contains code for knowledge distillation through student teacher training
- sagemaker.ipynb : Contains code used in sagemaker to deploy endpoint as well as invokation example

## Modular framework

I have also converted the base finetuning notebook into modular framework which can be run from terminal

For training:

`python main.py  --data_file "title.akas.tsv.gz" --task "train" --train_size 20000`

For testing:

`python main.py  --data_file "title.akas.tsv.gz" --task "test" --train_model_path "model/" --test_size 1000`

