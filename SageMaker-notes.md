## SageMaker Hands-on (pre LLM Training)

<br />

### SageMaker

#### Access Resources on SageMaker
![image-20230427132700003](assets/image-20230427132700003.png)
<br />


#### SageMaker Managed Resources
![image-20230427132700004](https://sagemaker-workshop.com/images/sm-containers.gif)
<br />
* https://sagemaker-workshop.com/custom/containers.html
<br />

<br />

### Hands-on Workshop

#### Preparation
1 - 在notebook console中cd到SageMaker文件路径：/home/ec2-user/SageMaker/
<br />
2 - git clone https://github.com/aws-samples/amazon-sagemaker-immersion-day
<br />
3 - 基于Workshop Lab-3b
https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/en-US/lab3/option1-b
<br />
中间跳过Launch the notebook instance

<br />

### Developer Guide

https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html  

<br />

### API文档

Estimator - https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

HF Estimator - https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-estimator

TrochEstimator - https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html  

<br />

```python
import time
from sagemaker.estimator import Estimator

instance_count = 2
envs = {
            'NODE_NUMBER':str(instance_count),
            'MODEL_S3_BUCKET': sagemaker_default_bucket
}

image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04 '


est = Estimator(role=role,
                      entry_point='run_train.py',
                      source_dir='./',
                      base_job_name='some-job-name',
                      instance_count=instance_count,
                      instance_type='ml.p4de.24xlarge',
                      image_uri=image_uri,
                      environment=envs,
                      max_run=3600*24*2, # 训练任务存续的时间上限
                      keep_alive_period_in_seconds=3600, # warm pool
               )


## data channel
data_channel = {'train123':'s3://some-bucket-name/datasets/data-path-train/',
           'val123':'s3://some-bucket-name/datasets/data-path-val/'}

est.fit(data_channel)

'''
# HuggingFace Estimator
huggingface_estimator = HuggingFace(
                            entry_point          = 'start.py',        
                            source_dir           = 'src',             
                            instance_type        = 'ml.p4de.24xlarge', 
                            instance_count       = instance_count,
                            base_job_name        = job_name,      
                            role                 = role,           
                            transformers_version = '4.17',        
                            pytorch_version      = '1.10',        
                            py_version           = 'py38',
                            environment = environment,
                        )

# Pytorch Estimator
pytorch_estimator = PyTorch(
                        entry_point="start.py",
                        source_dir= 'src',
                        role=role,
                        py_version="py38",
                        framework_version="1.11.0",
                        instance_count=2,
                        instance_type="ml.c5.2xlarge",
                        hyperparameters={"epochs": 1, "backend": "gloo"},
                    )
'''
```


### Sample Code

Sagemaker官方examples - https://github.com/aws/amazon-sagemaker-examples


<br />

### Refs
Train 175+ billion parameter NLP models with model parallel additions and Hugging Face on Amazon SageMaker https://aws.amazon.com/cn/blogs/machine-learning/train-175-billion-parameter-nlp-models-with-model-parallel-additions-and-hugging-face-on-amazon-sagemaker/
<br />
Training large language models on Amazon SageMaker: Best practices https://aws.amazon.com/cn/blogs/machine-learning/training-large-language-models-on-amazon-sagemaker-best-practices/
<br />
Deploy BLOOM-176B and OPT-30B on Amazon SageMaker with large model inference Deep Learning Containers and DeepSpeed https://aws.amazon.com/cn/blogs/machine-learning/deploy-bloom-176b-and-opt-30b-on-amazon-sagemaker-with-large-model-inference-deep-learning-containers-and-deepspeed/
<br />
Deploy large models on Amazon SageMaker using DJLServing and DeepSpeed model parallel inference https://aws.amazon.com/cn/blogs/machine-learning/deploy-large-models-on-amazon-sagemaker-using-djlserving-and-deepspeed-model-parallel-inference/
https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-large-model-inference.html
<br />
Distributed Training:
https://github.com/aws/amazon-sagemaker-examples/tree/main/training/distributed_training/pytorch/model_parallel
https://github.com/aws-samples/sagemaker-distributed-training-workshop
<br />
LLM model hosting https://github.com/aws-samples/sagemaker-hosting/tree/main/Large-Language-Model-Hosting
<br />
Bloom 176B https://github.com/aws/amazon-sagemaker-examples/tree/main/inference/nlp/realtime/llm/bloom_176b
<br />
GPT-J  https://github.com/aws-samples/sagemaker-hosting/blob/main/Large-Language-Model-Hosting/LLM-Deployment-SageMaker/intro_to_llm_deployment.ipynb
<br />
GPT-Neo-X https://github.com/aws-samples/sagemaker-hosting/blob/main/Large-Language-Model-Hosting/Optimize-LLM/djl_accelerate_deploy_g5_12x_GPT_NeoX.ipynb
<br />