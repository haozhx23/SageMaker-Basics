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

<br />

### SageMaker Core API Sample (Based on Notebook)
<br />

1. Notebook Instance 仅做算力资源的调用，交互式coding，下载HF模型参数/训练数据推送S3等。只需要python环境+SageMaker SDK，本身不运行训练任务

2. 根据任务的算力/显存需要，按需配置```instance_type``` 、```instance_count```参数

3. 任务（如模型训练）所需的环境，不需要在算力机上安装conda环境等。```image_uri``` + ```requirement.txt```基本可以满足

4. API调用后，会把指定的code、model parameter、data一起拷贝到资源机（全托管）。任务结束后把训练好的model parameter拷回s3进行持久化

5. 大模型场景，一般为了加速 资源机<-->s3 的拷贝过程（避免拷贝操作占用算力机），需要的额外操作详见Large-model-training-notice

<br />

```python
from sagemaker.estimator import Estimator

instance_count = 2
envs = {
            'NODE_NUMBER':str(instance_count),
            'MODEL_S3_BUCKET': sagemaker_default_bucket
}

# 基础镜像，已经集成大部分依赖（注意us-east-1需要切换实际区域如us-west-2等）
# 其他依赖，可以在source_dir中的requirements.txt中，以文本形式指定
image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04'


'''
entry_point - 入口脚本，兼容.py/.sh

source_dir - 上传至训练机/opt/ml/code路径的内容，需要包括entry_point。改路径下存在的requirement.txt会自动执行。或整体改用dependency参数，详情参考API文档

base_job_name - Estimator API会追加时间戳等标记保证全局job_name唯一性。或直接在fit()中指定

max_run - Large Model场景如果预计到任务时间较长，需要按需调整。初始的limit 5天，可以提ticket提升至28天

keep_alive_period_in_seconds - SageMaker的warm pool https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html。需要根据机型，提升limit
'''

# 有其他的Estimator形式。底层都是基于docker，没有本质区别。
est = Estimator(role=role,
                      entry_point='run_train.py',
                      source_dir='./',
                      base_job_name='some-job-name',
                      instance_count=instance_count,
                      instance_type='ml.p4de.24xlarge',
                      image_uri=image_uri,
                      environment=envs,
                      # hyperparameters=hyps, # 如果不需要env，可以用hyper params带入所需变量
                      max_run=3600*24*2, # 训练任务存续的时间上限
                      keep_alive_period_in_seconds=3600,
                      disable_profiler=True,
                      debugger_hook_config=False)


## data channel
## 训练数据在S3的路径
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

