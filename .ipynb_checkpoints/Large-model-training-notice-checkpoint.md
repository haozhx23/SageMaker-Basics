## SageMaker LM Training



#### Developer Guide

https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html




#### API文档

Estimator - https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html

HF Estimator - https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html#hugging-face-estimator

TrochEstimator - https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html



#### Sample Code

Sagemaker官方examples - https://github.com/aws/amazon-sagemaker-examples

- 主要关注training/inference/

AWS Samples - https://github.com/aws-samples

SSH helper - https://github.com/aws-samples/sagemaker-ssh-helper

FlanT5 - https://github.com/yuhuiaws/DeepSpeed-training-LLM-on-SageMaker-for-multiple-nodes

- Deepspeed多机多卡Sample

Alpaca / Vicuna - https://github.com/snowolf/alpaca-on-amazon-sagemaker

- 包含Torchrun & Docker build Sample



#### Docker相关

Sagemaker预制的镜像列表

- https://github.com/aws/deep-learning-containers/blob/master/available_images.md 
- https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-us-east-1.html

自建镜像需要的SM Training Toolkit - https://github.com/aws/sagemaker-training-toolkit

- 使用自定义镜像时，Dockerfile中仅需要额外增加```RUN pip3 install sagemaker-training```

Docker Container in SM - https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html

Self-build docker - https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step5

- 包含非默认名称的权限配置

其他

- 扩展SageMaker镜像 - https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.html

- SageMaker内置镜像Source file参考 - https://github.com/aws/deep-learning-containers/tree/master/pytorch/training/docker
- SageMaker内置镜像提供的env列表 - https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
- 如果用notebook instance构建镜像，需要提前把docker的存储路径修改到/ec2-user/SagaMaker/some_temp_path，来使用外挂存储。避免空间不足。&重启docker。



#### SageMaker API Sample

```python
import time
from sagemaker.estimator import Estimator

instance_count = 2
envs = {
            'NODE_NUMBER':str(instance_count),
            'MODEL_S3_BUCKET': sagemaker_default_bucket
}

image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04 '


'''
entry_point - 入口脚本，兼容.py/.sh

source_dir - 上传至训练机/opt/ml/code路径的内容，需要包括entry_point。改路径下存在的requirement.txt会自动执行。或整体改用dependency参数，详情参考API文档

base_job_name - Estimator API会追加时间戳等标记保证全局job_name唯一性。或直接在fit()中指定

max_run - Large Model场景如果预计到任务时间较长，需要按需调整。初始的limit 5天，可以提ticket提升至28天

keep_alive_period_in_seconds - SageMaker的warm pool https://docs.aws.amazon.com/sagemaker/latest/dg/train-warm-pools.html。需要根据机型，提升limit
'''


est = Estimator(role=role,
                      entry_point='run_train.py',
                      source_dir='./',
                      base_job_name='some-job-name',
                      instance_count=instance_count,
                      instance_type='ml.p4d.24xlarge',
                      image_uri=image_uri,
                      environment=envs,
                      # hyperparameters=hyps, # 如果不需要env，可以用hyper params带入所需变量
                      max_run=3600*24*2, # 训练任务存续的时间上限
                      keep_alive_period_in_seconds=3600,
                      disable_profiler=True,
                      debugger_hook_config=False)


## data channel
datachnl = {'train123':'s3://some-bucket-name/datasets/data-path-train/',
           'val123':'s3://some-bucket-name/datasets/data-path-val/'}

est.fit(datachnl)
```



#### 入口脚本Sample

```shell
#!/bin/bash

# 1 - 模型参数文件，从s3拷贝到资源机	
	# s5cmd等效于aws cli的aws s3 cp命令，速度更快
	# （大模型以外的场景不需要该操作）
chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/some-large-model/pretrain/* /tmp/large_model_pretrain/

# 2 - 训练数据从s3拷贝到资源机
	# （不需要操作，Sagemaker会自动拷贝到资源机的默认路径/opt/ml/input/data/）
	# e.g. /opt/ml/input/data/train123，这里train123跟Sagemaker Estimator fit() 传入的channel中的key名称一致


# 3 - 代码及脚本
	# （不需要操作，Sagemaker Estimator参数中指定的source_dir/dependency等，会自动上传到资源机的默认路径/opt/ml/code）


#torchrun
#python -m torch.distributed.run
deepspeed --num_gpus=8 /opt/ml/code/model_file/train.py \
    --deepspeed ds.json \
    --model_name_or_path /tmp/large_model_pretrain/ \
    --data_path /opt/ml/input/data/train123/sample_dataset.json \
    --output_dir /tmp/large_model_out \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --cache_dir /tmp \


# ************************************
# 4 - 训练后的模型参数，从资源机拷贝到S3
	# （*注意LLM场景，务必不能用/opt/ml/model作为模型的输出路径，否则Sagemaker会执行 model file -> tar --s3 cp--> s3
	# 会有小时级别的时间消耗
chmod +x ./s5cmd
./s5cmd sync /tmp/large_model_out s3://$MODEL_S3_BUCKET/ds-llama/output/$(date +%Y-%m-%d-%H-%M-%S)/
```



#### 训练过程中的存储原理（参考）

包括SM默认使用的路径、自动tar、tmp路径等

https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html

https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html

https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html

<u>***Note：仅做原理参考。LM场景因为模型太大，务必走上述Sample Code中的形式，直接用s5cmd对参数文件、checkpoint等进行控制</u>



#### 需要注意的点

如果在SageMaker Console手动停止任务后，长时间任务状态没有变化或者停止，需要快速开ticket，由后台engineer操作强停。


