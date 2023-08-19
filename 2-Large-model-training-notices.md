## SageMaker LM Training


<br />

### Sample Code

Sagemaker官方examples - https://github.com/aws/amazon-sagemaker-examples
- 主要关注training & advanced_functionality

AWS Samples - https://github.com/aws-samples

Alpaca / Vicuna Sample
- https://github.com/snowolf/alpaca-on-amazon-sagemaker
- 包含Docker build & Fsx

Multi-node Llama with DeepSpeed or SMP
- https://github.com/yuhuiaws/finetuning-and-deploying-llama-on-Sagemaker

Multi-node Alpaca training & QLoRA
- https://github.com/haozhx23/Alpaca-Llama2-Multinode-on-SageMaker
- https://github.com/haozhx23/Llama2-QLoRA-on-SageMaker

<br />

### Docker相关

Sagemaker预制的镜像列表

- https://github.com/aws/deep-learning-containers/blob/master/available_images.md 
- https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-us-east-1.html

自建镜像需要的SM Training Toolkit
- https://github.com/aws/sagemaker-training-toolkit
- 使用自定义镜像时，Dockerfile中仅需要额外增加```RUN pip3 install sagemaker-training```

Docker Container in SM 
- https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html

Self-build docker 
- https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#byoc-training-step5
- 包含非默认名称的权限配置

其他

- 扩展SageMaker镜像 - https://sagemaker-examples.readthedocs.io/en/latest/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.html
- SageMaker内置镜像Source file参考 - https://github.com/aws/deep-learning-containers/tree/master/pytorch/training/docker
- SageMaker内置镜像提供的env列表 - https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py
- 如果用notebook instance构建镜像，需要提前把docker的存储路径修改到/ec2-user/SagaMaker/some_temp_path，来使用外挂存储。避免空间不足。&重启docker  

<br />
<br />

### 训练启动流程（以.sh为例）

```shell
#!/bin/bash

# 1 - 模型参数文件，从s3拷贝到资源机
	# s5cmd等效于aws cli的aws s3 cp命令，速度更快
	# （大模型以外的场景不需要该操作）
    # 注意destination必须为/tmp的prefix（大机型自带的NVME存储）
    # In multi-node training, only copy on LocalRank 0
chmod +x ./s5cmd
./s5cmd sync s3://$MODEL_S3_BUCKET/some-large-model/pretrain/* /tmp/large_model_pretrain/

# 2 - 训练数据从s3拷贝到资源机
	# （默认不需要操作，Sagemaker会自动拷贝到资源机的默认路径/opt/ml/input/data/）
	# e.g. /opt/ml/input/data/train123，这里train123跟Sagemaker Estimator.fit() 传入的channel中的key名称一致
    # *也可以使用1中的加速拷贝形式，或其他Steam传输形式


# 3 - 代码及脚本
	# （不需要操作，Sagemaker Estimator参数中指定的source_dir/dependency等，会自动上传到资源机的默认路径/opt/ml/code）


torchrun  --num_gpus=8 /opt/ml/code/model_file/train.py \
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
./s5cmd sync /tmp/large_model_out s3://$MODEL_S3_BUCKET/some-output-path/output/$(date +%Y-%m-%d-%H-%M-%S)/
```

* 可以在训练脚本中，使用os.system()进行s5cmd拷贝的执行，用于训练过程中的checkpoint即时地持久化至s3，长时间训练下无需处理资源机上的存储空间。

<br />

### 训练过程中的存储路径设置（参考）
<u>***Note：LLM场景因模型较大，务必走上述Sample Code中的形式，直接用s5cmd对参数文件、checkpoint等进行控制。以下仅为SageMaker默认机制中的原理参考。</u>

包括SM默认使用的路径、自动tar、tmp路径等

https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html
https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html



<br />

### 需要注意的点

如果在SageMaker Console手动停止任务后，长时间任务状态没有变化或者停止，需要快速开ticket，由后台engineer操作强停。

<br />

### Refs
SSH helper - https://github.com/aws-samples/sagemaker-ssh-helper
<br />
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