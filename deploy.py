import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import boto3

# If this throws an error, replace with your actual SageMaker role ARN
role = 'arn:aws:iam::711387109323:role/sagemaker-execution-role-shubhan'

model_uri = "s3://sagemaker-models-shubhan-1/models/svm_model.tar.gz"

sklearn_model = SKLearnModel(
    model_data=model_uri,
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    py_version="py3",
)

predictor = sklearn_model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1
)
