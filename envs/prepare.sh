conda create -n wreckognizer python=3.8
echo "conda activate wreckognizer" >> ~/.bashrc

pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.10.0
pip install pandas==1.1.3
pip install pyarrow==0.17.1
pip install pymongo==3.7.2
pip install boto3==1.17.101
pip install PyYAML==5.4.1
pip install scikit-learn==0.24.2
pip install scipy==1.4.1
pip install clearml
pip install black
pip install plotly==5.3.0
pip install plotly-express==0.4.0
pip install p_tqdm