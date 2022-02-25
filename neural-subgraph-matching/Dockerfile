# syntax=docker/dockerfile:1

FROM python:3.6

WORKDIR /app

RUN python3 -m pip install -U matplotlib==2.1.1 && \
    pip3 install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install torch-scatter==2.0.4 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html && \
    pip3 install torch-sparse==0.6.1 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html && \
    pip3 install torch-cluster==1.5.4 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html && \
    pip3 install torch-geometric==1.4.3 -f https://data.pyg.org/whl/torch-1.4.0+cpu.html && \
    pip3 install deepsnap==0.1.2 && \
    pip3 install networkx==2.4 && \
    pip3 install numpy==1.19.5 && \
    pip3 install scikit_learn==0.21.3 && \
    pip3 install scipy==1.1.0 && \
    pip3 install seaborn==0.9.0 && \
    pip3 install test_tube==0.7.5 && \
    pip3 install tqdm==4.43.0 && \
    pip3 install wandb

COPY common/ common/
COPY subgraph_matching/ subgraph_matching/
COPY experiments.py experiments.py

CMD [ "python3", "experiments.py"]