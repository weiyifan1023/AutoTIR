#mkdir -p hotpotqa
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/train.jsonl -O hotpotqa/train.jsonl
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/dev.jsonl -O hotpotqa/dev.jsonl
#
#mkdir -p 2wikimultihopqa
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl -O 2wikimultihopqa/train.jsonl
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/dev.jsonl -O 2wikimultihopqa/dev.jsonl
#
#mkdir -p musique
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl -O musique/train.jsonl
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/dev.jsonl -O musique/dev.jsonl
#
#mkdir -p bamboogle
#wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl -O bamboogle/test.jsonl



#  https://huggingface.co ===> https://hf-mirror.com 使用镜像进行下载
mkdir -p hotpotqa
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/train.jsonl -O hotpotqa/train.jsonl
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/dev.jsonl -O hotpotqa/dev.jsonl

# 创建目录并下载 2wikimultihopqa 数据集
mkdir -p 2wikimultihopqa
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl -O 2wikimultihopqa/train.jsonl
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/dev.jsonl -O 2wikimultihopqa/dev.jsonl

# 创建目录并下载 musique 数据集
mkdir -p musique
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl -O musique/train.jsonl
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/dev.jsonl -O musique/dev.jsonl

# 创建目录并下载 bamboogle 数据集
mkdir -p bamboogle
wget https://hf-mirror.com/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl -O bamboogle/test.jsonl