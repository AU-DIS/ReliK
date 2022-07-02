#  Knowledge Graphs - Embeddings & Rules (KGE&R)
Playing with embedding and rule mining in knowledge graphs

### Data - KGs
* [AMIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/)
    * e.g., [YAGO2 sample](http://resources.mpi-inf.mpg.de/yago-naga/amie/data/yago2_sample/yago2core.10kseedsSample.compressed.notypes.tsv)
* [KGDatasets](https://github.com/ZhenfengLei/KGDatasets) (used, among others, in [PyKEEN](https://github.com/pykeen/pykeen))

### Data - Rules
* [AMIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/)
* [AMIE](https://github.com/idirlab/kgcompletion/blob/master/AMIE/AMIEs-rules.zip) (from [KG-Completion Re-evaluation](https://github.com/idirlab/kgcompletion))

### Frameworks
* AMIE [[paper](http://resources.mpi-inf.mpg.de/yago-naga/amie/amie.pdf)] [[github](https://github.com/lajus/amie)] [[web](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/)]
* PyKEEN [[paper](https://arxiv.org/abs/2006.13365)] [[github](https://github.com/pykeen/pykeen)] [[web](https://pykeen.readthedocs.io/en/latest/index.html)]
* KG-Completion Re-evaluation [[paper](https://arxiv.org/abs/2003.08001)] [[github](https://github.com/idirlab/kgcompletion)] [[web](https://medium.com/@fakrami/re-evaluation-of-knowledge-graph-completion-methods-7dfe2e981a77)]
* OpenKE [[paper](https://www.aclweb.org/anthology/D18-2024/)] [[github](https://github.com/thunlp/OpenKE)] [[web](http://openke.thunlp.org/)]
* LibKGE [[paper](https://www.aclweb.org/anthology/2020.emnlp-demos.22/)] [[github](https://github.com/uma-pi1/kge)]
* DGL-KE [[paper](https://arxiv.org/abs/2004.08532)] [[github](https://github.com/awslabs/dgl-ke)]

### Training embeddings
* Done with [OpenKE](http://openke.thunlp.org)
* Install OpenKE by following its [installation guide](https://github.com/thunlp/OpenKE) (it requires [PyTorch](https://pytorch.org/), among others)
* See the `Data` section of the [OpenKE guide](https://github.com/thunlp/OpenKE) for the training-data format, and see `src/generate_openke_data.py` to generate it
* See `src/openke/train_transe_yago2sample.py` for an example on how to train the well-known [TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) embedding model
