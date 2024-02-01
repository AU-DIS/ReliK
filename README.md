# ReliK

ReliK is a measure to capture reliability in Knowledge Graph Embeddings (KGE) in a local neighborhood.<br>
A Knowledge graph is considered a list of triples in the structure (head, relation, tail)

> e.g. ("Leonardo da Vinci", "painted", "Mona Lisa")

## Running the implementation of ReliK

The code is in python and located in the "approach" folder. The requirements are noted in `requirements.txt` and can be installed with that help.<br>

To run the code:

> python experiment_controller.py -d [dataset] -e [embedding] -t [task_list] -s [size_subgraphs] -n [nmb_subgraphs] -heur [heuristic] -r [sample_size] -c [classifier_type]

The task_list schould be seperated by ``,``.<br>

To run the ReliK calculations for the ``CodexSmall`` dataset with the `TransE` embedding would look like:
> python experiment_controller.py -d CodexSmall -e TransE -t siblings -s 60 -n 100 -heur binomial -r 0.1

If there is no pretrained embedding that fits the dataset and type, it will be trained before continuing with the experiment. You can also pretrain the embedding with:

> python experiment_controller.py -d [dataset] -e [embedding] -st

If additional info is needed

> python experiment_controller -h

provides more info, how to run experiments.

## Results

DataSplits for reproducability will be stored in a `KFold` folder.<br>

The trained embeddings will be stored in a `trainedEmbeddings` folder.<br>

The resulting data will be stored in a `scoreData` folder.<br>

## Further information

If there are any questions about this, feel free to contact: `maximilian.egger[at]cs.au.dk`