# Evidence-based Trustworthiness

This is the code and the data for paper *“Evidence-based Trustworthiness"*.

## Dataset:
We use Emergent (Ferreira et al., 2016) as the corpus to evaluate our models. We keep a copy under folder dataset/.

## Pre-processing:
Our solution can incoporate any off-the-shelf textual entailment model as a part of the global inference framework as a way to link the evidence to the claim.

Here, we leverage the model ''https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz'' released by allennlp.

To generate the prediction that if the evidence supports or contradicts the claim, you can first run ```python entailment.py```.

Its output will be later fed to our trustworthiness algorithm.

Note that to achieve the best performance, you should fine-tune the textual entailment first.

## Solution:
To run our trustworthiness algorithm and generate the evaluation metrics, you can run ```python method.py```.