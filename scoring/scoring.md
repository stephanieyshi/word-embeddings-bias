# Evaluation Metrics
There are several evaluation metrics, which tend to relate to vector distance, that have been used in literature to evaluate the results of debiasing. Since there is literature that shows that existing attempts to debias word embeddings have been insufficient, we believe it is best to use a variety of evaluation metrics to compare the results that have been done so far.

Note: the scoring file cannot be run separately without a baseline model since there is no "gold standard". To run the baseline, enter `python3 simple-baseline.py` in the command line.

## Primary Metrics: Projection
### Direct Bias
We first aim to measure direct bias, as described in Man is to Computer Programmer (Bolukbasi et al. 2016). They use the following metric:

<img src="https://latex.codecogs.com/gif.latex?\text{DirectBias}_c&space;=&space;\frac{1}{|N|}&space;sum_{w&space;in&space;N}&space;|cos(hat{w},&space;g)|^cs" />

where <img src="https://latex.codecogs.com/gif.latex?c" /> is a strictness parameter, where <img src="https://latex.codecogs.com/gif.latex?g" />  is a learned gender direction, and where <img src="https://latex.codecogs.com/gif.latex?N" />  is the set of gender neutral words. A higher value indicates more bias, while a lower value indicates less indirect bias.

### Indirect Bias
Another metric of bias used in Bolukbasi et al. 2016 is indirect bias, which is measured as follows:

<img src="https://latex.codecogs.com/gif.latex?\beta(w,&space;v)&space;=&space;\frac{\left(&space;w&space;\cdot&space;v&space;-&space;\frac{w_\perp&space;\cdot&space;v_\perp}{\left\lVert&space;w_\perp\right\rVert_2&space;\left\lVert&space;v_\perp\right\rVert_2}\right&space;)&space;}{w&space;\cdot&space;v}"/>

where <img src="https://latex.codecogs.com/gif.latex?w_g&space;=&space;(w&space;\cdot&space;g)&space;g" /> and <img src="https://latex.codecogs.com/gif.latex?w_\perp&space;=&space;w&space;-&space;w_g" />. A higher magnitude indicates more bias, while a lower magnitude indicates less indirect bias.

## Secondary Metric: Correlation
### Pearson Correlation
In Lipstick on a Pig (Gonen et al. 2019), Pearson correlation is used to measure the similarity in biases between embeddings before debiasing and embeddings after debiasing.  It is defined as:

<img src="https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2012/10/pearson.gif" />
