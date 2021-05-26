# CNN for Stock Market Prediction

Predict the Stock price will go up or not at the next candlestick

## Usage

### Prepare Dataset
~~~
python preprocess_data.py {windows} {dimension} {testing/training}
python generatedata.py {root} {original_dir} {destination_dir} {testing/training}
~~~
ex
~~~
python preprocess_data.py 120 50 training
python generatedata.py dataset 20_50 dataset_20_50
~~~

### Remove alpha channel
~~~
cd /dataset/dataset_20_50
find . -name "*.png" -exec convert "{}" -alpha off "{}" \;
~~~

### Training
~~~
python CNN.py -i {datasetdir} -e {epoch} -d {dimension} -b {batchsize} -o {result_report}
~~~
ex
~~~
python CNN.py -i dataset/dataset_20_50 -e 50 -d 50 -b 8 -o 20_50_result.txt
~~~


