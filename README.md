# Stock Returns with Option Data

In order to start the training just envoke the main script. The configurations can be altered in the config.json file.

```shell
python main.py
```

The autoencoder training can also be done seperately, without all other preprocessing steps. It takes the same configuration file as the main script.

```shell
python models/autoencoder.py
```

After completing once the prerpocessing and the initial training of the autoencoder model. Set the "preprocess" setting in the config file to "false".


Performing the t-test must be envoked seperately. Just pass the experiment log to the function:

```python
from utils.t_test import ttest

files = ["evaluation_run_2022-06-28_17-11-51_data_yearly_3", "evaluation_run_2022-06-28_17-01-01_data_monthly_3"]
for files in files:
    stats = ttest(filename=files)
    print(stats)
```

Similarily the plots can be created:

```python
from utils.plots import create_yearly_plot

files = ["evaluation_run_2022-06-28_17-11-51_data_yearly_3", "evaluation_run_2022-06-28_17-01-01_data_monthly_3"]
for files in files:
    plots(filename=files)
```