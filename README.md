### qa-system-for-injection-drug-use

The `src` folder contains 2 sub-folders: `modeling` and `data_prep`

* The `modeling` sub-folder contains codes for tokenizing QA samples, finetuning and testing the QA model
* The `data_prep` sub-folder contains codes for cleaning the notes, an example of automated gold-standard answer extraction, and an example of parsing rules.

The `plot_data` folder contains data tables used for the error plots in the paper.

The `environment.yml` file has all the necesary packages to create the Conda environment.


`The codes for QA modeling is heavily adapted from the (huggingface example on question-answering task)[https://github.com/huggingface/transformers/blob/master/examples/legacy/question-answering/run_squad.py] with some modifications`
