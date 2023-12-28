### This code was used for QA model development described in the paper titled "Question-Answering System Extracts Information on Injection Drug Use from Clinical Notes".


The `src` folder contains 2 sub-folders: `modeling` and `data_prep`

* The `modeling` sub-folder contains codes for tokenizing QA samples, finetuning and testing the QA model
* The `data_prep` sub-folder contains codes for cleaning the notes, an example of note enrichment, an example of automated gold-standard answer extraction, an example of parsing rules, and an example of question-to-answer mapping.

The `plot_data` folder contains data tables used for the error plots in the paper.

The `environment.yml` file has all the necesary packages to create the Conda environment.
