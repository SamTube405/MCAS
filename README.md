# MCAS
Multiplatform Cascade Simulator

# Prediction Module

source activate pnnl_socialsim<br>
python run_predictions.py --config ./metadata/configs/seed_predictions.json

## Directory Hierarchy:

```bash
ml_input/
└── domain_name
    ├── infoids_platform_name_local_topics.txt
    └── platform_name
        ├── global
        │   └── sample_global_features.pkl.gz
        ├── local
        │   └── sample_local_features.pkl.gz
        └── target
            └── total_shares
                └── sample_target_total_shares.pkl.gz

```

```bash
ml_output
└── domain_name
    └── platform_name
        └──local_topics/global_topics
            └── total_shares
                └── LSTM_{start sim period}_{end sim period}_{input/output time horizon}_{source name}
                    ├── best_retrained_model.h5.h5
                    ├── Gperformance_simulation.pkl.gz
                    ├── gt_data_simulations.pkl.gz
                    └── simulations_data.pkl.gz
```

## Config File Parameters
- VERSION_TAG: unique file name for the simulation output file.
- DOMAIN: (e.g., wh, vz, cpec)
- PLATFORM: sm platform to predict (e.g., twitter, youtube)
- PREDICTION_TYPE: (e.g., total shares, new users, old users)
- start_train_period: start date for training data
- end_train_period: end date for training data
- start_val_period: start date for validation data
- end_val_period: end date for validation data
- start_sim_period: start date for simulation
- end_sim_period: end date for simulation
- time_window -> n_in: input time horizon
- time_window -> n_out: output time horizon
- FEATURES_PATH: 
  + GLOBAL: path to global features (not per infoID). It can accept list of paths.
  + LOCAL: path to features per infoID. Can accept list of paths.
  + TARGET: path to dataframe consisting of target variable(s)
- INFORMATION_IDS: path to information IDs to predict
- EVALUATION: boolean (if true it assumes we have GT data to evaluate against, else false).
- DAILY: boolean (if true, then predictions/features are in daily granularity else it assumes weekly granularity)
- MODEL_TYPE: either local_topics (we have features per topic) or global_topics (we only have global features available)

## How to Run
1. Run volume predictor scripts
    1. python run_predictions_lstm.py --config ./metadata/configs/sample_volume_predictor_config.json
2. Generate MCAS Models (See Notebook at examples/Generate_MCAS_Models.ipynb)
3. Run Cascades (Modify configuration files correspondingly)
    1. python run_cascade_props.py --config metadata/configs/sample_cascades_file_config.json
    2. python run_cascade.py --config metadata/configs/sample_cascades_file_config.json
4. Merge cascade outputs (See Notebook at examples/Generate_Cascade_Outputs.ipynb)
5. Run new user scripts (Modify parameters within the code correspondingly)
    1. python run_newuser_replace_script.py


## Related Publications
1. Horawalavithana, Sameera, et al. "Online discussion threads as conversation pools: predicting the growth of discussion threads on reddit." Computational and Mathematical Organization Theory (2021): 1-29. https://www.cse.usf.edu/dsg/data/publications/papers/Data-driven-Studies-on-Social-Networks-Privacy-and-Simulation.pdf
2. NG, Kin, et al. "Social-Media Activity Forecasting with Exogenous Information Signals", Advances in Social Network Analysis and Mining (ASONAM 2021) https://arxiv.org/pdf/2109.11024
3. Horawalavithana, Sameera. "Data-driven Studies on Social Networks: Privacy and Simulation" Chap. 5. Ph.D. Dissertation, University of South Florida, 2021 https://www.cse.usf.edu/dsg/data/publications/papers/Data-driven-Studies-on-Social-Networks-Privacy-and-Simulation.pdf

## Acknowledgments
This work is supported by the DARPA SocialSim Program and the Air Force Research Laboratory under contract FA8650-18-C-7825.
 
