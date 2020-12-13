# MCAS
Multiplatform Cascade Simulator

# Prediction Module

source activate pnnl_socialsim<br>
python run_predictions.py --config ./metadata/configs/seed_predictions.json

## Directory Hierarchy:

```bash
ml_input
└── cpec
    ├── cpec_infoids.txt
    └── twitter
        ├── global
        ├── local
        │   ├── news_counts_overtime_day.pkl.gz
        │   └── reddit_counts_overtime_day.pkl.gz
        └── target
            ├── activated_user
            ├── response
            └── seed
                └── cpec_tw_seeds_overtime_day.pkl.gz
```

```bash
ml_output
└── cpec
    └── twitter
        └── seed
            └── MLPRegressor_2020-06-16_2020-06-29_v1_prev_day
                ├── best_hyper_parameters.pkl.gz
                ├── best_model.h5
                ├── Gperformance.pkl.gz
                ├── gt_data.pkl.gz
                └── simulations_data.pkl.gz
```

## Config File Parameters
- VERSION_TAG: unique file name for the simulation output file.
- DOMAIN: (e.g., wh, vz, cpec)
- PLATFORM: sm platform to predict
- PREDICTION_TYPE: either seed, response, activated_user
- start_train_period: start date for training data
- end_train_period: end date for training data
- start_sim_period: start date for simulation
- end_sim_period: end date for simulation
- time_lag: for previous day (days:1 and hours:0). Currently tested for previous day only.
- FEATURES_PATH: 
  + GLOBAL: path to global features (not per infoID). It can accept list of paths instead.
  + LOCAL: path to per infoID features. Can accept list of paths instead.
  + TARGET: path to dataframe consisting of target variable
- INFORMATION_IDS: path to information IDs
- MODEL_PARAMS: dictionary of ML model keys and parameters values. Note if parameter includes a list of tuples, it must be converted to list of lists

Available ML models: ['RandomForestRegressor','ExtraTreesRegressor','BaggingRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'GaussianProcessRegressor', 'IsotonicRegression', 'ARDRegression', 'HuberRegressor', 'LinearRegression', 'LogisticRegression', 'LogisticRegressionCV', 'PassiveAggressiveRegressor', 'SGDRegressor', 'TheilSenRegressor', 'RANSACRegressor', 'KNeighborsRegressor', 'RadiusNeighborsRegressor', 'MLPRegressor', 'DecisionTreeRegressor', 'ExtraTreeRegressor']

The script can test different models at once if multiple models are defined in the config files along with their corresponding parameters for optimization.
 
