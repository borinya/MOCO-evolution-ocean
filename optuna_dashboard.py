import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

# Load your study
study = optuna.load_study(study_name='your_study_name', storage='sqlite:///app/MoCo/MOCOv3-MNIST/transformer_optuna.db')

# Plot optimization history
fig1 = plot_optimization_history(study)
fig1.show()

# Plot parameter importances
fig2 = plot_param_importances(study)
fig2.show()

