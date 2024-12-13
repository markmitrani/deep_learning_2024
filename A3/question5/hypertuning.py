import optuna
from q5_MLP import objective_MLP
from q5_RNN import objective_RNN
from q5_LSTM import objective_LSTM


def main():
    
    # for MLP
    study_mlp = optuna.create_study(direction="maximize")
    study_mlp.optimize(objective_MLP, n_trials=15)
    print("---------------------- MLP ----------------------")
    print(f"Best Parameters: {study_mlp.best_params}")
    print(f"Best Value: {study_mlp.best_value}")
    print("-------------------------------------------------\n")
    
    # RNN
    print("---------------------- RNN ----------------------")
    study_rnn = optuna.create_study(direction="maximize")
    study_rnn.optimize(objective_RNN, n_trials=15)
    print(f"Best Parameters: {study_rnn.best_params}")
    print(f"Best Value: {study_rnn.best_value}")
    print("-------------------------------------------------\n")
    
    # LSTM
    print("---------------------- LSTM ----------------------")
    study_lstm = optuna.create_study(direction="maximize")
    study_lstm.optimize(objective_LSTM, n_trials=15)
    print(f"Best Parameters: {study_lstm.best_params}")
    print(f"Best Value: {study_lstm.best_value}")
    print("--------------------------------------------------\n")
    

if __name__ == "__main__":
    main()
