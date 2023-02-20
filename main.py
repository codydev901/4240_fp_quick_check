import pandas as pd
import numpy as np
import json

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import plotly.express as px


def load_allocate_data():
    """
    Loads raw data, separates into a 60/20/20 train/validation/test random split
    in regards to input/target.
    """

    # Read In
    raw_df = pd.read_csv("data/winequality-white.csv", delimiter=";")
    print(raw_df.info())
    print(raw_df.head())

    # 60/20/20 Train/Validation/Test Split
    np.random.seed(1337)  # Set seed for reproducibility
    all_indices = list(range(len(raw_df)))  # Get a list of all indices
    np.random.shuffle(all_indices)          # Shuffle it

    train_index_cutoff = int(.60 * len(all_indices))    # Take first 60% for train
    twenty_percent_count = int(.20 * len(all_indices))  # Then two other ~20% groups

    # Set Allocations
    train_indices = all_indices[:train_index_cutoff]
    validation_indices = all_indices[train_index_cutoff:train_index_cutoff+twenty_percent_count]
    test_indices = all_indices[train_index_cutoff+twenty_percent_count:]

    print("Train/Validation/Test Allocation Counts")
    print(len(train_indices))
    print(len(validation_indices))
    print(len(test_indices))
    print(len(train_indices) + len(validation_indices) + len(test_indices), len(raw_df))

    train_input = []
    train_target = []
    validation_input = []
    validation_target = []
    test_input = []
    test_target = []

    input_keys = raw_df.columns[:-1]
    target_keys = raw_df.columns[-1:]
    print(input_keys)
    print(target_keys)
    # Iterate and Assign Train/Target
    for i, row in raw_df.iterrows():
        input_data = [row[k] for k in input_keys]
        target_data = [row[k] for k in target_keys]
        if i in train_indices:
            train_input.append(input_data)
            train_target.append(target_data)
        elif i in validation_indices:
            validation_input.append(input_data)
            validation_target.append(target_data)
        else:
            test_input.append(input_data)
            test_target.append(target_data)

    # Write For Easy Access
    data_obj = {"train_input": train_input,
                "train_target": train_target,
                "validation_input": validation_input,
                "validation_target": validation_target,
                "test_input": test_input,
                "test_target": test_target}

    with open("data/quick_data.json", "w") as w_file:
        json.dump(data_obj, w_file)


def build_train_mlp():
    """
    Very quick MLP model w/ callback to save on improvement
    in validation acc
    """

    with open("data/quick_data.json", "r") as r_file:
        info_json = json.load(r_file)

    input_dim = len(info_json["train_input"][0])
    target_dim = len(info_json["train_target"][0])

    model = Sequential()
    model.add(Dense(input_dim, activation="relu"))
    model.add(Dense(input_dim*3, activation="relu"))
    model.add(Dense(target_dim, activation="linear"))

    model.compile(optimizer="nadam", loss="mae")

    train_input = np.array(info_json["train_input"])
    train_target = np.array(info_json["train_target"])
    validation_input = np.array(info_json["validation_input"])
    validation_target = np.array(info_json["validation_target"])

    # Save model on improve of val_loss w/ Callback
    es_c = ModelCheckpoint(filepath="models/quick_test.h5",
                           save_best_only=True,
                           save_weights_only=False,
                           verbose=1)

    model.fit(x=train_input, y=train_target,
              validation_data=(validation_input, validation_target),
              batch_size=4,
              epochs=500,
              verbose=1,
              callbacks=[es_c])


def check_calc_predictions():

    with open("data/quick_data.json", "r") as r_file:
        info_json = json.load(r_file)

    train_input = np.array(info_json["train_input"])
    validation_input = np.array(info_json["validation_input"])
    test_input = np.array(info_json["test_input"])

    model = load_model("models/quick_test.h5")

    train_pred = model.predict(train_input)
    validation_pred = model.predict(validation_input)
    test_pred = model.predict(test_input)

    res = [["uuid", "temp_x", "allocation_id", "allocation", "prediction", "ground_truth", "abs_error"]]

    count = 0
    for i, train_pred in enumerate(train_pred):
        allocation = "train"
        g_t = info_json["train_target"][i][0]
        pred_v = train_pred[0]
        abs_error = abs(g_t - pred_v)
        res.append([f"{allocation}_{i}", count, i, allocation, pred_v, g_t, abs_error])
        count += 1

    for i, val_pred in enumerate(validation_pred):
        allocation = "validation"
        g_t = info_json["validation_target"][i][0]
        pred_v = val_pred[0]
        abs_error = abs(g_t - pred_v)
        res.append([f"{allocation}_{i}", count, i, allocation, pred_v, g_t, abs_error])
        count += 1

    for i, test_pred in enumerate(test_pred):
        allocation = "test"
        g_t = info_json["test_target"][i][0]
        pred_v = test_pred[0]
        abs_error = abs(g_t - pred_v)
        res.append([f"{allocation}_{i}", count, i, allocation, pred_v, g_t, abs_error])
        count += 1

    res = pd.DataFrame(data=res[1:], columns=res[0])
    res.to_csv("data/prediction_results.csv", index=False)


def quick_plot():

    df = pd.read_csv("data/prediction_results.csv")

    fig = px.scatter(data_frame=df,
                     x="ground_truth", y="prediction", color="allocation",
                     facet_col="allocation", facet_col_wrap=1,
                     trendline="ols", title="Wine Quality Quick Test"
                     )

    fig.show()


def main():

    # load_allocate_data()

    # build_train_mlp()

    check_calc_predictions()

    quick_plot()


if __name__ == "__main__":

    main()
