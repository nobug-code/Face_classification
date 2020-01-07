import pandas as pd
test_csv_path = "/tmp/pycharm/files/test_vision.csv"
tmp = pd.read_csv(test_csv_path, header=None)
tmp["label"] = -1
tmp.to_csv("/tmp/pycharm_project_917/files/test_vision.csv", index=False, header=False)
