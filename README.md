- pip install -r requirements.txt

```bash
cd $project_path
export PYTHONPATH="$project_path:$PYTHONPATH"
python3 scripts/preprocessing.py

# global model
python3 global_/gen_train_data.py
python3 global_/global_model.py
python3 global_/prepare_local_data.py

# local model
python3 local/gae/train.py

# estimate cluster size
python3 cluster_size/count.py
```
