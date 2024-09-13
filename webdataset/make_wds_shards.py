import json
import os
import yaml
import webdataset as wds

from tqdm import tqdm

data_path = 'stage1.5-a.yaml.debug'
data_path = 'stage1.5-a.yaml'
with open(data_path, "r") as file:
	yaml_data = yaml.safe_load(file)
	datasets = yaml_data.get("datasets")
	dataset_paths = [dataset.get("json_path") for dataset in datasets]
print(f"{dataset_paths}")

output = 'stage1.5-a'
output = 'stage1.5-a-debug'
output = 'stage1.5-a-uniqid'
if not os.path.exists(output):
    os.mkdir(output)

with wds.ShardWriter(os.path.join(output, 'llava-ov-%d.tar'), maxcount=10000) as shard_writer:
    global_id = 0
    for json_file in dataset_paths:
      # Load data
      with open(json_file, 'r') as f:
        data = json.load(f)

        for entry in tqdm(data):
          sample = {
            "__key__": str(global_id),
            "id": str(global_id),
            "conversations": json.dumps(entry['conversations']).encode("utf-8"),
          }
          if 'image' in entry:
            sample['image'] = entry['image']
          shard_writer.write(sample)
          global_id += 1

      print(f"Dataset {json_file} successfully converted to wds")

